"""
Build affordance prior: statistics of GT contact-part distribution
for each (category, intent) pair from the OakInk training set.

Output: a JSON file mapping (category, intent) -> part contact distribution,
used by the affordance accuracy metric to replace hand-crafted rules.

Usage:
    python scripts/build_affordance_prior.py \
        --oakink_dir oakink \
        --output assets/affordance_prior.json \
        --split train \
        [--contact_thresh 0.004] \
        [--n_surface_points 4096] \
        [--expected_part_ratio 0.1]

Approach (coordinate-safe):
  1. Load the OIShape dataset (GT hand vertices + aligned object meshes).
  2. For each sample:
     a. Sample surface points from the aligned object mesh (same coord system as hand).
     b. Assign part labels to surface points via nearest-neighbor to OakBase part vertices.
     c. Compute hand-to-surface-point distances to find contact points.
     d. Count contacts per part.
  3. Aggregate by (category, intent) and save.

Key insight: The actual distance computation is between hand vertices and
the aligned object surface (same coordinate system). OakBase parts are only
used for labeling via nearest-neighbor, which is robust to mild coordinate
offsets between OakBase parts and aligned meshes.
"""

import argparse
import json
import os
import sys
import numpy as np
import trimesh
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.metrics.affordance_accuracy import (
    OakBasePartLoader,
    assign_part_labels,
    oakbase_cat,
)
from lib.datasets.utils import (
    ALL_CAT,
    ALL_INTENT,
    get_obj_path,
)


# ---------------------------------------------------------------------------
# Minimal config object to instantiate OIShape
# ---------------------------------------------------------------------------
class _Cfg:
    """Minimal config to drive OIShape.__init__."""

    def __init__(self, split="train"):
        self.DATA_SPLIT = split
        self.OBJ_CATES = "all"
        self.INTENT_MODE = ["use", "hold", "liftup", "handover"]
        # augmentation disabled
        self.AUG_RIGID_P = 0.0


# ---------------------------------------------------------------------------
# Core: compute part contact counts for one sample
# ---------------------------------------------------------------------------
def compute_sample_contacts(
    hand_verts: np.ndarray,          # (V, 3) GT hand vertices (aligned coord)
    obj_surface_pts: np.ndarray,     # (N, 3) sampled from aligned mesh
    part_labels: np.ndarray,         # (N,) part id for each surface point
    n_parts: int,
    contact_thresh: float = 0.004,
) -> np.ndarray:
    """
    Compute contact counts per part for a single sample.

    1. For each object surface point, compute min distance to hand vertices.
    2. Mark surface points with distance <= contact_thresh as contacts.
    3. Count contacts per part.

    Returns: contact_counts (n_parts,) int array
    """
    N = obj_surface_pts.shape[0]
    V = hand_verts.shape[0]

    # Compute distances: obj surface -> hand (min over hand verts)
    if N * V < 50_000_000:
        diff = obj_surface_pts[:, None, :] - hand_verts[None, :, :]  # (N, V, 3)
        dist_sq = (diff ** 2).sum(axis=-1)  # (N, V)
        min_dist = np.sqrt(dist_sq.min(axis=1))  # (N,)
    else:
        min_dist = np.full(N, np.inf, dtype=np.float32)
        chunk = max(1, 50_000_000 // V)
        for i in range(0, N, chunk):
            end = min(i + chunk, N)
            diff = obj_surface_pts[i:end, None, :] - hand_verts[None, :, :]
            dist_sq = (diff ** 2).sum(axis=-1)
            min_dist[i:end] = np.sqrt(dist_sq.min(axis=1))

    # Find contact points
    contact_mask = min_dist <= contact_thresh
    if contact_mask.sum() == 0:
        return np.zeros(n_parts, dtype=np.int64)

    contact_part_labels = part_labels[contact_mask]
    return np.bincount(contact_part_labels, minlength=n_parts).astype(np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build affordance prior from GT training data")
    parser.add_argument("--oakink_dir", type=str, default="/home/kingston/wzc/data/OakInk",
                        help="Root of OakInk dataset (contains OakBase/ and shape/)")
    parser.add_argument("--output", type=str, default="assets/affordance_prior.json",
                        help="Output JSON path")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"],
                        help="Which data split to use for building prior")
    parser.add_argument("--contact_thresh", type=float, default=0.004,
                        help="Contact distance threshold in meters (default: 4mm)")
    parser.add_argument("--n_surface_points", type=int, default=4096,
                        help="Number of surface points to sample per object (default: 4096)")
    parser.add_argument("--expected_part_ratio", type=float, default=0.10,
                        help="Min fraction of total contacts for a part to be 'expected' (default: 0.10)")
    args = parser.parse_args()

    # Paths
    oakbase_dir = os.path.join(args.oakink_dir, "OakBase")
    shape_dir = os.path.join(args.oakink_dir, "shape")
    meta_dir = os.path.join(shape_dir, "metaV2")
    data_dir = shape_dir

    assert os.path.isdir(oakbase_dir), f"OakBase not found: {oakbase_dir}"
    assert os.path.isdir(meta_dir), f"metaV2 not found: {meta_dir}"

    # Set environment variable for OIShape
    os.environ["OAKINK_DIR"] = args.oakink_dir

    # Initialize dataset
    print(f"Loading OIShape (split={args.split}, all categories, all intents)...")
    from lib.datasets.oishape_dataset import OIShape
    cfg = _Cfg(split=args.split)
    dataset = OIShape(cfg)
    print(f"  Total samples: {len(dataset)}")

    # Initialize part loader
    part_loader = OakBasePartLoader(oakbase_dir, meta_dir)

    # ---------------------------------------------------------------------------
    # Cache: per-object precomputation
    #   obj_id -> {
    #       "parts": list of {"name", "attr", "verts"},
    #       "obj_mesh": trimesh mesh (aligned + centered),
    #       "surface_pts": (N,3),
    #       "part_labels": (N,),
    #       "n_parts": int,
    #   }
    # ---------------------------------------------------------------------------
    obj_cache = {}

    def get_obj_data(obj_id, cate_id):
        """Load and cache per-object data: aligned mesh + part labels on surface."""
        if obj_id in obj_cache:
            return obj_cache[obj_id]

        # Load OakBase parts (original coord system)
        parts = part_loader.load_parts(obj_id)
        if parts is None:
            obj_cache[obj_id] = None
            return None

        # Load aligned object mesh (same as OIShape.get_obj_mesh)
        try:
            obj_path = get_obj_path(obj_id, data_dir, meta_dir, use_downsample=True)
            obj_mesh = trimesh.load(obj_path, process=False, force="mesh", skip_materials=True)
        except Exception:
            obj_cache[obj_id] = None
            return None

        # Apply bbox centering (same as OIShape)
        bbox_center = (obj_mesh.vertices.min(0) + obj_mesh.vertices.max(0)) / 2.0
        obj_mesh.vertices = obj_mesh.vertices - bbox_center

        # Sample surface points from the aligned+centered mesh
        sample_result = trimesh.sample.sample_surface(obj_mesh, args.n_surface_points)
        surface_pts = np.asarray(sample_result[0], dtype=np.float32)

        # Assign part labels to surface points.
        # OakBase parts are in original coords; surface_pts are in centered coords.
        # We center the part verts with the same bbox_center for label assignment.
        centered_parts = []
        for p in parts:
            centered_parts.append({
                "name": p["name"],
                "attr": p["attr"],
                "verts": (p["verts"] - bbox_center).astype(np.float32),
            })

        part_labels = assign_part_labels(surface_pts, centered_parts)

        result = {
            "parts": centered_parts,
            "surface_pts": surface_pts,
            "part_labels": part_labels,
            "n_parts": len(parts),
        }
        obj_cache[obj_id] = result
        return result

    # ---------------------------------------------------------------------------
    # Accumulate contact statistics
    # ---------------------------------------------------------------------------
    stats = defaultdict(lambda: defaultdict(lambda: None))

    skipped_no_parts = 0
    skipped_no_hand = 0
    skipped_no_contact = 0
    processed = 0

    action_id_to_intent = {v: k for k, v in ALL_INTENT.items()}

    for idx in tqdm(range(len(dataset)), desc="Building prior"):
        grasp = dataset.grasp_list[idx]

        hand_verts = grasp.get("hand_verts")
        if hand_verts is None:
            skipped_no_hand += 1
            continue
        hand_verts = np.asarray(hand_verts, dtype=np.float32)

        obj_id = grasp["obj_id"]
        cate_id = grasp["cate_id"]
        action_id = grasp["action_id"]
        intent_name = action_id_to_intent.get(action_id, None)
        if intent_name is None:
            continue

        # Get object data (parts + labeled surface points)
        obj_data = get_obj_data(obj_id, cate_id)
        if obj_data is None:
            skipped_no_parts += 1
            continue

        # Compute contacts
        contact_counts = compute_sample_contacts(
            hand_verts=hand_verts,
            obj_surface_pts=obj_data["surface_pts"],
            part_labels=obj_data["part_labels"],
            n_parts=obj_data["n_parts"],
            contact_thresh=args.contact_thresh,
        )

        if contact_counts.sum() == 0:
            skipped_no_contact += 1
            # Still count this sample in n_samples for completeness
            # but don't add to contact_counts

        # Aggregate
        oak_cat = oakbase_cat(cate_id)
        entry = stats[oak_cat].get(intent_name)
        if entry is None:
            entry = {
                "n_samples": 0,
                "n_with_contact": 0,
                "part_names": [p["name"] for p in obj_data["parts"]],
                "part_attrs": [p["attr"] for p in obj_data["parts"]],
                "contact_counts": np.zeros(obj_data["n_parts"], dtype=np.int64),
            }
            stats[oak_cat][intent_name] = entry

        if len(contact_counts) == len(entry["contact_counts"]):
            entry["n_samples"] += 1
            if contact_counts.sum() > 0:
                entry["n_with_contact"] += 1
            entry["contact_counts"] += contact_counts

        processed += 1

    print(f"\nProcessed: {processed}")
    print(f"Skipped (no parts):    {skipped_no_parts}")
    print(f"Skipped (no hand):     {skipped_no_hand}")
    print(f"Skipped (no contact):  {skipped_no_contact}")

    # ---------------------------------------------------------------------------
    # Normalize and determine expected parts
    # ---------------------------------------------------------------------------
    prior = {}

    for cat_name, intents in stats.items():
        prior[cat_name] = {}
        for intent_name, entry in intents.items():
            if entry is None or entry["n_samples"] == 0:
                continue

            total_contacts = entry["contact_counts"].sum()
            if total_contacts == 0:
                dist = [0.0] * len(entry["contact_counts"])
            else:
                dist = (entry["contact_counts"] / float(total_contacts)).tolist()

            # Determine expected parts: those with >= expected_part_ratio of total contacts
            expected_parts = [
                i for i, d in enumerate(dist)
                if d >= args.expected_part_ratio
            ]

            # Fallback: if no part reaches the threshold, use the top part
            if not expected_parts and len(dist) > 0:
                expected_parts = [int(np.argmax(dist))]

            prior[cat_name][intent_name] = {
                "n_samples": entry["n_samples"],
                "n_with_contact": entry["n_with_contact"],
                "contact_rate": round(entry["n_with_contact"] / entry["n_samples"], 4),
                "part_names": entry["part_names"],
                "part_attrs": entry["part_attrs"],
                "contact_dist": [round(d, 4) for d in dist],
                "expected_parts": expected_parts,
            }

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(prior, f, indent=2, ensure_ascii=False)

    print(f"\nPrior saved to: {args.output}")

    # Print summary
    print("\n===== Affordance Prior Summary =====")
    for cat_name in sorted(prior.keys()):
        intents = prior[cat_name]
        for intent_name in sorted(intents.keys()):
            entry = intents[intent_name]
            n = entry["n_samples"]
            n_c = entry["n_with_contact"]
            names = entry["part_names"]
            dist = entry["contact_dist"]
            expected = entry["expected_parts"]
            expected_names = [names[i] for i in expected]

            dist_str = ", ".join(f"{names[i]}:{dist[i]:.2f}" for i in range(len(names)))
            print(f"  {cat_name:20s} | {intent_name:10s} | n={n:5d} c={n_c:5d} | {dist_str}")
            print(f"  {'':20s} | {'':10s} |                  expected: {expected_names}")


if __name__ == "__main__":
    main()
