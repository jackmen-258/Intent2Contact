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
        [--expected_strategy adaptive_mean] \
        [--expected_part_ratio 0.10] \
        [--expected_relative_factor 0.5] \
        [--expected_topk 2]

Approach (coordinate-safe):
  1. Load the OIShape dataset (GT hand vertices + aligned object meshes).
  2. For each sample:
     a. Sample surface points from the aligned object mesh (same coord system as hand).
     b. Assign part labels to surface points via nearest-neighbor to OakBase part vertices.
     c. Compute hand-to-surface-point distances to find contact points.
     d. Normalize contact counts to a per-sample distribution, then accumulate.
  3. Aggregate by (category, intent) via sample-average (not contact-point-count-weighted).
  4. Determine expected parts using the selected strategy.
  5. Annotate each (category, intent) entry with discriminability flags.

Key design decisions vs. original:
  [FIX-1] Sample-normalized aggregation:
    Contact counts are normalized per sample BEFORE accumulation, so each grasp
    contributes equally regardless of how many surface points fall within threshold.
    Previously a single large-contact sample could dominate the distribution.

  [FIX-2] Explicit part-count mismatch tracking:
    Samples whose object has a different number of parts than the first registered
    sample for that (cat, intent) pair are now counted explicitly and reported,
    rather than silently dropped with processed++ still incrementing.

  [FIX-3] Handover semantic flag:
    The `handover` intent in OakInk contains grasps from BOTH the giver and the
    receiver, causing the distribution to spread across all parts and making the
    prior near-trivially satisfied. Each entry now carries `is_handover` and
    `has_semantic_ambiguity` flags so downstream metrics can stratify results.

  [FIX-4] Adaptive expected-part selection:
    Three strategies are available via --expected_strategy:
      - "fixed_ratio"   : original behaviour, threshold = expected_part_ratio
      - "adaptive_mean" : threshold = max(expected_part_ratio,
                           1/n_parts * expected_relative_factor)
                          Scales naturally with the number of parts so that a
                          2-part object requires >25% (not >10%) to qualify.
      - "top_k"         : keep at most the top-k parts (--expected_topk), filtered
                          by a minimum ratio (--expected_part_ratio).

  [FIX-5] Intent-discriminability annotation:
    After building the full prior, entries where all intents of the same category
    share identical expected_parts are flagged with `intent_discriminative=False`.
    Evaluation code should report these separately to avoid masking failures.
"""

import argparse
import json
import os
import sys
import warnings
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
        self.AUG_RIGID_P = 0.0


# ---------------------------------------------------------------------------
# Core: compute per-sample normalized part contact distribution
# ---------------------------------------------------------------------------
def compute_sample_contacts(
    hand_verts: np.ndarray,       # (V, 3) GT hand vertices (aligned coord)
    obj_surface_pts: np.ndarray,  # (N, 3) sampled from aligned mesh
    part_labels: np.ndarray,      # (N,) part id per surface point
    n_parts: int,
    contact_thresh: float = 0.004,
) -> tuple:
    """
    Compute per-sample contact distribution (normalized) for a single grasp.

    Returns
    -------
    contact_dist : np.ndarray, shape (n_parts,), float32
        Per-part fraction of contact points.  Sums to 1.0 when there is any
        contact, all-zeros when no contact is detected.
    has_contact : bool
        True iff at least one surface point is within contact_thresh of the hand.
    """
    N = obj_surface_pts.shape[0]
    V = hand_verts.shape[0]

    # Batched distance: obj surface -> closest hand vertex
    if N * V < 50_000_000:
        diff = obj_surface_pts[:, None, :] - hand_verts[None, :, :]  # (N, V, 3)
        dist_sq = (diff ** 2).sum(axis=-1)                            # (N, V)
        min_dist = np.sqrt(dist_sq.min(axis=1))                       # (N,)
    else:
        min_dist = np.full(N, np.inf, dtype=np.float32)
        chunk = max(1, 50_000_000 // V)
        for i in range(0, N, chunk):
            end = min(i + chunk, N)
            diff = obj_surface_pts[i:end, None, :] - hand_verts[None, :, :]
            dist_sq = (diff ** 2).sum(axis=-1)
            min_dist[i:end] = np.sqrt(dist_sq.min(axis=1))

    contact_mask = min_dist <= contact_thresh
    n_contact = int(contact_mask.sum())

    if n_contact == 0:
        return np.zeros(n_parts, dtype=np.float32), False

    contact_part_labels = part_labels[contact_mask]
    raw_counts = np.bincount(contact_part_labels, minlength=n_parts).astype(np.float32)

    # [FIX-1] Normalize per sample so every grasp contributes equally.
    contact_dist = raw_counts / raw_counts.sum()
    return contact_dist, True


# ---------------------------------------------------------------------------
# Expected-part selection strategies
# ---------------------------------------------------------------------------
def select_expected_parts(
    dist: list,
    strategy: str,
    fixed_ratio: float,
    relative_factor: float,
    topk: int,
) -> list:
    """
    Return indices of 'expected' parts given a normalized distribution.

    Parameters
    ----------
    dist           : list of float, length = n_parts, sums to ~1.0
    strategy       : one of {"fixed_ratio", "adaptive_mean", "top_k"}
    fixed_ratio    : minimum fraction threshold (used by fixed_ratio and top_k)
    relative_factor: for adaptive_mean, a part qualifies if
                     dist[i] >= (1/n_parts) * relative_factor
    topk           : maximum number of expected parts for top_k strategy
    """
    n_parts = len(dist)
    if n_parts == 0:
        return []

    dist_arr = np.asarray(dist, dtype=np.float64)

    if strategy == "fixed_ratio":
        # [ORIGINAL] Fixed absolute threshold.
        expected = [i for i, d in enumerate(dist_arr) if d >= fixed_ratio]

    elif strategy == "adaptive_mean":
        # [FIX-4] Threshold scales with the number of parts.
        # For a 2-part object: threshold = max(0.10, 0.5 * 0.5) = 0.25
        # For a 5-part object: threshold = max(0.10, 0.2 * 0.5) = 0.10
        # This prevents all parts of a 2-part object from qualifying at 0.1.
        uniform = 1.0 / n_parts
        adaptive_thresh = max(fixed_ratio, uniform * relative_factor)
        expected = [i for i, d in enumerate(dist_arr) if d >= adaptive_thresh]

    elif strategy == "top_k":
        # [FIX-4] Keep at most top-k parts, but still filter by minimum ratio.
        order = np.argsort(dist_arr)[::-1]
        expected = []
        for i in order[:topk]:
            if dist_arr[i] >= fixed_ratio:
                expected.append(int(i))

    else:
        raise ValueError(f"Unknown expected_strategy: {strategy!r}")

    # Fallback: always include at least the dominant part.
    if not expected and len(dist_arr) > 0:
        expected = [int(np.argmax(dist_arr))]

    return sorted(expected)


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
                        help="Contact distance threshold in metres (default: 4 mm)")
    parser.add_argument("--n_surface_points", type=int, default=4096,
                        help="Surface points to sample per object (default: 4096)")

    # [FIX-4] Strategy arguments
    parser.add_argument("--expected_strategy", type=str,
                        choices=["fixed_ratio", "adaptive_mean", "top_k"],
                        default="adaptive_mean",
                        help="Strategy to select expected parts from the distribution "
                             "(default: adaptive_mean)")
    parser.add_argument("--expected_part_ratio", type=float, default=0.10,
                        help="Minimum contact fraction for a part to be 'expected'. "
                             "Used as lower bound in all strategies (default: 0.10)")
    parser.add_argument("--expected_relative_factor", type=float, default=0.5,
                        help="[adaptive_mean only] Multiplier on the uniform baseline "
                             "1/n_parts. Higher = stricter. (default: 0.5)")
    parser.add_argument("--expected_topk", type=int, default=2,
                        help="[top_k only] Maximum number of expected parts (default: 2)")

    args = parser.parse_args()

    # Validate
    if args.expected_relative_factor <= 0:
        parser.error("--expected_relative_factor must be positive")
    if args.expected_topk < 1:
        parser.error("--expected_topk must be >= 1")

    # Paths
    oakbase_dir = os.path.join(args.oakink_dir, "OakBase")
    shape_dir   = os.path.join(args.oakink_dir, "shape")
    meta_dir    = os.path.join(shape_dir, "metaV2")
    data_dir    = shape_dir

    assert os.path.isdir(oakbase_dir), f"OakBase not found: {oakbase_dir}"
    assert os.path.isdir(meta_dir),    f"metaV2 not found: {meta_dir}"

    os.environ["OAKINK_DIR"] = args.oakink_dir

    print(f"Loading OIShape (split={args.split}, all categories, all intents)...")
    from lib.datasets.oishape_dataset import OIShape
    cfg     = _Cfg(split=args.split)
    dataset = OIShape(cfg)
    print(f"  Total samples: {len(dataset)}")

    part_loader = OakBasePartLoader(oakbase_dir, meta_dir)

    # ------------------------------------------------------------------
    # Object cache: per obj_id  ->  aligned mesh + labeled surface points
    # ------------------------------------------------------------------
    obj_cache = {}

    def get_obj_data(obj_id, cate_id):
        if obj_id in obj_cache:
            return obj_cache[obj_id]

        parts = part_loader.load_parts(obj_id)
        if parts is None:
            obj_cache[obj_id] = None
            return None

        try:
            obj_path = get_obj_path(obj_id, data_dir, meta_dir, use_downsample=True)
            obj_mesh = trimesh.load(obj_path, process=False, force="mesh",
                                    skip_materials=True)
        except Exception:
            obj_cache[obj_id] = None
            return None

        # Align to bbox centre (mirrors OIShape preprocessing)
        bbox_center         = (obj_mesh.vertices.min(0) + obj_mesh.vertices.max(0)) / 2.0
        obj_mesh.vertices  -= bbox_center

        surface_pts = np.asarray(
            trimesh.sample.sample_surface(obj_mesh, args.n_surface_points)[0],
            dtype=np.float32,
        )

        # Centre OakBase part verts with the same offset
        centered_parts = [
            {
                "name": p["name"],
                "attr": p["attr"],
                "verts": (p["verts"] - bbox_center).astype(np.float32),
            }
            for p in parts
        ]

        part_labels = assign_part_labels(surface_pts, centered_parts)

        result = {
            "parts":       centered_parts,
            "surface_pts": surface_pts,
            "part_labels": part_labels,
            "n_parts":     len(parts),
        }
        obj_cache[obj_id] = result
        return result

    # ------------------------------------------------------------------
    # Accumulate contact statistics
    # ------------------------------------------------------------------
    # entry structure per (cat, intent):
    #   n_samples          : int  – total samples attempted for this group
    #   n_with_contact     : int  – samples with >=1 contact point
    #   n_part_mismatch    : int  – [FIX-2] samples skipped due to part-count mismatch
    #   part_names         : list[str]
    #   part_attrs         : list
    #   contact_dist_sum   : np.ndarray(n_parts,) float32
    #                        sum of per-sample normalized distributions [FIX-1]
    stats = defaultdict(lambda: defaultdict(lambda: None))

    skipped_no_parts    = 0
    skipped_no_hand     = 0
    skipped_no_contact  = 0
    skipped_part_mismatch = 0   # [FIX-2]
    processed           = 0

    action_id_to_intent = {v: k for k, v in ALL_INTENT.items()}

    for idx in tqdm(range(len(dataset)), desc="Building prior"):
        grasp = dataset.grasp_list[idx]

        hand_verts = grasp.get("hand_verts")
        if hand_verts is None:
            skipped_no_hand += 1
            continue
        hand_verts = np.asarray(hand_verts, dtype=np.float32)

        obj_id      = grasp["obj_id"]
        cate_id     = grasp["cate_id"]
        action_id   = grasp["action_id"]
        intent_name = action_id_to_intent.get(action_id, None)
        if intent_name is None:
            continue

        obj_data = get_obj_data(obj_id, cate_id)
        if obj_data is None:
            skipped_no_parts += 1
            continue

        # [FIX-1] Per-sample normalized distribution (not raw counts)
        contact_dist, has_contact = compute_sample_contacts(
            hand_verts=hand_verts,
            obj_surface_pts=obj_data["surface_pts"],
            part_labels=obj_data["part_labels"],
            n_parts=obj_data["n_parts"],
            contact_thresh=args.contact_thresh,
        )

        if not has_contact:
            skipped_no_contact += 1

        oak_cat = oakbase_cat(cate_id)
        entry   = stats[oak_cat].get(intent_name)

        if entry is None:
            # Initialise from first valid sample for this (cat, intent) pair
            entry = {
                "n_samples":         0,
                "n_with_contact":    0,
                "n_part_mismatch":   0,    # [FIX-2]
                "part_names":        [p["name"] for p in obj_data["parts"]],
                "part_attrs":        [p["attr"] for p in obj_data["parts"]],
                "contact_dist_sum":  np.zeros(obj_data["n_parts"], dtype=np.float64),
            }
            stats[oak_cat][intent_name] = entry

        # [FIX-2] Explicit mismatch detection with counter (no longer silent)
        if len(contact_dist) != len(entry["contact_dist_sum"]):
            entry["n_part_mismatch"] += 1
            skipped_part_mismatch    += 1
            # Do NOT increment n_samples; this sample does not contribute.
            processed += 1
            continue

        # Accumulate [FIX-1]: add normalised per-sample distribution
        entry["n_samples"]   += 1
        if has_contact:
            entry["n_with_contact"] += 1
            entry["contact_dist_sum"] += contact_dist
            # Zero-contact samples are counted in n_samples but contribute
            # nothing to the distribution (correct: they give no location signal).

        processed += 1

    print(f"\nProcessed:                   {processed}")
    print(f"Skipped (no parts):          {skipped_no_parts}")
    print(f"Skipped (no hand):           {skipped_no_hand}")
    print(f"Skipped (no contact):        {skipped_no_contact}")
    print(f"Skipped (part mismatch):     {skipped_part_mismatch}")   # [FIX-2]

    # ------------------------------------------------------------------
    # Normalise and select expected parts
    # ------------------------------------------------------------------
    prior = {}

    for cat_name, intents in stats.items():
        prior[cat_name] = {}
        for intent_name, entry in intents.items():
            if entry is None or entry["n_samples"] == 0:
                continue

            n_with_contact = entry["n_with_contact"]

            # [FIX-1] Average over samples that actually had contact.
            if n_with_contact > 0:
                dist = (entry["contact_dist_sum"] / float(n_with_contact)).tolist()
            else:
                dist = [0.0] * len(entry["contact_dist_sum"])

            n_parts = len(dist)

            expected_parts = select_expected_parts(
                dist=dist,
                strategy=args.expected_strategy,
                fixed_ratio=args.expected_part_ratio,
                relative_factor=args.expected_relative_factor,
                topk=args.expected_topk,
            )

            # [FIX-3] Flag handover entries explicitly so downstream code can
            # report them separately.  Handover grasps in OakInk contain both
            # the giver's and the receiver's pose, spreading contact across
            # functional and non-functional parts and making the prior looser
            # than for single-role intents.
            is_handover = (intent_name == "handover")

            prior[cat_name][intent_name] = {
                "n_samples":       entry["n_samples"],
                "n_with_contact":  n_with_contact,
                "n_part_mismatch": entry["n_part_mismatch"],  # [FIX-2]
                "contact_rate":    round(n_with_contact / entry["n_samples"], 4),
                "part_names":      entry["part_names"],
                "part_attrs":      entry["part_attrs"],
                "contact_dist":    [round(d, 4) for d in dist],
                "expected_parts":  expected_parts,
                # [FIX-3] Semantic flags
                "is_handover":          is_handover,
                "has_semantic_ambiguity": is_handover,
                # [FIX-5] placeholder; filled in the discriminability pass below
                "intent_discriminative": True,
            }

    # ------------------------------------------------------------------
    # [FIX-5] Annotate intent-discriminability per category
    # ------------------------------------------------------------------
    non_discriminative_cats = []
    for cat_name, intents in prior.items():
        if len(intents) < 2:
            continue
        ep_sets = {
            intent: tuple(sorted(entry["expected_parts"]))
            for intent, entry in intents.items()
        }
        all_same = len(set(ep_sets.values())) == 1
        if all_same:
            non_discriminative_cats.append(cat_name)
            for entry in intents.values():
                entry["intent_discriminative"] = False

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(prior, f, indent=2, ensure_ascii=False)

    print(f"\nPrior saved to: {args.output}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n===== Affordance Prior Summary =====")
    print(f"Expected-part strategy : {args.expected_strategy}")
    if args.expected_strategy == "fixed_ratio":
        print(f"  fixed_ratio          : {args.expected_part_ratio}")
    elif args.expected_strategy == "adaptive_mean":
        print(f"  fixed_ratio (floor)  : {args.expected_part_ratio}")
        print(f"  relative_factor      : {args.expected_relative_factor}")
    else:
        print(f"  top_k                : {args.expected_topk}")
        print(f"  min_ratio            : {args.expected_part_ratio}")

    print(f"\nNon-discriminative categories ({len(non_discriminative_cats)}/{len(prior)}):")
    print(f"  {non_discriminative_cats}")

    if skipped_part_mismatch > 0:
        print(f"\n⚠  {skipped_part_mismatch} samples skipped due to part-count mismatch.")
        print("   These samples were NOT counted in n_samples.")
        print("   Check whether multiple OakBase instances exist for the same category "
              "with different part topologies.")

    print()
    for cat_name in sorted(prior.keys()):
        intents = prior[cat_name]
        for intent_name in sorted(intents.keys()):
            entry       = intents[intent_name]
            n           = entry["n_samples"]
            n_c         = entry["n_with_contact"]
            n_mm        = entry["n_part_mismatch"]
            names       = entry["part_names"]
            dist        = entry["contact_dist"]
            expected    = entry["expected_parts"]
            exp_names   = [names[i] for i in expected]
            disc_flag   = "" if entry["intent_discriminative"] else " [non-discrim]"
            ho_flag     = " [handover⚠]" if entry["is_handover"] else ""
            mm_flag     = f" [mismatch={n_mm}]" if n_mm > 0 else ""

            dist_str = ", ".join(f"{names[i]}:{dist[i]:.2f}" for i in range(len(names)))
            print(
                f"  {cat_name:20s} | {intent_name:10s} | "
                f"n={n:5d} c={n_c:5d}{disc_flag}{ho_flag}{mm_flag}"
            )
            print(f"  {'':20s} | {'':10s} | dist: {dist_str}")
            print(f"  {'':20s} | {'':10s} | expected: {exp_names}")


if __name__ == "__main__":
    main()