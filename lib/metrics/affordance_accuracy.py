"""
Affordance Accuracy Metric — model-free intent evaluation.

Evaluates whether generated grasps contact the correct functional regions
of the object, based on OakBase part segmentation annotations.

Metrics:
  - Contact-Valid : whether valid contact exists (d <= alpha_max)
  - Aff-Acc       : contacted parts exactly match the expected parts
  - Aff-Coverage  : contacted parts cover all expected parts

The "expected functional region" for each (category, intent) pair is derived
from ground-truth training data statistics, stored as a JSON prior file.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# OakBase category name mapping (yoda <-> OakBase directory names differ)
# ---------------------------------------------------------------------------
YODA_TO_OAKBASE = {
    "cameras":        "camera",
    "fryingpan":      "frying_pan",
    "gamecontroller": "game_controller",
    "lotion_pump":    "lotion_bottle",
    "scissors":       "scissor",
    "squeezable":     "squeeze_tube",
    "pen":            "marker",
}


def oakbase_cat(yoda_cat: str) -> str:
    """Convert yoda category name to OakBase directory name."""
    return YODA_TO_OAKBASE.get(yoda_cat, yoda_cat)


# ---------------------------------------------------------------------------
# Part loading utilities
# ---------------------------------------------------------------------------
class OakBasePartLoader:
    """
    Loads and caches OakBase part segmentation data.
    Each object has multiple parts, each with:
      - a PLY mesh  (part geometry)
      - a JSON file (part name + affordance attributes)
    """

    def __init__(self, oakbase_dir: str, meta_dir: str):
        """
        Args:
            oakbase_dir: path to OakBase root (e.g. oakink/OakBase)
            meta_dir:    path to metaV2 (e.g. oakink/shape/metaV2)
        """
        self.oakbase_dir = oakbase_dir
        self.meta_dir = meta_dir

        # Load metadata for obj_id -> obj_name mapping
        with open(os.path.join(meta_dir, "object_id.json"), "r") as f:
            self.real_meta = json.load(f)
        with open(os.path.join(meta_dir, "virtual_object_id.json"), "r") as f:
            self.virtual_meta = json.load(f)
        with open(os.path.join(meta_dir, "yodaobject_cat.json"), "r") as f:
            self.yoda_cat = json.load(f)

        # Cache: obj_id -> { "parts": [ {name, attr, verts}, ... ] }
        self._cache: Dict[str, dict] = {}

    def _resolve_obj(self, obj_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve obj_id to (category_name, obj_name).
        Returns (None, None) if not found.
        """
        # Look up in both real and virtual meta
        info = (self.real_meta.get(obj_id) or
                self.virtual_meta.get(obj_id) or
                self.real_meta.get(obj_id.upper()) or
                self.virtual_meta.get(obj_id.lower()))
        if info is None:
            return None, None

        obj_name = info["name"]
        # Category prefix: digits [1:3] of obj_id
        prefix = obj_id[1:3]
        yoda_name = self.yoda_cat.get(prefix)
        if yoda_name is None:
            return None, None

        cat_name = oakbase_cat(yoda_name)
        return cat_name, obj_name

    def load_parts(self, obj_id: str) -> Optional[List[dict]]:
        """
        Load part info for an object.
        Returns list of dicts: [{"name": str, "attr": [str], "verts": ndarray(V,3)}, ...]
        Returns None if parts are unavailable.
        """
        if obj_id in self._cache:
            return self._cache[obj_id]

        cat_name, obj_name = self._resolve_obj(obj_id)
        if cat_name is None:
            self._cache[obj_id] = None
            return None

        part_dir = os.path.join(self.oakbase_dir, cat_name, obj_name)
        if not os.path.isdir(part_dir):
            self._cache[obj_id] = None
            return None

        # Find all part_XX.json files
        json_files = sorted([f for f in os.listdir(part_dir) if f.endswith(".json")])
        if not json_files:
            self._cache[obj_id] = None
            return None

        parts = []
        for jf in json_files:
            with open(os.path.join(part_dir, jf), "r") as f:
                part_meta = json.load(f)

            # Corresponding PLY file
            ply_file = jf.replace(".json", ".ply")
            ply_path = os.path.join(part_dir, ply_file)
            if not os.path.exists(ply_path):
                continue

            # Load PLY vertices (lightweight, no trimesh dependency)
            verts = _load_ply_vertices(ply_path)
            if verts is None:
                continue

            parts.append({
                "name": part_meta.get("name", ""),
                "attr": part_meta.get("attr", []),
                "verts": verts,
            })

        result = parts if parts else None
        self._cache[obj_id] = result
        return result

    def get_category(self, obj_id: str) -> Optional[str]:
        """Return the yoda category name for an object."""
        prefix = obj_id[1:3]
        return self.yoda_cat.get(prefix)


def _load_ply_vertices(ply_path: str) -> Optional[np.ndarray]:
    """
    Load vertices from a PLY file.
    Tries trimesh first; falls back to a manual PLY parser
    that handles both ASCII and binary formats.
    """
    try:
        import trimesh
        mesh = trimesh.load(ply_path, process=False, force="mesh")
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        if verts.shape[0] > 0:
            return verts
    except Exception:
        pass

    # Fallback: manual PLY parser (handles ASCII + binary_little_endian)
    try:
        return _parse_ply(ply_path)
    except Exception:
        return None


def _parse_ply(ply_path: str) -> Optional[np.ndarray]:
    """
    Minimal PLY vertex reader supporting both ASCII and binary_little_endian.
    Only extracts the first 3 properties (x, y, z) per vertex.
    """
    import struct

    n_verts = 0
    is_binary = False
    prop_types = []   # list of (dtype_char, byte_size) per vertex property
    header_end_offset = 0

    # --- Parse header ---
    with open(ply_path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                return None
            line_str = line.decode("ascii", errors="replace").strip()

            if line_str.startswith("format binary_little_endian"):
                is_binary = True
            elif line_str.startswith("format ascii"):
                is_binary = False
            elif line_str.startswith("element vertex"):
                n_verts = int(line_str.split()[-1])
            elif line_str.startswith("property"):
                parts = line_str.split()
                # parts: ["property", type, name]
                ptype = parts[1] if len(parts) >= 3 else "float"
                if ptype in ("float", "float32"):
                    prop_types.append(("f", 4))
                elif ptype in ("double", "float64"):
                    prop_types.append(("d", 8))
                elif ptype in ("int", "int32"):
                    prop_types.append(("i", 4))
                elif ptype in ("uint", "uint32"):
                    prop_types.append(("I", 4))
                elif ptype in ("short", "int16"):
                    prop_types.append(("h", 2))
                elif ptype in ("ushort", "uint16"):
                    prop_types.append(("H", 2))
                elif ptype in ("char", "int8"):
                    prop_types.append(("b", 1))
                elif ptype in ("uchar", "uint8"):
                    prop_types.append(("B", 1))
                else:
                    prop_types.append(("f", 4))  # fallback
            elif line_str == "end_header":
                header_end_offset = f.tell()
                break

    if n_verts == 0:
        return None

    # --- Read vertices ---
    if is_binary:
        # Binary little-endian
        vertex_size = sum(s for _, s in prop_types)
        fmt = "<" + "".join(c for c, _ in prop_types)

        with open(ply_path, "rb") as f:
            f.seek(header_end_offset)
            raw = f.read(vertex_size * n_verts)

        if len(raw) < vertex_size * n_verts:
            return None

        verts = np.empty((n_verts, 3), dtype=np.float64)
        for i in range(n_verts):
            vals = struct.unpack_from(fmt, raw, i * vertex_size)
            verts[i, 0] = vals[0]  # x
            verts[i, 1] = vals[1]  # y
            verts[i, 2] = vals[2]  # z

        return verts.astype(np.float32)

    else:
        # ASCII
        verts = []
        with open(ply_path, "r") as f:
            # Skip header
            for line in f:
                if line.strip() == "end_header":
                    break
            # Read vertices
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(verts) >= n_verts:
                    break

        if not verts:
            return None
        return np.array(verts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Part label assignment for object surface points
# ---------------------------------------------------------------------------
def assign_part_labels(
    obj_surface_pts: np.ndarray,    # (N, 3) sampled object surface points
    parts: List[dict],              # from OakBasePartLoader.load_parts()
    max_part_verts: int = 2000,     # subsample each part to at most this many verts
) -> np.ndarray:
    """
    Assign each surface point to the nearest part using KDTree.
    Returns: part_ids (N,) int array, values in [0, num_parts).
    """
    from scipy.spatial import cKDTree

    # Build a combined array of all part vertices with labels
    # Subsample dense parts to keep the KDTree small
    all_verts = []
    all_labels = []
    for pid, part in enumerate(parts):
        v = part["verts"]
        if v is None or len(v) == 0:
            continue
        # Subsample if too many vertices (stride-based, fast and deterministic)
        if len(v) > max_part_verts:
            stride = max(1, len(v) // max_part_verts)
            v = v[::stride]
        all_verts.append(v)
        all_labels.append(np.full(len(v), pid, dtype=np.int32))

    if not all_verts:
        return np.zeros(obj_surface_pts.shape[0], dtype=np.int32)

    all_verts = np.concatenate(all_verts, axis=0)   # (M, 3)
    all_labels = np.concatenate(all_labels, axis=0)  # (M,)

    # KDTree nearest-neighbor: O(N * log M) instead of O(N * M)
    tree = cKDTree(all_verts)
    _, nearest_idx = tree.query(obj_surface_pts, k=1)

    return all_labels[nearest_idx]


# ---------------------------------------------------------------------------
# Intent -> functional region mapping
# ---------------------------------------------------------------------------

# Attributes that define "functional" (non-grasping) parts
FUNCTIONAL_ATTRS = {
    "cut_sth", "stab_sth", "shear_sth",               # blades
    "contain_sth", "flow_in_sth", "flow_out_sth",      # containers
    "illuminate_sth", "point_to_sth",                   # flashlight/tools
    "knock_sth",                                        # hammer head
    "tighten_sth", "loosen_sth",                        # screwdriver tip
    "brush_sth",                                        # toothbrush head
    "spray_sth", "pump_out_sth", "trigger_sth",         # sprayers
    "observe_sth",                                      # binoculars lens
    "clamp_sth",                                        # pincer jaws
    "control_sth", "pressed/unpressed_by_hand",         # buttons/controls
    "attach_to_(ear)", "attach_to_(eyes)",              # wearables
    "attach_to_(head)", "attach_to_(top_of_ear)",
}

GRASP_ATTRS = {"held_by_hand"}


class AffordancePrior:
    """
    Loads and queries pre-computed GT contact-part distribution prior.
    Built by scripts/build_affordance_prior.py.
    """

    def __init__(self, prior_path: str):
        with open(prior_path, "r", encoding="utf-8") as f:
            self._prior = json.load(f)

    def get_expected_parts(
        self, category: str, intent: str, part_names: List[str]
    ) -> Optional[List[int]]:
        """
        Look up expected parts from the GT prior.

        Args:
            category: OakBase category name (e.g., "mug")
            intent:   intent name (e.g., "use")
            part_names: list of part names for this object

        Returns:
            List of expected part indices, or None if not found in prior.
        """
        cat_entry = self._prior.get(category)
        if cat_entry is None:
            return None
        intent_entry = cat_entry.get(intent)
        if intent_entry is None:
            return None

        # Match by part names: the prior stores part_names for the
        # representative object of this category. If the current object
        # has the same number of parts, use the prior's expected_parts directly.
        prior_names = intent_entry.get("part_names", [])
        expected = intent_entry.get("expected_parts", [])

        if len(prior_names) == len(part_names):
            return expected

        # If part count differs (e.g., different variant), try to match by name.
        # Map prior expected part names -> indices in current object's parts.
        expected_names = {prior_names[i] for i in expected if i < len(prior_names)}
        matched = [i for i, n in enumerate(part_names) if n in expected_names]
        return matched if matched else None

    def get_contact_dist(self, category: str, intent: str) -> Optional[List[float]]:
        """Return the GT contact distribution for (category, intent)."""
        cat_entry = self._prior.get(category)
        if cat_entry is None:
            return None
        intent_entry = cat_entry.get(intent)
        if intent_entry is None:
            return None
        return intent_entry.get("contact_dist")


def get_expected_parts_for_intent(
    parts: List[dict],
    intent: str,
    prior: Optional[AffordancePrior] = None,
    category: Optional[str] = None,
) -> List[int]:
    """
    Determine which part indices are expected contact regions for a given intent.

    Priority:
      1. If a GT prior is provided and has an entry for (category, intent),
         use the data-driven expected parts.
      2. Otherwise, fall back to attribute-based rules.

    Attribute-based rules:
      - "use":      hand should contact parts with "held_by_hand" attribute
                     (i.e. handle/grip), leaving functional parts free.
      - "hold":     hand can contact any "held_by_hand" part (body or handle).
      - "liftup":   similar to "hold" — stable grasp on body or handle.
      - "handover": hand should contact NON-functional parts, leaving functional
                     parts accessible to the receiver.
    """
    n_parts = len(parts)
    if n_parts == 0:
        return []

    # --- Priority 1: GT prior ---
    if prior is not None and category is not None:
        part_names = [p.get("name", "") for p in parts]
        expected = prior.get_expected_parts(category, intent, part_names)
        if expected is not None and len(expected) > 0:
            return expected

    # --- Priority 2: attribute-based rules ---

    # Classify each part
    has_grasp = []      # parts with "held_by_hand"
    has_functional = []  # parts with functional attributes
    has_no_func = []     # parts with "no_function"

    for pid, part in enumerate(parts):
        attrs = set(part.get("attr", []))
        if attrs & GRASP_ATTRS:
            has_grasp.append(pid)
        if attrs & FUNCTIONAL_ATTRS:
            has_functional.append(pid)
        if "no_function" in attrs:
            has_no_func.append(pid)

    if intent == "use":
        # "use" intent: hand should be on the grip/handle to operate the tool.
        # Prefer parts with "held_by_hand" that are NOT purely functional.
        grasp_only = [p for p in has_grasp if p not in has_functional]
        if grasp_only:
            return grasp_only
        # If handle is also functional (e.g., cup body is both "held_by_hand"
        # and "contain_sth"), still allow it
        if has_grasp:
            return has_grasp
        # Fallback: non-functional parts
        non_func = [p for p in range(n_parts) if p not in has_functional]
        return non_func if non_func else list(range(n_parts))

    elif intent == "hold":
        # "hold": any graspable part (handle or body)
        if has_grasp:
            return has_grasp
        return list(range(n_parts))

    elif intent == "liftup":
        # "liftup": similar to hold — stable grip
        if has_grasp:
            return has_grasp
        return list(range(n_parts))

    elif intent == "handover":
        # "handover": leave functional part free for receiver
        # Contact non-functional parts
        non_func = [p for p in range(n_parts) if p not in has_functional]
        if non_func:
            return non_func
        # Fallback: allow any part
        return list(range(n_parts))

    else:
        # Unknown intent: allow any part
        return list(range(n_parts))


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------
def compute_affordance_metrics(
    hand_verts: np.ndarray,         # (V, 3)  hand vertex positions
    obj_surface_pts: np.ndarray,    # (N, 3)  object surface points
    parts: List[dict],              # from OakBasePartLoader
    intent: str,                    # "use", "hold", "liftup", "handover"
    alpha_init: float = 0.002,      # initial contact distance threshold (m), stricter than the original setup
    alpha_step: float = 0.001,      # step to increase alpha
    alpha_max: float = 0.006,       # maximum alpha before marking invalid
    prior: Optional[AffordancePrior] = None,
    category: Optional[str] = None,
) -> dict:
    """
    Compute affordance accuracy metrics for a single grasp.

    Returns dict with:
      - contact_valid (bool): whether any contact exists
      - aff_acc (float): 1.0 if dominant contact part is in expected parts, else 0.0
      - aff_ratio (float): fraction of contact points falling on expected parts
      - alpha_used (float): the alpha threshold that produced contacts
      - n_contact_pts (int): number of contact points
      - dominant_part (int): part index with most contacts (-1 if no contact)
    """
    # Step 1: Assign part labels to object surface points
    part_labels = assign_part_labels(obj_surface_pts, parts)  # (N,)
    n_parts = len(parts)

    # Step 2: Compute hand-to-object distances using KDTree
    # For each object surface point, find minimum distance to any hand vertex
    from scipy.spatial import cKDTree
    hand_tree = cKDTree(hand_verts)
    dist_obj2hand, _ = hand_tree.query(obj_surface_pts, k=1)
    dist_obj2hand = dist_obj2hand.astype(np.float32)

    # Step 3: Adaptive threshold — find contact points
    alpha = alpha_init
    contact_mask = dist_obj2hand <= alpha
    while contact_mask.sum() == 0 and alpha < alpha_max:
        alpha += alpha_step
        contact_mask = dist_obj2hand <= alpha

    n_contact = int(contact_mask.sum())

    # Step 4: If still no contact, mark as invalid
    if n_contact == 0:
        return {
            "contact_valid": False,
            "aff_acc": 0.0,
            "aff_ratio": 0.0,
            "alpha_used": alpha,
            "n_contact_pts": 0,
            "dominant_part": -1,
        }

    # Step 5: Count contacts per part
    contact_part_labels = part_labels[contact_mask]  # (n_contact,)
    part_contact_counts = np.bincount(contact_part_labels, minlength=n_parts)

    dominant_part = int(part_contact_counts.argmax())

    # Step 6: Get expected parts for intent
    expected_parts = set(get_expected_parts_for_intent(parts, intent, prior=prior, category=category))

    # Step 7: Compute metrics
    aff_acc = 1.0 if dominant_part in expected_parts else 0.0
    contacts_on_expected = sum(part_contact_counts[p] for p in expected_parts)
    aff_ratio = float(contacts_on_expected) / float(n_contact)

    return {
        "contact_valid": True,
        "aff_acc": aff_acc,
        "aff_ratio": aff_ratio,
        "alpha_used": alpha,
        "n_contact_pts": n_contact,
        "dominant_part": dominant_part,
    }


# ---------------------------------------------------------------------------
# Batch evaluation (for evaluate_grasps.py integration)
# ---------------------------------------------------------------------------
def compute_affordance_metrics_batch(
    dumped_grasps_loader,
    oakbase_dir: str,
    meta_dir: str,
    n_obj_points: int = 2048,
    prior_path: Optional[str] = None,
) -> dict:
    """
    Evaluate affordance accuracy over all dumped grasps.

    Args:
        dumped_grasps_loader: DumpedGraspsLoader instance
        oakbase_dir: path to OakBase root
        meta_dir: path to metaV2 directory
        n_obj_points: number of surface points to sample per object
        prior_path: path to affordance_prior.json (from build_affordance_prior.py).
                    If provided, uses GT data-driven expected parts.
                    If None, falls back to attribute-based rules.

    Returns:
        dict with aggregated metrics
    """
    import trimesh

    part_loader = OakBasePartLoader(oakbase_dir, meta_dir)

    # Load GT prior if available
    prior = None
    prior_source = "attribute-rules"
    if prior_path is not None and os.path.isfile(prior_path):
        prior = AffordancePrior(prior_path)
        prior_source = f"GT-prior ({prior_path})"
        print(f"  [Affordance] Using GT prior: {prior_path}")
    else:
        print(f"  [Affordance] No prior file found, using attribute-based rules")

    all_contact_valid = []
    all_aff_acc = []
    all_aff_ratio = []
    skipped = 0
    no_parts = 0

    for i in range(len(dumped_grasps_loader)):
        item = dumped_grasps_loader[i]
        if item is None:
            skipped += 1
            continue

        obj_id = item["obj_id"]
        intent_name = item.get("intent_name", "unknown")
        hand_verts = np.asarray(item["hand_verts_r"], dtype=np.float32)

        # Load parts
        parts = part_loader.load_parts(obj_id)
        if parts is None:
            no_parts += 1
            continue

        # Get category for prior lookup
        category = part_loader.get_category(obj_id)
        if category is not None:
            category = oakbase_cat(category)

        # Sample object surface points from watertight mesh
        obj_wt = item.get("obj_wt")
        if obj_wt is None:
            skipped += 1
            continue

        sample_result = trimesh.sample.sample_surface(obj_wt, n_obj_points)
        obj_pts = np.asarray(sample_result[0], dtype=np.float32)

        # Compute metrics
        metrics = compute_affordance_metrics(
            hand_verts=hand_verts,
            obj_surface_pts=obj_pts,
            parts=parts,
            intent=intent_name,
            prior=prior,
            category=category,
        )

        all_contact_valid.append(float(metrics["contact_valid"]))
        all_aff_acc.append(metrics["aff_acc"])
        all_aff_ratio.append(metrics["aff_ratio"])

    n_evaluated = len(all_contact_valid)

    result = {
        "n_evaluated": n_evaluated,
        "n_skipped": skipped,
        "n_no_parts": no_parts,
        "prior_source": prior_source,
        "contact_valid_rate": float(np.mean(all_contact_valid)) if all_contact_valid else 0.0,
        "aff_acc": float(np.mean(all_aff_acc)) if all_aff_acc else 0.0,
        "aff_ratio": float(np.mean(all_aff_ratio)) if all_aff_ratio else 0.0,
    }

    # Also compute metrics only for contact-valid samples
    valid_mask = [v > 0.5 for v in all_contact_valid]
    valid_aff_acc = [a for a, m in zip(all_aff_acc, valid_mask) if m]
    valid_aff_ratio = [r for r, m in zip(all_aff_ratio, valid_mask) if m]

    result["aff_acc_valid_only"] = float(np.mean(valid_aff_acc)) if valid_aff_acc else 0.0
    result["aff_ratio_valid_only"] = float(np.mean(valid_aff_ratio)) if valid_aff_ratio else 0.0

    return result
