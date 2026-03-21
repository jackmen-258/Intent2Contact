import hashlib
import os
import re
import numpy as np
import torch
import trimesh
import random
import math
from manotorch.manolayer import ManoLayer, MANOOutput
from .utils import (
    ALL_CAT,
    ALL_INTENT,
    ALL_SPLIT,
    CENTER_IDX,
    check_valid,
    get_hand_parameter,
    get_obj_path,
    to_list,
    suppress_trimesh_logging
)
from tqdm import tqdm

# ---------------------------
# Rigid augmentation helpers
# ---------------------------

def _axis_angle_to_matrix_rodrigues(aa: torch.Tensor) -> torch.Tensor:
    """
    aa: (..., 3) axis-angle
    return: (..., 3, 3)
    """
    # Rodrigues' rotation formula
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-8)  # (...,1)
    axis = aa / theta
    x, y, z = axis.unbind(dim=-1)

    zero = torch.zeros_like(x)
    K = torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )  # (...,3,3)

    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(K.shape[:-2] + (3, 3))
    sin = torch.sin(theta)[..., None]  # (...,1,1)
    cos = torch.cos(theta)[..., None]  # (...,1,1)
    R = eye + sin * K + (1.0 - cos) * (K @ K)
    return R


def _matrix_to_axis_angle_fallback(R: torch.Tensor) -> torch.Tensor:
    """
    R: (..., 3, 3)
    return: (..., 3) axis-angle
    Note: stable enough for small/medium rotations used in augmentation.
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)  # (...)

    # For small angles, axis is ill-defined; use first-order approximation
    small = theta < 1e-5
    wx = R[..., 2, 1] - R[..., 1, 2]
    wy = R[..., 0, 2] - R[..., 2, 0]
    wz = R[..., 1, 0] - R[..., 0, 1]
    w = torch.stack([wx, wy, wz], dim=-1)  # (...,3)

    denom = (2.0 * torch.sin(theta)).unsqueeze(-1).clamp_min(1e-8)  # (...,1)
    axis = w / denom
    aa = axis * theta.unsqueeze(-1)

    # small-angle fallback: aa ≈ 0.5 * w
    aa = torch.where(small.unsqueeze(-1), 0.5 * w, aa)
    return aa


def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    try:
        from pytorch3d.transforms import axis_angle_to_matrix  # type: ignore
        return axis_angle_to_matrix(aa)
    except Exception:
        return _axis_angle_to_matrix_rodrigues(aa)


def _matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    try:
        from pytorch3d.transforms import matrix_to_axis_angle  # type: ignore
        return matrix_to_axis_angle(R)
    except Exception:
        return _matrix_to_axis_angle_fallback(R)


def _euler_xyz_to_matrix(rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor) -> torch.Tensor:
    """
    rx,ry,rz: (,) or (B,) radians
    return: (...,3,3)
    """
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    Rx = torch.stack(
        [torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], -1),
         torch.stack([torch.zeros_like(cx), cx, -sx], -1),
         torch.stack([torch.zeros_like(cx), sx, cx], -1)], -2)

    Ry = torch.stack(
        [torch.stack([cy, torch.zeros_like(cy), sy], -1),
         torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], -1),
         torch.stack([-sy, torch.zeros_like(cy), cy], -1)], -2)

    Rz = torch.stack(
        [torch.stack([cz, -sz, torch.zeros_like(cz)], -1),
         torch.stack([sz, cz, torch.zeros_like(cz)], -1),
         torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], -1)], -2)

    return Rz @ Ry @ Rx

class OIShape:
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.data_split = cfg.DATA_SPLIT
        self.is_train = ("train" in self.data_split)

        self.intent_mode = cfg.INTENT_MODE
        self.category = cfg.OBJ_CATES

        self.use_downsample_mesh = True
        self.n_samples = 2048
        self.mano_assets_root = "assets/mano_v1_2"

        assert 'OAKINK_DIR' in os.environ, "environment variable 'OAKINK_DIR' is not set"
        data_dir = os.path.join(os.environ['OAKINK_DIR'], "shape")
        oi_shape_dir = os.path.join(data_dir, "oakink_shape_v2")
        meta_dir = os.path.join(data_dir, "metaV2")

        self.data_dir = data_dir
        self.meta_dir = meta_dir
        self.oi_shape_dir = oi_shape_dir

        if self.data_split == 'all':
            self.data_split = ALL_SPLIT
        if self.category == 'all':
            self.category = ALL_CAT
        if self.intent_mode == 'all':
            self.intent_mode = list(ALL_INTENT)

        self.data_split = to_list(self.data_split)
        self.categories = to_list(self.category)
        self.intent_mode = to_list(self.intent_mode)
        assert (check_valid(self.data_split, ALL_SPLIT) and check_valid(self.categories, ALL_CAT) and
                check_valid(self.intent_mode, list(ALL_INTENT))), "invalid data split, category, or intent!"

        self.intent_idx = [ALL_INTENT[i] for i in self.intent_mode]
        self.action_id_to_intent = {v: k for k, v in ALL_INTENT.items()}

        self.mano_layer = ManoLayer(center_idx=CENTER_IDX, mano_assets_root=self.mano_assets_root)

        self.obj_warehouse = {}
        self.obj_bbox_centers = {}

        self.grasp_list = self._prepare_data()
        self.obj_id_set = {g["obj_id"] for g in self.grasp_list}

    def _prepare_data(self):
        # region ===== filter with regex >>>>>
        grasp_list = []
        category_begin_idx = []
        seq_cat_matcher = re.compile(r"(.+)/(.{6})_(.{4})_([_0-9]+)/([\-0-9]+)")
        for cat in tqdm(self.categories, desc="Process categories"):
            real_matcher = re.compile(rf"({cat}/(.{{6}})/.{{10}})/hand_param\.pkl$")
            virtual_matcher = re.compile(rf"({cat}/(.{{6}})/.{{10}})/(.{{6}})/hand_param\.pkl$")
            path = os.path.join(self.oi_shape_dir, cat)
            category_begin_idx.append(len(grasp_list))
            for cur, dirs, files in os.walk(path, followlinks=False):
                dirs.sort()
                for f in files:
                    re_match = virtual_matcher.findall(os.path.join(cur, f))
                    is_virtual = len(re_match) > 0
                    re_match = re_match + real_matcher.findall(os.path.join(cur, f))
                    if len(re_match) > 0:
                        # ? regex should return : [(path, raw_oid, tag, [oid])]
                        assert len(re_match) == 1, "regex should return only one match"
                        source = open(os.path.join(self.oi_shape_dir, re_match[0][0], "source.txt")).read()
                        grasp_cat_match = seq_cat_matcher.findall(source)[0]
                        pass_stage, raw_obj_id, action_id, subject_id = (grasp_cat_match[0], grasp_cat_match[1],
                                                                         grasp_cat_match[2], grasp_cat_match[3])
                        obj_id = re_match[0][2] if is_virtual else re_match[0][1]
                        assert (is_virtual and raw_obj_id == re_match[0][1]) or obj_id == raw_obj_id
                        # * filter with intent mode
                        if action_id not in self.intent_idx:
                            continue
                        # * filter with data split
                        obj_id_hash = int(hashlib.md5(obj_id.encode("utf-8")).hexdigest(), 16)  # random select
                        if obj_id_hash % 10 < 8 and "train" not in self.data_split:
                            continue
                        elif obj_id_hash % 10 == 8 and "val" not in self.data_split:
                            continue
                        elif obj_id_hash % 10 == 9 and "test" not in self.data_split:
                            continue

                        hand_pose, hand_shape, hand_tsl = get_hand_parameter(os.path.join(cur, f))
                        grasp_item = {
                            "cate_id": cat,
                            "obj_id": obj_id,
                            "hand_joints": None,
                            "hand_verts": None,
                            "hand_pose": hand_pose,
                            "hand_shape": hand_shape,
                            "hand_tsl": hand_tsl,
                            "is_virtual": is_virtual,
                            "raw_obj_id": raw_obj_id,
                            "action_id": action_id,
                            "subject_id": subject_id,
                            "file_path": os.path.join(cur, f),
                        }
                        grasp_list.append(grasp_item)
        # endregion <<<<

        # region ===== cal hand joints >>>>>
        batch_hand_pose = []
        batch_hand_shape = []
        batch_hand_tsl = []
        for _, g in enumerate(grasp_list):
            batch_hand_pose.append(g["hand_pose"])
            batch_hand_shape.append(g["hand_shape"])
            batch_hand_tsl.append(g["hand_tsl"])
        batch_hand_shape = torch.from_numpy(np.stack(batch_hand_shape))
        batch_hand_pose = torch.from_numpy(np.stack(batch_hand_pose))
        batch_hand_tsl = np.stack(batch_hand_tsl)
        mano_output: MANOOutput = self.mano_layer(batch_hand_pose, batch_hand_shape)
        batch_hand_joints = mano_output.joints.numpy() + batch_hand_tsl[:, None, :]
        batch_hand_verts = mano_output.verts.numpy() + batch_hand_tsl[:, None, :]
        batch_hand_tsl = batch_hand_joints[:, CENTER_IDX]  # center idx from 0 to 9
        for i in range(len(grasp_list)):
            grasp_list[i]["hand_joints"] = batch_hand_joints[i]
            grasp_list[i]["hand_verts"] = batch_hand_verts[i]
            grasp_list[i]["hand_tsl"] = batch_hand_tsl[i]
        # endregion <<<<<

        return grasp_list

    def __len__(self):
        return len(self.grasp_list)

    def get_obj_mesh(self, idx):
        obj_id = self.grasp_list[idx]["obj_id"]
        if obj_id not in self.obj_warehouse:
            obj_path = get_obj_path(obj_id, self.data_dir, self.meta_dir, use_downsample=self.use_downsample_mesh)
            obj_trimesh = trimesh.load(obj_path, process=False, force="mesh", skip_materials=True)
            bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2
            obj_trimesh.vertices = obj_trimesh.vertices - bbox_center
            self.obj_warehouse[obj_id] = obj_trimesh
        return self.obj_warehouse[obj_id]

    def get_obj_id(self, idx):
        return self.grasp_list[idx]["obj_id"]

    def get_hand_joints(self, idx):
        return self.grasp_list[idx]["hand_joints"]

    def get_hand_shape(self, idx):
        return self.grasp_list[idx]["hand_shape"]

    def get_hand_pose(self, idx):
        return self.grasp_list[idx]["hand_pose"]

    def get_obj_rotmat(self, idx):
        return np.eye(3, dtype=np.float32)

    def get_intent(self, idx):
        act_id = self.grasp_list[idx]["action_id"]
        intent_name = self.action_id_to_intent[act_id]
        return int(act_id), intent_name

    def __getitem__(self, idx):
        grasp = self.grasp_list[idx]
        obj_mesh = self.get_obj_mesh(idx)

        sample = trimesh.sample.sample_surface(obj_mesh, self.n_samples)
        
        obj_verts = np.array(sample[0], dtype=np.float32)
        obj_vn = np.array(obj_mesh.face_normals[sample[1]], dtype=np.float32)
        obj_id = self.get_obj_id(idx)

        # 获取手部参数并确保是numpy数组
        hand_pose = np.array(grasp["hand_pose"], dtype=np.float32)   # (48,)
        hand_tsl = np.array(grasp["hand_tsl"], dtype=np.float32)     # (3,)
        hand_shape = np.array(grasp["hand_shape"], dtype=np.float32) # (10,)
        hand_verts = np.array(grasp["hand_verts"], dtype=np.float32) # (V,3)

        # 最终确保所有数据类型正确
        obj_verts = np.array(obj_verts, dtype=np.float32)
        obj_vn = np.array(obj_vn, dtype=np.float32)
        hand_pose = np.array(hand_pose, dtype=np.float32)
        hand_tsl = np.array(hand_tsl, dtype=np.float32)
        hand_shape = np.array(hand_shape, dtype=np.float32)
        hand_verts = np.array(hand_verts, dtype=np.float32)

        # ---------------------------
        # Rigid augmentation (train only)
        # ---------------------------
        if self.is_train:
            p = float(getattr(self.cfg, "AUG_RIGID_P", 0.8))
            yaw_range_deg = float(getattr(self.cfg, "AUG_RIGID_YAW_RANGE_DEG", 30.0))
            tilt_std_deg = float(getattr(self.cfg, "AUG_RIGID_TILT_STD_DEG", 10.0))
            trans_std_m = float(getattr(self.cfg, "AUG_RIGID_TRANS_STD_M", 0.0))

            if random.random() < p:
                # 用 torch 生成旋转/轴角复合，最后转回 numpy（避免自己写 numpy 版 axis-angle 变换）
                device = torch.device("cpu")
                dtype = torch.float32

                yaw = (torch.rand((), device=device, dtype=dtype) * 2 - 1) * (yaw_range_deg * math.pi / 180.0)  # [-30°, 30°]
                rx = (torch.rand((), device=device, dtype=dtype) * 2 - 1) * (tilt_std_deg * math.pi / 180.0)    # [-10°, 10°]
                ry = (torch.rand((), device=device, dtype=dtype) * 2 - 1) * (tilt_std_deg * math.pi / 180.0)    # [-10°, 10°]
                rz = yaw
                R = _euler_xyz_to_matrix(rx, ry, rz)  # (3,3)

                t = torch.randn(3, device=device, dtype=dtype) * trans_std_m  # (3,)

                R_np = R.numpy().astype(np.float32)
                t_np = t.numpy().astype(np.float32)

                # 物体点与法向
                obj_verts = (obj_verts @ R_np.T) + t_np
                obj_vn = (obj_vn @ R_np.T)

                # 手的顶点与平移
                hand_verts = (hand_verts @ R_np.T) + t_np
                hand_tsl = (hand_tsl @ R_np.T) + t_np

                # 更新 MANO 全局旋转（hand_pose[:3] 是 axis-angle）
                aa_global = torch.from_numpy(hand_pose[:3]).to(dtype=dtype)  # (3,)
                R_hand = _axis_angle_to_matrix(aa_global)                    # (3,3)
                R_new = R @ R_hand
                aa_new = _matrix_to_axis_angle(R_new).numpy().astype(np.float32)
                hand_pose[:3] = aa_new

        intent_id, intent_name = self.get_intent(idx)
        intent_id = intent_id - 1  # [1, 2, 3, 4] -> [0, 1, 2, 3]
        
        # 返回标准的数据类型
        result = {
            "obj_verts": obj_verts,
            "obj_vn": obj_vn,
            "hand_pose": hand_pose,
            "hand_tsl": hand_tsl,
            "hand_shape": hand_shape,
            "hand_verts": hand_verts,
            "intent_id": intent_id,
            "intent_name": intent_name,
            "obj_id": obj_id,
            "cate_id": grasp["cate_id"],
            "obj_rotmat": self.get_obj_rotmat(idx),
            "sample_idx": idx,
        }
        
        return result
