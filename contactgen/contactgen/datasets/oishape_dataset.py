import hashlib
import os
import re
import numpy as np
import torch
import trimesh
from tqdm import tqdm

import pickle
import json
import glob
import logging
from manopth.manolayer import ManoLayer
from manopth.rodrigues_layer import batch_rodrigues

ALL_CAT = [
    "apple",
    "banana",
    "binoculars",
    "bottle",
    "bowl",
    "cameras",
    "can",
    "cup",
    "cylinder_bottle",
    "donut",
    "eyeglasses",
    "flashlight",
    "fryingpan",
    "gamecontroller",
    "hammer",
    "headphones",
    "knife",
    "lightbulb",
    "lotion_pump",
    "mouse",
    "mug",
    "pen",
    "phone",
    "pincer",
    "power_drill",
    "scissors",
    "screwdriver",
    "squeezable",
    "stapler",
    "teapot",
    "toothbrush",
    "trigger_sprayer",
    "wineglass",
    "wrench",
]

ALL_SPLIT = [
    "train",
    "val",
    "test",
]

ALL_INTENT = {
    "use": "0001",
    "hold": "0002",
    "liftup": "0003",
    "handover": "0004",
}

CENTER_IDX = 0

def to_list(x):
    if isinstance(x, list):
        return x
    return [x]

def check_valid(list, valid_list):
    for x in list:
        if x not in valid_list:
            return False
    return True

def suppress_trimesh_logging():
    logger = logging.getLogger("trimesh")
    logger.setLevel(logging.ERROR)

def get_hand_parameter(path):
    pose = pickle.load(open(path, "rb"))
    return pose["pose"], pose["shape"], pose["tsl"]

def get_obj_path(oid, data_path, meta_path, use_downsample=True, key="align"):
    obj_suffix_path = "align_ds" if use_downsample else "align"
    real_meta = json.load(open(os.path.join(meta_path, "object_id.json"), "r"))
    virtual_meta = json.load(open(os.path.join(meta_path, "virtual_object_id.json"), "r"))
    if oid in real_meta:
        obj_name = real_meta[oid]["name"]
        obj_path = os.path.join(data_path, "OakInkObjectsV2")
    else:
        obj_name = virtual_meta[oid]["name"]
        obj_path = os.path.join(data_path, "OakInkVirtualObjectsV2")
    obj_mesh_path = list(
        glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj")) +
        glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.ply")))
    if len(obj_mesh_path) > 1:
        obj_mesh_path = [p for p in obj_mesh_path if key in os.path.split(p)[1]]
    assert len(obj_mesh_path) == 1
    return obj_mesh_path[0]

# ---------------------------
# Rigid augmentation helpers
# ---------------------------
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

def _matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    try:
        from pytorch3d.transforms import matrix_to_axis_angle  # type: ignore
        return matrix_to_axis_angle(R)
    except Exception:
        return _matrix_to_axis_angle_fallback(R)

class OIShape:
    def __init__(self, split='train'):
        suppress_trimesh_logging()

        self.data_split = split
        self.is_train = ("train" in self.data_split)

        self.category = "all"
        self.intent_idx = set(ALL_INTENT.values())

        self.use_downsample_mesh = True
        self.n_samples = 2048

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

        self.data_split = to_list(self.data_split)
        self.categories = to_list(self.category)

        assert (check_valid(self.data_split, ALL_SPLIT) and check_valid(self.categories, ALL_CAT), 
                "invalid data split or category")

        self.mano_layer = ManoLayer(
            ncomps=45, 
            flat_hand_mean=True, 
            side="right", 
            mano_root=os.path.join("mano/models"),
            use_pca=False, 
            joint_rot_mode="rotmat"
        )

        self.obj_warehouse = {}

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
        for g in grasp_list:
            batch_hand_pose.append(g["hand_pose"])
            batch_hand_shape.append(g["hand_shape"])
            batch_hand_tsl.append(g["hand_tsl"])

        batch_hand_pose = torch.from_numpy(np.stack(batch_hand_pose)).float()   # (B, 48)
        batch_hand_shape = torch.from_numpy(np.stack(batch_hand_shape)).float() # (B, 10)
        batch_hand_tsl = torch.from_numpy(np.stack(batch_hand_tsl)).float()     # (B, 3)

        # 将 axis-angle 转为 rotmat
        batch_hand_pose_aa = batch_hand_pose.view(-1, 16, 3)  # (B, 16, 3)
        batch_handrot = batch_rodrigues(batch_hand_pose_aa.view(-1, 3)).view(-1, 16, 3, 3)  # (B, 16, 3, 3)

        # 调用 MANO layer
        batch_hand_verts, batch_hand_frames = self.mano_layer(batch_handrot, th_trans=batch_hand_tsl, th_v_template=None)
        
        batch_hand_verts = batch_hand_verts.cpu().numpy()
        batch_hand_joints = batch_hand_frames[:, :, :3, 3].cpu().numpy()
        batch_hand_tsl = batch_hand_joints[:, CENTER_IDX, :]

        for i in range(len(grasp_list)):
            obj_id = grasp_list[i]["obj_id"]
            
            # 加载物体并计算 bbox_center
            obj_path = get_obj_path(obj_id, self.data_dir, self.meta_dir, use_downsample=self.use_downsample_mesh)
            obj_trimesh = trimesh.load(obj_path, process=False, force="mesh", skip_materials=True)
            bbox_center = (obj_trimesh.vertices.min(0) + obj_trimesh.vertices.max(0)) / 2.0
            
            # 对齐到 object-centered 坐标系
            grasp_list[i]["hand_joints"] = batch_hand_joints[i] - bbox_center
            grasp_list[i]["hand_verts"] = batch_hand_verts[i] - bbox_center
            grasp_list[i]["hand_tsl"] = batch_hand_tsl[i] - bbox_center
        
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

    def __getitem__(self, idx):
        grasp = self.grasp_list[idx]
        obj_mesh = self.get_obj_mesh(idx)

        sample = trimesh.sample.sample_surface(obj_mesh, self.n_samples)
        
        obj_verts = np.array(sample[0], dtype=np.float32)
        obj_vn = np.array(obj_mesh.face_normals[sample[1]], dtype=np.float32)

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
            orient = torch.FloatTensor(1, 3).uniform_(-np.pi/6, np.pi/6)  # (1,3)
            aug_rot_mats = batch_rodrigues(orient.view(-1, 3)).view([1, 3, 3])
            aug_rot_mat = aug_rot_mats[0]  # (3,3)

            # 1) 物体顶点与法向
            obj_verts = obj_verts @ aug_rot_mat.numpy().T
            obj_vn = obj_vn @ aug_rot_mat.numpy().T

            # 2) 手部全局旋转更新
            hand_pose_t = torch.from_numpy(hand_pose).unsqueeze(0).float()  # (1,48)
            hand_pose_aa = hand_pose_t.view(1, -1, 3)  # (1,16,3)
            handrot = batch_rodrigues(hand_pose_aa.view(-1, 3)).view(1, 16, 3, 3)  # (1,16,3,3)

            global_orient = handrot[0, 0]  # (3,3)
            rhand_rotmat = handrot[0, 1:]  # (15,3,3)

            # 更新全局旋转
            global_orient = aug_rot_mat @ global_orient
            handrot = torch.cat([global_orient.unsqueeze(0), rhand_rotmat], dim=0).unsqueeze(0)  # (1,16,3,3)

            hand_tsl_t = torch.from_numpy(hand_tsl).unsqueeze(0).float()  # (1,3)
            hand_shape_t = torch.from_numpy(hand_shape).unsqueeze(0).float()  # (1,10)

            # 3) 计算 hand_frames（需要 MANO layer 返回 frames）
            hand_verts_mano, hand_frames = self.mano_layer(handrot, th_trans=hand_tsl_t, th_betas=hand_shape_t)

            # 4) 通过 root 关节计算平移偏移（与 Grab 一致）
            root_center = hand_frames[:, 0, :3, 3]  # (1,3)
            th_trans = (root_center[:, None, :] @ aug_rot_mat.T).squeeze(dim=1) - root_center + hand_tsl_t

            # 5) 最终手部参数
            hand_verts_final, hand_frames_final = self.mano_layer(handrot, th_trans=th_trans, th_v_template=None)
            hand_verts = hand_verts_final.squeeze(0).numpy().astype(np.float32)
            hand_frames = hand_frames_final.squeeze(0).numpy().astype(np.float32)

            # 更新 hand_pose（将旋转矩阵转回 axis-angle）
            global_orient_aa = _matrix_to_axis_angle(global_orient)  # (3,)
            rhand_aa = _matrix_to_axis_angle(rhand_rotmat.view(-1, 3, 3)).view(15, 3)  # (15,3)
            hand_pose = torch.cat([global_orient_aa, rhand_aa.flatten()]).numpy().astype(np.float32)  # (48,)
            hand_tsl = th_trans.squeeze(0).numpy().astype(np.float32)  # (3,)
        else:
            # 验证集/测试集：无增强，但仍需计算 hand_frames
            hand_pose_t = torch.from_numpy(hand_pose).unsqueeze(0).float()
            hand_pose_aa = hand_pose_t.view(1, -1, 3)
            handrot = batch_rodrigues(hand_pose_aa.view(-1, 3)).view(1, 16, 3, 3)

            hand_tsl_t = torch.from_numpy(hand_tsl).unsqueeze(0).float()
            hand_shape_t = torch.from_numpy(hand_shape).unsqueeze(0).float()

            hand_verts_mano, hand_frames = self.mano_layer(handrot, th_trans=hand_tsl_t, th_v_template=None)
            hand_verts = hand_verts_mano.squeeze(0).numpy().astype(np.float32)
            hand_frames = hand_frames.squeeze(0).numpy().astype(np.float32)
        
        result = {
            "hand_verts": hand_verts,      # (778, 3)
            "hand_frames": hand_frames,    # (16, 4, 4)
            "obj_verts": obj_verts,        # (N, 3)
            "obj_vn": obj_vn,              # (N, 3)
        }
        
        return result
