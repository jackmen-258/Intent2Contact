import os
import pickle
import numpy as np
import torch
from torch.utils import data
import trimesh

from manopth.manolayer import ManoLayer
from manopth.rodrigues_layer import batch_rodrigues


# .npz 文件是 NumPy 库使用的文件格式，用于存储和交换 NumPy 数组数据
# .npz 文件实际上是一个压缩的 ZIP 文件，它可以包含一个或多个 NumPy 数组，以及它们的元数据（如数据类型、形状等）

class Grab(data.Dataset):
    def __init__(self, 
                 dataset_root='grab_data',
                 ds_name='train',
                 n_samples=2048):
        super().__init__()
        self.ds_name = ds_name
        self.ds_path = os.path.join(dataset_root, ds_name)
        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%ds_name))

        frame_names = np.load(os.path.join(dataset_root, ds_name, 'frame_names.npz'))['frame_names']
        # np.asarray 将输入的数据转换为 NumPy 数组
        self.frame_names = np.asarray([os.path.join(dataset_root, fname) for fname in frame_names])
        # 提取 subject ID 和物体名称(eg. train/data/s4/cup_pass_1/177.npz)
        self.frame_sbjs = np.asarray([name.split('/')[-3] for name in self.frame_names])
        self.frame_objs = np.asarray([name.split('/')[-2].split('_')[0] for name in self.frame_names])
        self.n_samples = n_samples
        
        # 找出所有唯一的 subject ID
        self.sbjs = np.unique(self.frame_sbjs)
        self.sbj_info = np.load(os.path.join(dataset_root, 'sbj_info.npy'), allow_pickle=True).item()
        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        # 将 subject ID 替换为从0开始的索引
        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        # 加载手部网格的面数据
        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)
        
        with open(os.path.join("assets/closed_mano_faces.pkl"), 'rb') as f:
            self.hand_faces = pickle.load(f)

        # 设置物体网格的路径
        self.obj_root = os.path.join(dataset_root, "obj_meshes")
        # 初始化 MANO 层
        self.mano_layer = ManoLayer(ncomps=45, flat_hand_mean=True, side="right", mano_root=os.path.join("mano/models"), use_pca=False, joint_rot_mode="rotmat")

    def __len__(self):
        # 903
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]
        
    def _np2torch(self,ds_path):
        # allow_pickle=True 参数允许加载 pickle 序列化的对象
        data = np.load(ds_path, allow_pickle=True)
        # 创建一个字典 data_torch，它将 NumPy 数据文件中的每个数组 k 转换为 PyTorch 张量
        data_torch = {k:torch.tensor(data[k]).float() for k in data.files}
        return data_torch

    # __getitem__ 是一个特殊方法（也称为魔术方法），它用于定义当对象被索引或切片时的行为
    # 这个方法通常在自定义的数据结构或类中实现，以便它们可以像列表、元组或字典那样通过索引访问元素
    # 当使用索引或切片操作符（如 []）访问对象的元素时，Python 会自动调用该对象的 __getitem__ 方法
    def __getitem__(self, item):
        # 获取物体名称，并构建物体网格文件的路径
        obj_name = self.frame_objs[item]
        obj_mesh_path = os.path.join(self.obj_root, obj_name + '.ply')
        # 加载物体网格
        obj_mesh = trimesh.load(obj_mesh_path, file_type="ply")
        # obj_faces 是一个索引数组，用于指定每个面由哪些顶点组成
        obj_faces = obj_mesh.faces

        # 获取物体的旋转矩阵和平移向量
        rot_mat = self.ds["root_orient_obj_rotmat"][item].numpy().reshape(3, 3)
        transl = self.ds["trans_obj"][item].numpy()
        
        # 计算物体顶点
        # @是矩阵乘法运算符，用于将每个顶点通过旋转矩阵进行旋转
        obj_verts = obj_mesh.vertices @ rot_mat + transl
        # obj_verts.mean() 计算所有顶点坐标的平均值，即几何中心的坐标
        # axis=0 表示沿着第一个维度进行计算，即对所有顶点进行计算，而不是对每个坐标分量单独计算
        offset = obj_verts.mean(axis=0, keepdims=True)
        # 从每个顶点坐标中减去 offset，即将物体的几何中心移动到坐标系的原点
        obj_verts = obj_verts - offset

        # 获取主体索引和主体顶点模板
        sbj_idx = self.frame_sbjs[item]
        v_template = self.sbj_vtemp[sbj_idx]

        # 获取手部(右手)的全局定向和姿态旋转矩阵
        global_orient = self.ds['global_orient_rhand_rotmat'][item]
        rhand_rotmat = self.ds['fpose_rhand_rotmat'][item]

        # 初步处理手部姿态
        handrot = torch.cat([global_orient, rhand_rotmat], dim=0).unsqueeze(dim=0)
        th_trans = self.ds['trans_rhand'][item].unsqueeze(dim=0) - torch.FloatTensor(offset)
        th_v_template = v_template.unsqueeze(dim=0)
        # 调用mano_layer函数计算手部顶点和关节变换矩阵
        hand_verts, hand_frames = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)

        # 逆变换顶点，将顶点坐标从世界坐标系变换回物体的局部坐标系
        # 在旋转矩阵的情况下，由于它是正交的，转置操作和逆矩阵是等价的，即 rot_mat.T == rot_mat.inverse()
        obj_verts = obj_verts @ rot_mat.T

        # 当手部与物体交互时（如抓取、操作），手部的方向需要根据物体的方向进行调整
        # 将手部的旋转矩阵与物体的旋转矩阵相乘，是为了确保手部模型正确地相对于物体的方向定位
        global_orient = torch.from_numpy(rot_mat).float() @ global_orient
        handrot = torch.cat([global_orient, rhand_rotmat], dim=0).unsqueeze(dim=0)
        
        # 获取根关节位置(根关节定义在全局坐标系中)
        # hand_frames[:, 0, :3, 3] 提取根关节的平移部分，形状为 (B, 3)，表示每个样本的根关节的 x, y, z 坐标
        root_center = hand_frames[:, 0, :3, 3]
        # root_center[:, None, :] 通过在中间添加一个维度，将 root_center 的形状从 (B, 3) 变为 (B, 1, 3)
        # (root_center[:, None, :] @ rot_mat.T).squeeze(dim=1) - root_center 计算了旋转后的位置与原始位置之间的差，即局部坐标系下的位置偏移
        # 这个差值表示了从局部坐标系原点(即根关节的原始位置)到根关节的向量
        th_trans = (root_center[:, None, :] @ rot_mat.T).squeeze(dim=1) - root_center + th_trans
        # 更新手部顶点和关节变换矩阵
        hand_verts, hand_frames = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)
        
        # 在训练集上执行数据增强(obj_verts已改变)
        if self.ds_name == "train":
            # 生成随机旋转角度
            # 创建一个形状为 (1, 3) 的张量 orient，其元素值在 -π/6 到 π/6 之间均匀分布
            orient = torch.FloatTensor(1, 3).uniform_(-np.pi/6, np.pi/6)
            # 生成旋转矩阵
            # 使用 batch_rodrigues 函数将轴角向量转换为旋转矩阵
            aug_rot_mats = batch_rodrigues(orient.view(-1, 3)).view([1, 3, 3])
            aug_rot_mat = aug_rot_mats[0]

            obj_verts = obj_verts @ aug_rot_mat.numpy().T
            global_orient = aug_rot_mat @ global_orient
            handrot = torch.cat([global_orient, rhand_rotmat], dim=0).unsqueeze(dim=0)

            root_center = hand_frames[:, 0, :3, 3]
            th_trans = (root_center[:, None, :] @ aug_rot_mat.T).squeeze(dim=1) - root_center + th_trans
            hand_verts, hand_frames = self.mano_layer(handrot, th_trans=th_trans, th_v_template=th_v_template)

        hand_verts = hand_verts.squeeze(dim=0).float()
        hand_frames = hand_frames.squeeze(dim=0).float()

        obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
        # 采样表面点
        # sample 是一个包含采样结果的元组，通常包含采样点的位置和对应的面索引
        sample = trimesh.sample.sample_surface(obj_mesh, self.n_samples)
        # 提取采样点坐标
        obj_verts = sample[0].astype(np.float32)
        # 提取采样点所在面的法线
        obj_vn = obj_mesh.face_normals[sample[1]].astype(np.float32)

        return {
            "hand_verts": hand_verts,
            "hand_frames": hand_frames,
            "obj_verts": obj_verts,
            "obj_vn": obj_vn
        }