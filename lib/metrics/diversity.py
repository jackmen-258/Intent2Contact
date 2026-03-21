import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.stats import entropy as scipy_entropy
import torch

def diversity(cluster_array, cls_num=20):
    """
    对齐 HALO：
      - KMeans: scipy.cluster.vq.kmeans
      - Entropy: scipy.stats.entropy(counts)  # 默认自然对数 ln
      - avg_dist: mean(distance to assigned center)
    """
    if cluster_array is None:
        return 0.0, 0.0
    cluster_array = np.asarray(cluster_array, dtype=np.float32)
    if cluster_array.ndim != 2 or cluster_array.shape[0] < 2:
        return 0.0, 0.0

    cls_num = int(cls_num)
    cls_num = max(1, min(cls_num, cluster_array.shape[0]))

    codes, _distortion = kmeans(cluster_array, cls_num)
    vecs, dist = vq(cluster_array, codes)

    counts, _ = np.histogram(vecs, bins=len(codes))

    # ✅ HALO：直接对 counts 求 entropy（内部会归一化），默认 ln
    ent = float(scipy_entropy(counts))
    avg_dist = float(np.mean(dist)) if dist is not None and len(dist) > 0 else 0.0
    return ent, avg_dist

def xyz_to_xyz1(xyz):
    """ Convert xyz vectors from [BS, ..., 3] to [BS, ..., 4] for matrix multiplication
    """
    ones = torch.ones([*xyz.shape[:-1], 1], device=xyz.device)
    # print("xyz shape", xyz.shape)
    # print("one", ones.shape)
    return torch.cat([xyz, ones], dim=-1)

def pad34_to_44(mat):
    last_row = torch.tensor([0., 0., 0., 1.], device=mat.device).reshape(1, 4).repeat(*mat.shape[:-2], 1, 1)
    return torch.cat([mat, last_row], dim=-2)

def convert_joints(joints, source, target):
    halo_joint_to_mano = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
    mano_joint_to_halo = np.array([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20])
    mano_joint_to_biomech = np.array([0, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19, 4, 8, 12, 16, 20])
    biomech_joint_to_mano = np.array([0, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14, 19, 5, 10, 15, 20])
    halo_joint_to_biomech = np.array([0, 13, 1, 4, 10, 7, 14, 2, 5, 11, 8, 15, 3, 6, 12, 9, 16, 17, 18, 19, 20])
    # halo_joint_to_biomech = np.array([0, 11, 15, 19, 4, 1, 5, 9, 8, 13, 17, 2, 12, 18, 3, 7, 16, 6, 10, 14, 20])
    biomech_joint_to_halo = np.array([0, 2, 7, 12, 3, 8, 13, 5, 10, 15, 4, 9, 14, 1, 6, 11, 16, 17, 18, 19, 20])

    if source == 'halo' and target == 'biomech':
        # conn = nasa_joint_to_mano[mano_joint_to_biomech]
        # print(conn)
        # tmp_mano = joints[:, halo_joint_to_mano]
        # out = tmp_mano[:, mano_joint_to_biomech]
        return joints[:, halo_joint_to_biomech]
    if source == 'biomech' and target == 'halo':
        return joints[:, biomech_joint_to_halo]
    if source == 'mano' and target == 'biomech':
        return joints[:, mano_joint_to_biomech]
    if source == 'biomech' and target == 'mano':
        return joints[:, biomech_joint_to_mano]
    if source == 'halo' and target == 'mano':
        return joints[:, halo_joint_to_mano]
    if source == 'mano' and target == 'halo':
        return joints[:, mano_joint_to_halo]

    print("-- Undefined convertion. Return original tensor --")
    return joints

def normalize(bv, eps=1e-8):  # epsilon
    """
    Normalizes the last dimension of bv such that it has unit length in
    euclidean sense
    """
    eps_mat = torch.tensor(eps, device=bv.device)
    norm = torch.max(torch.norm(bv, dim=-1, keepdim=True), eps_mat)
    bv_n = bv / norm
    return bv_n

def angle2(v1, v2):
    """
    Numerically stable way of calculating angles.
    See: https://scicomp.stackexchange.com/questions/27689/numerically-stable-way-of-computing-angles-between-vectors
    """
    eps = 1e-10  # epsilon
    eps_mat = torch.tensor([eps], device=v1.device)
    n_v1 = v1 / torch.max(torch.norm(v1, dim=-1, keepdim=True), eps_mat)
    n_v2 = v2 / torch.max(torch.norm(v2, dim=-1, keepdim=True), eps_mat)
    a = 2 * torch.atan2(
        torch.norm(n_v1 - n_v2, dim=-1), torch.norm(n_v1 + n_v2, dim=-1)
    )
    return a

def rotation_matrix(angles, axis):
    """
    Converts Rodrigues rotation formula into a rotation matrix
    """
    eps = torch.tensor(1e-6)  # epsilon
    # print("norm", torch.abs(torch.sum(axis ** 2, dim=-1) - 1))
    try:
        assert torch.any(
            torch.abs(torch.sum(axis ** 2, dim=-1) - 1) < eps
        ), "axis must have unit norm"
    except:
        print("Warning: axis does not have unit norm")
        # import pdb
        # pdb.set_trace()
    dev = angles.device
    batch_size = angles.shape[0]
    sina = torch.sin(angles).view(batch_size, 1, 1)
    cosa_1_minus = (1 - torch.cos(angles)).view(batch_size, 1, 1)
    a_batch = axis.view(batch_size, 3)
    o = torch.zeros((batch_size, 1), device=dev)
    a0 = a_batch[:, 0:1]
    a1 = a_batch[:, 1:2]
    a2 = a_batch[:, 2:3]
    cprod = torch.cat((o, -a2, a1, a2, o, -a0, -a1, a0, o), 1).view(batch_size, 3, 3)
    I = torch.eye(3, device=dev).view(1, 3, 3)
    R1 = cprod * sina
    R2 = cprod.bmm(cprod) * cosa_1_minus
    R = I + R1 + R2
    return R

def cross(bv_1, bv_2, do_normalize=False):
    """
    Computes the cross product of the last dimension between bv_1 and bv_2.
    If normalize is True, it normalizes the vector to unit length.
    """
    cross_prod = torch.cross(bv_1, bv_2, dim=-1)
    if do_normalize:
        cross_prod = normalize(cross_prod)
    return cross_prod

def get_alignment_mat(v1, v2):
    """
    Returns the rotation matrix R, such that R*v1 points in the same direction as v2
    """
    axis = cross(v1, v2, do_normalize=True)
    ang = angle2(v1, v2)
    R = rotation_matrix(ang, axis)
    return R

def transform_to_canonical(kp3d, skeleton='bmc'):
    """Undo global translation and rotation
    """
    normalization_mat = compute_canonical_transform(kp3d, skeleton=skeleton)
    kp3d = xyz_to_xyz1(kp3d)
    # import pdb
    # pdb.set_trace()
    kp3d_canonical = torch.matmul(normalization_mat.unsqueeze(1), kp3d.unsqueeze(-1))
    kp3d_canonical = kp3d_canonical.squeeze(-1)
    # Pad T from 3x4 mat to 4x4 mat
    normalization_mat = pad34_to_44(normalization_mat)
    return kp3d_canonical, normalization_mat

def compute_canonical_transform(kp3d, skeleton='bmc'):
    """
    Returns a transformation matrix T which when applied to kp3d performs the following
    operations:
    1) Center at the root (kp3d[:,0])
    2) Rotate such that the middle root bone points towards the y-axis
    3) Rotates around the x-axis such that the YZ-projection of the normal of the plane
    spanned by middle and index root bone points towards the z-axis
    """
    assert len(kp3d.shape) == 3, "kp3d need to be BS x 21 x 3"

    dev = kp3d.device
    bs = kp3d.shape[0]
    kp3d = kp3d.clone().detach()
    # Flip so that we compute the correct transformations below
    kp3d[0, :, 1] *= -1
    # Align root
    tx = kp3d[:, 0, 0]
    ty = kp3d[:, 0, 1]
    tz = kp3d[:, 0, 2]
    # Translation
    T_t = torch.zeros((bs, 3, 4), device=dev)
    T_t[:, 0, 3] = -tx
    T_t[:, 1, 3] = -ty
    T_t[:, 2, 3] = -tz
    T_t[:, 0, 0] = 1
    T_t[:, 1, 1] = 1
    T_t[:, 2, 2] = 1
    # Align middle root bone with -y-axis
    # x_axis = torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(bs, 3)  # FIXME
    y_axis = torch.tensor([[0.0, -1.0, 0.0]], device=dev).expand(bs, 3)
    v_mrb = normalize(kp3d[:, 3] - kp3d[:, 0])
    R_1 = get_alignment_mat(v_mrb, y_axis)
    # Align x-y plane along plane spanned by index and middle root bone of the hand
    # after R_1 has been applied to it
    v_irb = normalize(kp3d[:, 2] - kp3d[:, 0])
    normal = cross(v_mrb, v_irb).view(-1, 1, 3)
    normal_rot = torch.matmul(normal, R_1.transpose(1, 2)).view(-1, 3)
    z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(bs, 3)
    R_2 = get_alignment_mat(normal_rot, z_axis)
    # Include the flipping into the transformation
    T_t[0, 1, 1] = -1
    # Compute the canonical transform
    T = torch.bmm(R_2, torch.bmm(R_1, T_t))
    return T
