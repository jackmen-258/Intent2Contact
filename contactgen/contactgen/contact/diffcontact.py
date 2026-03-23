import torch
import pytorch3d.ops

# 这些函数共同完成了从手部和物体的几何数据中计算接触点的任务
# capsule_sdf 函数首先计算 sdf，然后 sdf_to_contact 函数将 sdf 转换为 contact value
# 最后 calculate_contact_capsule 函数使用这些 contact value 来生成 contact map

# Capsule SDF 可以想象成在每个网格顶点上“放置”一个胶囊
def capsule_sdf(mesh_verts, mesh_normals, query_points, query_normals, caps_rad, caps_top, caps_bot, foreach_on_mesh):
    """
    Find the SDF of query points to mesh verts
    Capsule SDF formulation from https://iquilezles.org/www/articles/distfunctions/distfunctions.htm

    :param mesh_verts: (batch, V, 3)
    :param mesh_normals: (batch, V, 3)
    :param query_points: (batch, Q, 3)
    :param caps_rad: scalar, radius of capsules
    :param caps_top: scalar, distance from mesh to top of capsule
    :param caps_bot: scalar, distance from mesh to bottom of capsule
    :param foreach_on_mesh: boolean, foreach point on mesh find closest query (V), or foreach query find closest mesh (Q)
    :return: normalized sdf + 1 (batch, V or Q)
    """
    # TODO implement normal check?
    if foreach_on_mesh:     # Foreach mesh vert, find closest query point
        # K=1 表示只找最近的一个点，return_nn=True 表示返回最近邻的索引(nearest_idx)和位置(nearest_idx)
        # 返回值：
        # knn_dists：最近邻点的距离数组，形状为 (batch, Q, K)
        # nearest_idx：最近邻点的索引数组，形状为 (batch, Q, K)
        # nearest_pos：最近邻点的位置数组，形状为 (batch, Q, K, 3)
        knn_dists, nearest_idx, nearest_pos = pytorch3d.ops.knn_points(mesh_verts, query_points, K=1, return_nn=True)   # TODO should attract capsule middle?

        # 对于每个网格顶点，沿着其法线方向向上移动 caps_top 的距离，从而确定胶囊顶部的位置
        capsule_tops = mesh_verts + mesh_normals * caps_top
        # 对于每个网格顶点，沿着其法线方向向下移动 caps_bot 的距离，从而确定胶囊底部的位置
        capsule_bots = mesh_verts + mesh_normals * caps_bot
        # nearest_pos[:, :, 0, :]表示所有批次的所有顶点的最近邻点(K=1)的位置
        delta_top = nearest_pos[:, :, 0, :] - capsule_tops
        # batched_index_select 根据 nearest_idx 从 query_normals 中选择出与每个网格顶点最近的查询点的法线
        # dim=2将每个顶点的三个坐标分量的点积结果进行求和(normals->[batch, V/Q, 3(x, y, z)])
        normal_dot = torch.sum(mesh_normals * batched_index_select(query_normals, 1, nearest_idx.squeeze(2)), dim=2)

    else:   # Foreach query vert, find closest mesh point
        knn_dists, nearest_idx, nearest_pos = pytorch3d.ops.knn_points(query_points, mesh_verts, K=1, return_nn=True)   # TODO should attract capsule middle?
        # 根据 nearest_idx 从 mesh_verts 中选择出与查询点最近的网格顶点
        closest_mesh_verts = batched_index_select(mesh_verts, 1, nearest_idx.squeeze(2))    # Shape (batch, V, 3)
        # 根据 nearest_idx 从 mesh_normals 中选择出与查询点最近的网格顶点的法线
        closest_mesh_normals = batched_index_select(mesh_normals, 1, nearest_idx.squeeze(2))    # Shape (batch, V, 3)

        capsule_tops = closest_mesh_verts + closest_mesh_normals * caps_top  # Coordinates of the top focii of the capsules (batch, V, 3)
        capsule_bots = closest_mesh_verts + closest_mesh_normals * caps_bot
        delta_top = query_points - capsule_tops
        normal_dot = torch.sum(query_normals * closest_mesh_normals, dim=2)

    bot_to_top = capsule_bots - capsule_tops  # Vector from capsule bottom to top
    along_axis = torch.sum(delta_top * bot_to_top, dim=2)   # Dot product
    top_to_bot_square = torch.sum(bot_to_top * bot_to_top, dim=2)
    # torch.clamp 函数确保 h 的值在 0 到 1 之间(对应于胶囊轴线的端点)
    h = torch.clamp(along_axis / top_to_bot_square, 0, 1)   # Could avoid NaNs with offset in division here
    # 计算查询点到胶囊轴线的距离
    dist_to_axis = torch.norm(delta_top - bot_to_top * h.unsqueeze(2), dim=2)   # Distance to capsule centerline

    # Capsule SDF = dist_to_axis - caps_rad
    # Normalized SDF + 1 = (dist_to_axis - caps_rad) / caps_rad + 1 = dist_to_axis / caps_rad
    # Normalized SDF->[-1, 1], Normalized SDF + 1->[0, 1]
    return dist_to_axis / caps_rad, normal_dot  # (Normalized SDF)+1 0 on endpoint, 1 on edge of capsule


# sdf_to_contact 函数提供了多种将 sdf 转换为接触值的方法
# 不同的转换方法会产生不同的接触值响应曲线，这些曲线可以调整接触值对距离变化的敏感度
def sdf_to_contact(sdf, dot_normal, method=0):
    """
    Transform normalized SDF into some contact value
    :param sdf: NORMALIZED SDF, 1 is surface of object
    :param method: select method
    :return: contact (batch, S, 1)
    """
    if method == 0:
        c = 1 / (sdf + 0.0001)   # Exponential dropoff
    elif method == 1:
        c = -sdf + 2    # Linear dropoff
    elif method == 2:
        c = 1 / (sdf + 0.0001)   # Exponential dropoff
        c = torch.pow(c, 2)
    elif method == 3:
        c = torch.sigmoid(-sdf + 2.5)
    elif method == 4:
        c = (-dot_normal/2+0.5) / (sdf + 0.0001)   # Exponential dropoff with sharp normal
    elif method == 5:
        c = 1 / (sdf + 0.0001)   # Proxy for other stuff

    # 将接触值 c 限制在0和1之间，确保它保持在接触表示的有效范围内
    return torch.clamp(c, 0.0, 1.0)


def calculate_contact_capsule(hand_verts, hand_normals, object_verts, object_normals,
                              caps_top=0.0005, caps_bot=-0.0015, caps_rad=0.001, caps_on_hand=False, contact_norm_method=0):
    """
    Calculates contact maps on object and hand.
    :param hand_verts: (batch, V, 3)
    :param hand_normals: (batch, V, 3)
    :param object_verts: (batch, O, 3)
    :param object_normals: (batch, O, 3)
    :param caps_top: ctop, distance to top capsule center
    :param caps_bot: cbot, distance to bottom capsule center
    :param caps_rad: crad, radius of the contact capsule
    :param caps_on_hand: are contact capsules placed on hand or object vertices
    :param contact_norm_method: select a distance-to-contact function
    :return: object contact (batch, O, 1), hand contact (batch, V, 1)
    """
    # caps_on_hand 决定胶囊是放置在手部顶点上还是物体顶点上
    # 对于 capsule_sdf 函数，无论 foreach_on_mesh 设置为 True 或 False，都是以 mesh_verts/mesh_normals 构建胶囊体
    if caps_on_hand:
        sdf_obj, dot_obj = capsule_sdf(hand_verts, hand_normals, object_verts, object_normals, caps_rad, caps_top, caps_bot, False)
        sdf_hand, dot_hand = capsule_sdf(hand_verts, hand_normals, object_verts, object_normals, caps_rad, caps_top, caps_bot, True)
    else:
        sdf_obj, dot_obj = capsule_sdf(object_verts, object_normals, hand_verts, hand_normals, caps_rad, caps_top, caps_bot, True)
        sdf_hand, dot_hand = capsule_sdf(object_verts, object_normals, hand_verts, hand_normals, caps_rad, caps_top, caps_bot, False)

    obj_contact = sdf_to_contact(sdf_obj, dot_obj, method=contact_norm_method) # * (dot_obj/2+0.5) # TODO dotting contact normal
    hand_contact = sdf_to_contact(sdf_hand, dot_hand, method=contact_norm_method)# * (dot_hand/2+0.5)

    # print('oshape, nshape', obj_contact.shape, (dot_obj/2+0.5).shape)##

    return obj_contact.unsqueeze(2), hand_contact.unsqueeze(2)


def batched_index_select(t, dim, inds):
    """
    Helper function to extract batch-varying indicies along array
    :param t: array to select from
    :param dim: dimension to select along
    :param inds: batch-vary indicies
    :return:
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out