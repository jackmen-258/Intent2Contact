import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# 处理点云数据
from .networks.pointnet import Pointnet, ResnetBlockFC
from .networks.pointnet2 import Pointnet2

def normalize_vector(v):
    # v->[batch, n_points, 3]
    batch, n_points = v.shape[:2]
    # 计算L2范数
    v_mag = torch.norm(v, p=2, dim=-1)

    # 创建一个非常小的正数 eps，用于避免在后续计算中除以零
    # torch.autograd.Variable 确保 eps 是一个可微分的变量
    eps = torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v_mag.device))
    # 创建一个掩码 valid_mask，用于标记那些范数大于 eps 的向量
    # 这些向量将被归一化，而范数小于或等于 eps 的向量将被替换为默认值[1.0, 0.0, 0.0]
    valid_mask = (v_mag > eps).float().view(batch, n_points, 1)
    backup = torch.tensor([1.0, 0.0, 0.0]).float().to(v.device).view(1, 1, 3).expand(batch, n_points, 3)
    v_mag = torch.max(v_mag, eps)
    v_mag = v_mag.view(batch, n_points, 1).expand(batch, n_points, v.shape[2])
    # 将每个向量 v 除以其范数 v_mag，从而实现归一化
    v = v / v_mag
    ret = v * valid_mask + backup * (1 - valid_mask)

    return ret

# 潜在分布是指模型从输入数据中学习到的、用于生成新数据的内部表示
class LetentEncoder(nn.Module):
    def __init__(self, in_dim, dim, out_dim):
        super().__init__()
        # ResnetBlockFC 是一个全连接的残差块，用于提取输入特征的深层特征
        # size_h 是隐藏层的维度(in_dim->dim->dim)
        self.block = ResnetBlockFC(size_in=in_dim, size_out=dim, size_h=dim)
        # 全连接层 fc_mean 用于从残差块的输出中预测潜在分布的均值
        # 均值通常直接参与重构损失的计算，目标是使重构的数据尽可能接近原始数据
        self.fc_mean = nn.Linear(dim, out_dim)
        # 全连接层 fc_logstd 用于从残差块的输出中预测潜在分布的对数标准差
        # 在 VAE 中，对数标准差用于控制潜在空间的分布，使得生成的数据具有多样性
        self.fc_logstd = nn.Linear(dim, out_dim)

    def forward(self, x):
        # final_nl=True 表示在残差块的输出中应用非线性激活函数
        x = self.block(x, final_nl=True)
        return self.fc_mean(x), self.fc_logstd(x)


class ContactGenModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(ContactGenModel, self).__init__()
        self.cfg = cfg
        self.n_neurons = cfg.n_neurons # 256
        self.latentD = cfg.latentD # 16
        self.hc = cfg.pointnet_hc # 64
        self.object_feature = cfg.obj_feature # 6

        # 定义手部分的数量，并创建一个嵌入层来嵌入手部分的类别标签
        # nn.Embedding 用于将离散的输入映射到连续的向量空间
        # 第一个参数是字典大小，第二个参数是每个嵌入向量的维度
        self.num_parts = 16
        self.embed_class = nn.Embedding(self.num_parts, self.hc)
        
        encode_dim = self.hc
        self.obj_pointnet = Pointnet2(in_dim=self.object_feature, hidden_dim=self.hc, out_dim=self.hc)

        self.contact_encoder = Pointnet(in_dim=encode_dim + 1, hidden_dim=self.hc, out_dim=self.hc)
        self.part_encoder = Pointnet(in_dim=encode_dim + self.latentD + self.hc, hidden_dim=self.hc, out_dim=self.hc)
        self.uv_encoder = Pointnet(in_dim=encode_dim + self.hc + 3, hidden_dim=self.hc, out_dim=self.hc)
       
        self.contact_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        self.part_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        self.uv_latent = LetentEncoder(in_dim=self.hc, dim=self.n_neurons, out_dim=self.latentD)
        
        # C->[N, 1]
        # P->[N, B]
        # D->[N, 3]
        self.contact_decoder = Pointnet(in_dim=encode_dim + self.latentD, hidden_dim=self.hc, out_dim=1)
        self.part_decoder = Pointnet(in_dim=encode_dim + self.latentD + self.latentD, hidden_dim=self.hc, out_dim=self.num_parts)
        self.uv_decoder = Pointnet(in_dim=self.hc + encode_dim + self.latentD, hidden_dim=self.hc, out_dim=3)

    # 关于占位符：Pointnet 返回两个输出 x 和 self.pool(x, dim=1)
    def encode(self, obj_cond, contacts_object, partition_object, uv_object):
        _, contact_latent = self.contact_encoder(torch.cat([obj_cond, contacts_object], -1))
        # 计算潜在分布 contact_latent 的均值和标准差
        contact_mu, contact_std = self.contact_latent(contact_latent)
        # 创建一个高斯分布，均值为 contact_mu，标准差为 e^contact_std 
        z_contact = torch.distributions.normal.Normal(contact_mu, torch.exp(contact_std))
        # 从上述高斯分布中采样，获取采样值 z_s_contact
        # rsample()方法采用重参数化技巧，使得生成的样本具有可导性
        # z_s_contact->[batch, n_points, latentD]
        z_s_contact = z_contact.rsample()

        # 假设有3个点，每个点有4个可能的类别
        # partition_object = torch.tensor([
        #    [0.1, 0.7, 0.1, 0.1],  # 点1属于类别1的概率最高
        #    [0.5, 0.2, 0.3, 0.0],  # 点2属于类别0的概率最高
        #    [0.0, 0.0, 0.8, 0.2]   # 点3属于类别2的概率最高
        # ])
        # partition_object.argmax(dim=-1) 获取每个点的手部分索引
        partition_feat = self.embed_class(partition_object.argmax(dim=-1))
        _, part_latent = self.part_encoder(torch.cat([obj_cond, z_s_contact.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1), partition_feat], -1))
        part_mu, part_std = self.part_latent(part_latent)
        z_part = torch.distributions.normal.Normal(part_mu, torch.exp(part_std))
        _, uv_latent = self.uv_encoder(torch.cat([obj_cond, partition_feat, uv_object], -1))
        uv_mu, uv_std = self.uv_latent(uv_latent)
        z_uv = torch.distributions.normal.Normal(uv_mu, torch.exp(uv_std))
        z_s_part = z_part.rsample()
        z_s_uv = z_uv.rsample()
        
        return z_contact, z_part, z_uv, z_s_contact, z_s_part, z_s_uv

    def decode(self, z_contact, z_part, z_uv, obj_cond, gt_partition_object=None):
 
        z_contact = z_contact.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        contacts_object, _ = self.contact_decoder(torch.cat([z_contact, obj_cond], -1))
        contacts_object = torch.sigmoid(contacts_object)

        z_part = z_part.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        partition_object, _ = self.part_decoder(torch.cat([z_part, obj_cond, z_contact], -1))
        z_uv = z_uv.unsqueeze(dim=1).repeat(1, obj_cond.shape[1], 1)
        
        if gt_partition_object is not None:
            partition_feat = self.embed_class(gt_partition_object.argmax(dim=-1))
        else:
            partition_object_ = F.one_hot(partition_object.detach().argmax(dim=-1), num_classes=16)
            partition_feat = self.embed_class(partition_object_.argmax(dim=-1))
        # 需要将离散的类别标签 partition_object 转换为连续的嵌入向量 partition_feat 以进行拼接
        uv_object, _ = self.uv_decoder(torch.cat([z_uv, obj_cond, partition_feat], -1))
        uv_object = normalize_vector(uv_object)
        return contacts_object, partition_object, uv_object
    
    def forward(self, verts_object, feat_object, contacts_object, partition_object, uv_object, **kwargs):
        # Pointnet2 提取物体点云特征
        obj_cond = self.obj_pointnet(torch.cat([verts_object, feat_object], -1))
        # 调用 encode() 方法进行编码，生成潜在分布(z_contact, z_part, z_uv)和采样(z_s_contact, z_s_part, z_s_uv)
        z_contact, z_part, z_uv, z_s_contact, z_s_part, z_s_uv = self.encode(obj_cond, contacts_object, partition_object, uv_object)
        # 将潜在分布的均值和标准差收集到字典 results 中
        results = {'mean_contact': z_contact.mean, 'std_contact': z_contact.scale,
                   'mean_part': z_part.mean, 'std_part': z_part.scale,
                   'mean_uv': z_uv.mean, 'std_uv': z_uv.scale}
        # 调用 decode() 方法进行解码，生成预测的 contact map, part map, direction map
        contacts_pred, partition_pred, uv_pred = self.decode(z_s_contact, z_s_part, z_s_uv, obj_cond, partition_object)

        # 将解码得到的预测结果更新到 results 字典中
        results.update({'contacts_object': contacts_pred,
                        'partition_object': partition_pred,
                        'uv_object': uv_pred})
        return results

    # sample() 方法用于从模型中生成新的样本
    def sample(self, verts_object, feat_object):
        bs = verts_object.shape[0]
        dtype = verts_object.dtype
        device = verts_object.device
        self.eval()
        with torch.no_grad():
            obj_cond = self.obj_pointnet(torch.cat([verts_object, feat_object], -1))
            # 使用 NumPy 从标准正态分布中生成随机样本 z_gen_contact
            # 将生成的样本转换为 PyTorch 张量，并设置数据类型和设备
            z_gen_contact = np.random.normal(0., 1., size=(bs, self.latentD))
            z_gen_contact = torch.tensor(z_gen_contact,dtype=dtype).to(device)

            z_gen_part = np.random.normal(0., 1., size=(bs, self.latentD))
            z_gen_part = torch.tensor(z_gen_part,dtype=dtype).to(device)

            z_gen_uv = np.random.normal(0., 1., size=(bs, self.latentD))
            z_gen_uv = torch.tensor(z_gen_uv,dtype=dtype).to(device)

            return self.decode(z_gen_contact, z_gen_part, z_gen_uv, obj_cond)