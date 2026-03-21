import pickle
import torch
from torch.nn import functional as F
import pytorch3d.ops
from pytorch3d.structures import Meshes
from .diffcontact import calculate_contact_capsule

class HandObject:
    def __init__(self, device, face_path="assets/closed_mano_faces.pkl"):
        with open(face_path, 'rb') as f:
            self.hand_faces = torch.Tensor(pickle.load(f)).unsqueeze(0).to(device)

    def forward(self, hand_verts, obj_verts, obj_vn):
        hand_mesh = Meshes(verts=hand_verts, faces=self.hand_faces)

        obj_contact_target, _ = calculate_contact_capsule(
            hand_mesh.verts_padded(),
            hand_mesh.verts_normals_padded(),
            obj_verts, obj_vn,
            caps_top=0.0005, 
            caps_bot=-0.0015,
            caps_rad=0.003,
            caps_on_hand=False
        )

        data_out = {
            "verts_object": obj_verts,
            "feat_object": obj_vn, 
            "contacts_object": obj_contact_target,
        }
        return data_out
