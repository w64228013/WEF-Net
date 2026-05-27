import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import SwinUNETR


features = None


def get_features_hook(module, input, output):
    global features
    features = output


def get_subjective_info(alpha, evidence, classes):
    S = alpha.sum(dim=1, keepdim=True)
    b = evidence / S
    u = classes / S
    return b, u


def get_evidence_info(belief, uncertainty, classes):
    S_a = 2 / uncertainty
    e_a = torch.mul(belief, S_a)
    alpha_a = e_a + 1
    return alpha_a, e_a


def ds_combination_rule(b1, u1, b2, u2):
    b1b2 = torch.mul(b1, b2)
    b1u2 = torch.mul(b1, u2)
    b2u1 = torch.mul(b2, u1)
    u1u2 = torch.mul(u1, u2)

    C = (1 - (torch.mul(b1[:, 0], b2[:, 1]) + torch.mul(b2[:, 0], b1[:, 1]))).unsqueeze(1)

    combi_b = (b1b2 + b1u2 + b2u1) / C
    combi_u = u1u2 / C

    return combi_b, combi_u


class SwinEDL_03_clinic(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = SwinUNETR(
            img_size=(128, 128, 64),
            in_channels=3,
            out_channels=3,
            spatial_dims=3,
            feature_size=24,
            drop_rate=0.0,
            attn_drop_rate=0.0
        )
        self.num_classes = num_classes
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.3)
        self.FC = nn.Linear(24, self.num_classes)

        self.clinic_FC = nn.Linear(7, self.num_classes)

        _ckpt_path = os.environ.get(
            'BRAIN_SWIN_CKPT',
            '/public/home/b_tyxia/code/stroke_zcy/Brain_Swin_UNETR.pth')
        if os.path.exists(_ckpt_path):
            checkpoint = torch.load(_ckpt_path, map_location='cpu')
            new_state_dict = {}
            for k, v in checkpoint.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict=False)
        else:
            print(f'[WARN] pretrained ckpt not found, train from scratch: {_ckpt_path}')

    def forward(self, x1, x2, clinic_x):
        device = x1.device

        hook = self.model.decoder1.register_forward_hook(get_features_hook)
        out = self.model(x1)
        hook.remove()
        x1 = self.pool(features)
        x1 = self.dropout(x1)
        x1 = torch.flatten(x1, 1)
        x1 = x1.to(device)
        x1 = self.FC(x1)

        hook = self.model.decoder1.register_forward_hook(get_features_hook)
        out = self.model(x2)
        hook.remove()
        x2 = self.pool(features)
        x2 = self.dropout(x2)
        x2 = torch.flatten(x2, 1)
        x2 = x2.to(device)
        x2 = self.FC(x2)

        clinic_x = self.clinic_FC(clinic_x)

        # [N, num_classes]
        alpha_list = []
        evidence_list = []
        first_evidence = F.softplus(x1)
        alpha_list.append(first_evidence + 1)
        evidence_list.append(first_evidence)
        second_evidence = F.softplus(x2)
        alpha_list.append(second_evidence + 1)
        evidence_list.append(second_evidence)

        b1, u1 = get_subjective_info(alpha_list[0], first_evidence, self.num_classes)
        b2, u2 = get_subjective_info(alpha_list[1], second_evidence, self.num_classes)

        cmobi_b, combi_u = ds_combination_rule(b1, u1, b2, u2)
        combination_bu = (cmobi_b, combi_u)
        combi_alpha, _ = get_evidence_info(combination_bu[0], combination_bu[1], self.num_classes)

        alpha_list.append(combi_alpha)
        combi_evidence = combi_alpha - 1
        evidence_list.append(combi_evidence)

        #######Clinic########
        clinic_evidence = F.softplus(clinic_x)
        clinic_alpha = clinic_evidence + 1
        clinic_b, clinic_u = get_subjective_info(clinic_alpha, clinic_evidence, self.num_classes)
        cmobi2_b, combi2_u = ds_combination_rule(b1, u1, clinic_b, clinic_u)
        combi2_alpha, _ = get_evidence_info(cmobi2_b, combi2_u, self.num_classes)
        combi2_evidence = combi2_alpha - 1
        evidence_list.append(combi2_evidence)

        return evidence_list
