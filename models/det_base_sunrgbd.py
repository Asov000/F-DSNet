from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import math
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import cfg
from datasets.dataset_info import KITTICategory

from models.model_util import get_box3d_corners_helper
from models.model_util import huber_loss

from models.common import Conv1d, Conv2d, DeConv1d, init_params
from models.common import softmax_focal_loss_ignore, get_accuracy

from ops.query_depth_point.query_depth_point import QueryDepthPoint
from ops.pybind11.box_ops_cc import rbbox_iou_3d_pair

# from utils.box_util import box3d_iou_pair # slow, not recommend

from models.box_transform import size_decode, size_encode, center_decode, center_encode, angle_decode, angle_encode
from datasets.dataset_info import DATASET_INFO

class Res2NetBottleneck1D(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, downsample=None, stride=1,
                 scales=4, groups=1, norm_layer=None):
        super().__init__()
        if planes % scales != 0:
            raise ValueError('planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        bottleneck_planes = groups * planes
        self.conv1 = nn.Conv1d(inplanes, bottleneck_planes,
                               kernel_size=1, stride=stride, bias=False)
        self.bn1 = norm_layer(bottleneck_planes)

        self.conv2 = nn.ModuleList([
            nn.Conv1d(bottleneck_planes // scales,
                      bottleneck_planes // scales,
                      kernel_size=3, padding=1, groups=groups, bias=False)
            for _ in range(scales - 1)
        ])
        self.bn2 = nn.ModuleList([
            norm_layer(bottleneck_planes // scales)
            for _ in range(scales - 1)
        ])

        self.conv3 = nn.Conv1d(bottleneck_planes,
                               planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))

        xs = torch.chunk(out, self.scales, dim=1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            else:
                y = self.conv2[s - 1](xs[s] + (ys[-1] if s > 1 else 0))
                ys.append(self.relu(self.bn2[s - 1](y)))

        out = torch.cat(ys, dim=1)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.relu(out + identity)
# ----------------------------- SE Channel Attention -----------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block that works with both 1‑D (B,C,L) and 2‑D/4‑D (B,C,H,W) tensors."""
    def __init__(self, channel: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.size()[:2]
        if x.dim() == 3:  # (B,C,L)
            y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        else:            # (B,C,H,W) or (B,C,H)
            y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, *([1] * (x.dim() - 2)))
        return x * y


# single scale PointNet module
class PointNetModule(nn.Module):
    def __init__(self, Infea, mlp, dist, nsample, use_xyz=True, use_feature=True):
        super(PointNetModule, self).__init__()
        self.dist = dist
        self.nsample = nsample
        self.use_xyz = use_xyz

        if Infea > 0:
            use_feature = True
        else:
            use_feature = False

        self.use_feature = use_feature

        self.query_depth_point = QueryDepthPoint(dist, nsample)

        if self.use_xyz:
            self.conv1 = Conv2d(Infea + 3, mlp[0], 1)
        else:
            self.conv1 = Conv2d(Infea, mlp[0], 1)

        self.conv2 = Conv2d(mlp[0], mlp[1], 1)
        self.conv3 = Conv2d(mlp[1], mlp[2], 1)

        # --- SE block ---

        init_params([self.conv1[0], self.conv2[0], self.conv3[0]], 'kaiming_normal')
        init_params([self.conv1[1], self.conv2[1], self.conv3[1]], 1)

    def forward(self, pc, feat, new_pc=None):
        batch_size = pc.size(0)

        npoint = new_pc.shape[2]
        k = self.nsample

        indices, num = self.query_depth_point(pc, new_pc)  # b*npoint*nsample

        assert indices.data.max() < pc.shape[2] and indices.data.min() >= 0
        grouped_pc = None
        grouped_feature = None

        if self.use_xyz:
            grouped_pc = torch.gather(
                pc, 2,
                indices.view(batch_size, 1, npoint * k).expand(-1, 3, -1)
            ).view(batch_size, 3, npoint, k)

            grouped_pc = grouped_pc - new_pc.unsqueeze(3)

        if self.use_feature:
            grouped_feature = torch.gather(
                feat, 2,
                indices.view(batch_size, 1, npoint * k).expand(-1, feat.size(1), -1)
            ).view(batch_size, feat.size(1), npoint, k)

        if self.use_feature and self.use_xyz:
            grouped_feature = torch.cat([grouped_pc, grouped_feature], 1)
        elif self.use_xyz:
            grouped_feature = grouped_pc.contiguous()

        grouped_feature = self.conv1(grouped_feature)
        grouped_feature = self.conv2(grouped_feature)
        grouped_feature = self.conv3(grouped_feature)

        # --- Apply SE attention ---

        valid = (num > 0).view(batch_size, 1, -1, 1)
        grouped_feature = grouped_feature * valid.float()

        return grouped_feature


# multi-scale PointNet module
class PointNetFeat(nn.Module):
    def __init__(self, input_channel=3, num_vec=0):
        super(PointNetFeat, self).__init__()


        self.num_vec = num_vec
        u = cfg.DATA.HEIGHT_HALF
        assert len(u) == 5
        self.pointnet1 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[0], 128, use_xyz=True, use_feature=False)

        self.pointnet2 = PointNetModule(
            input_channel - 3, [64, 64, 128], u[1], 128, use_xyz=True, use_feature=False)

        self.pointnet3 = PointNetModule(
            input_channel - 3, [128, 128, 256], u[2], 256, use_xyz=True, use_feature=False)

        self.pointnet4 = PointNetModule(
            input_channel - 3, [256, 256, 512], u[3], 256, use_xyz=True, use_feature=False)

        self.pointnet5 = PointNetModule(
            input_channel - 3, [256, 256, 512], u[4], 256, use_xyz=True, use_feature=False)

    def forward(self, point_cloud, sample_pc, feat=None, one_hot_vec=None):
        pc = point_cloud
        pc1, pc2, pc3, pc4, pc5 = sample_pc

        # -------- PointNet backbone --------
        feat1 = self.pointnet1(pc, feat, pc1)
        feat1, _ = torch.max(feat1, -1)

        feat2 = self.pointnet2(pc, feat, pc2)
        feat2, _ = torch.max(feat2, -1)

        feat3 = self.pointnet3(pc, feat, pc3)
        feat3, _ = torch.max(feat3, -1)

        feat4 = self.pointnet4(pc, feat, pc4)
        feat4, _ = torch.max(feat4, -1)

        feat5 = self.pointnet5(pc, feat, pc5)
        feat5, _ = torch.max(feat5, -1)

        # -------- Append category vector (one‑hot or zeros) --------
        if self.num_vec > 0:
            if one_hot_vec is None:
                # create zero vectors so that channel count is consistent
                bz = pc.size(0)
                zeros_template = pc.new_zeros(bz, self.num_vec, 1)
                repeat = lambda f: zeros_template.expand(-1, -1, f.shape[-1])
                feat1 = torch.cat([feat1, repeat(feat1)], 1)
                feat2 = torch.cat([feat2, repeat(feat2)], 1)
                feat3 = torch.cat([feat3, repeat(feat3)], 1)
                feat4 = torch.cat([feat4, repeat(feat4)], 1)
                feat5 = torch.cat([feat5, repeat(feat5)], 1)
            else:
                assert self.num_vec == one_hot_vec.shape[1]
                expanded = lambda f: one_hot_vec.unsqueeze(-1).expand(-1, -1, f.shape[-1])
                feat1 = torch.cat([feat1, expanded(feat1)], 1)
                feat2 = torch.cat([feat2, expanded(feat2)], 1)
                feat3 = torch.cat([feat3, expanded(feat3)], 1)
                feat4 = torch.cat([feat4, expanded(feat4)], 1)
                feat5 = torch.cat([feat5, expanded(feat5)], 1)

        return feat1, feat2, feat3, feat4, feat5


# FCN
class ConvFeatNet(nn.Module):
    def __init__(self, i_c=128, num_vec=10):
        super(ConvFeatNet, self).__init__()

        def _down(in_c, out_c, s):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm1d(out_c))

        # Block 1: initial convolution
        self.block1_conv1 = Conv1d(i_c + num_vec, 64, 3, 1, 1)

        # Block 2
        self.block2 = Res2NetBottleneck1D(
            inplanes=64, planes=32, stride=2, scales=4,
            downsample=_down(64, 128, 2))
        self.block2_merge = Conv1d(128 + 128 + num_vec, 128, 1, 1)

        # Block 3
        self.block3 = Res2NetBottleneck1D(
            inplanes=128, planes=64, stride=2, scales=4,
            downsample=_down(128, 256, 2))
        self.block3_merge = Conv1d(256 + 256 + num_vec, 256, 1, 1)

        # Block 4
        self.block4 = Res2NetBottleneck1D(
            inplanes=256, planes=128, stride=2, scales=4,
            downsample=_down(256, 512, 2))
        self.block4_merge = Conv1d(512 + 512 + num_vec, 512, 1, 1)

        # Deconvolution layers
        self.block4_deconv = DeConv1d(512, 256, 4, 4, 0)
        self.block3_deconv = DeConv1d(256, 256, 2, 2, 0)
        self.block2_deconv = DeConv1d(128, 256, 1, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2, x3, x4):
        # ★ block-1
        x = self.block1_conv1(x1)

        # ★ block-2
        x = self.block2(x)
        x = self.block2_merge(torch.cat([x, x2], 1))
        f2 = x     # 128 ch

        # ★ block-3
        x = self.block3(x)
        x = self.block3_merge(torch.cat([x, x3], 1))
        f3 = x     # 256 ch

        # ★ block-4
        x = self.block4(x)
        x = self.block4_merge(torch.cat([x, x4], 1))
        f4 = x     # 512 ch

        # ------------- 解码 / 上采样 -----------------
        d1 = self.block2_deconv(f2)
        d2 = self.block3_deconv(f3)[..., :d1.shape[-1]]
        d3 = self.block4_deconv(f4)[..., :d1.shape[-1]]

        # 拼接输出 (B, 1024, N)
        return torch.cat([d1, d2, d3,d1], 1)



# the whole pipeline
class PointNetDet(nn.Module):
    def __init__(self, input_channel=3, num_vec=0, num_classes=2):
        super(PointNetDet, self).__init__()


        dataset_name = cfg.DATA.DATASET_NAME
        assert dataset_name in DATASET_INFO
        self.category_info = DATASET_INFO[dataset_name]

        self.num_size_cluster = len(self.category_info.CLASSES)
        self.mean_size_array = self.category_info.MEAN_SIZE_ARRAY

        self.feat_net = PointNetFeat(input_channel, num_vec)
        self.conv_net = ConvFeatNet(128, num_vec)

        self.num_classes = num_classes

        num_bins = cfg.DATA.NUM_HEADING_BIN
        self.num_bins = num_bins

        output_size = 3 + num_bins * 2 + self.num_size_cluster * 4

        self.reg_out = nn.Conv1d(1024, output_size, 1)
        self.cls_out = nn.Conv1d(1024, 2, 1)
        self.ctr_head = nn.Sequential(
            nn.Conv1d(1024, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 1, 1)  # logits
        )
        self.relu = nn.ReLU(True)

        nn.init.kaiming_uniform_(self.cls_out.weight, mode='fan_in')
        nn.init.kaiming_uniform_(self.reg_out.weight, mode='fan_in')
        nn.init.normal_(self.ctr_head[-1].weight, 0, 0.01)
        nn.init.constant_(self.ctr_head[-1].bias, 0.0)

        self.cls_out.bias.data.zero_()
        self.reg_out.bias.data.zero_()

    def _slice_output(self, output):

        batch_size = output.shape[0]

        num_bins = self.num_bins
        num_sizes = self.num_size_cluster

        center = output[:, 0:3].contiguous()

        heading_scores = output[:, 3:3 + num_bins].contiguous()

        heading_res_norm = output[:, 3 + num_bins:3 + num_bins * 2].contiguous()

        size_scores = output[:, 3 + num_bins * 2:3 + num_bins * 2 + num_sizes].contiguous()

        size_res_norm = output[:, 3 + num_bins * 2 + num_sizes:].contiguous()
        size_res_norm = size_res_norm.view(batch_size, num_sizes, 3)

        return center, heading_scores, heading_res_norm, size_scores, size_res_norm

    def get_center_loss(self, pred_offsets, gt_offsets):

        center_dist = torch.norm(gt_offsets - pred_offsets, 2, dim=-1)
        center_loss = huber_loss(center_dist, delta=3.0)

        return center_loss

    def get_heading_loss(self, heading_scores, heading_res_norm, heading_class_label, heading_res_norm_label):

        heading_class_loss = F.cross_entropy(heading_scores, heading_class_label)

        # b, NUM_HEADING_BIN -> b, 1
        heading_res_norm_select = torch.gather(heading_res_norm, 1, heading_class_label.view(-1, 1))

        heading_res_norm_loss = huber_loss(
            heading_res_norm_select.squeeze(1) - heading_res_norm_label, delta=1.0)

        return heading_class_loss, heading_res_norm_loss

    def get_size_loss(self, size_scores, size_res_norm, size_class_label, size_res_label_norm):
        batch_size = size_scores.shape[0]
        size_class_loss = F.cross_entropy(size_scores, size_class_label)

        # b, NUM_SIZE_CLUSTER, 3 -> b, 1, 3
        size_res_norm_select = torch.gather(size_res_norm, 1,
                                            size_class_label.view(batch_size, 1, 1).expand(
                                                batch_size, 1, 3))

        size_norm_dist = torch.norm(
            size_res_label_norm - size_res_norm_select.squeeze(1), 2, dim=-1)

        size_res_norm_loss = huber_loss(size_norm_dist, delta=1.0)

        return size_class_loss, size_res_norm_loss

    def get_corner_loss(self, preds, gts):

        center_label, heading_label, size_label = gts
        center_preds, heading_preds, size_preds = preds

        corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label)
        corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label + np.pi, size_label)

        corners_3d_pred = get_box3d_corners_helper(center_preds, heading_preds, size_preds)

        # N, 8, 3
        corners_dist = torch.min(
            torch.norm(corners_3d_pred - corners_3d_gt, 2, dim=-1).mean(-1),
            torch.norm(corners_3d_pred - corners_3d_gt_flip, 2, dim=-1).mean(-1))
        # corners_dist = torch.norm(corners_3d_pred - corners_3d_gt, 2, dim=-1)
        corners_loss = huber_loss(corners_dist, delta=1.0)

        return corners_loss, corners_3d_gt

    def forward(self,
                data_dicts):
        point_cloud = data_dicts.get('point_cloud')
        one_hot_vec = data_dicts.get('one_hot')
        cls_label = data_dicts.get('cls_label')
        size_class_label = data_dicts.get('size_class')
        center_label = data_dicts.get('box3d_center')
        heading_label = data_dicts.get('box3d_heading')
        size_label = data_dicts.get('box3d_size')

        center_ref1 = data_dicts.get('center_ref1')
        center_ref2 = data_dicts.get('center_ref2')
        center_ref3 = data_dicts.get('center_ref3')
        center_ref4 = data_dicts.get('center_ref4')
        center_ref5 = data_dicts.get('center_ref5')

        batch_size = point_cloud.shape[0]

        object_point_cloud_xyz = point_cloud[:, :3, :].contiguous()
        if point_cloud.shape[1] > 3:
            object_point_cloud_i = point_cloud[:, [3], :].contiguous()
        else:
            object_point_cloud_i = None

        mean_size_array = torch.from_numpy(self.mean_size_array).type_as(point_cloud)

        feat1, feat2, feat3, feat4, feat5 = self.feat_net(
            object_point_cloud_xyz,
            [center_ref1, center_ref2, center_ref3, center_ref4, center_ref5],
            object_point_cloud_i,
            one_hot_vec)

        x = self.conv_net(feat1, feat2, feat3, feat4)

        cls_scores = self.cls_out(x)
        outputs = self.reg_out(x)
        ctr_scores = self.ctr_head(x)

        num_out = outputs.shape[2]
        output_size = outputs.shape[1]
        # b, c, n -> b, n, c
        cls_scores = cls_scores.permute(0, 2, 1).contiguous().view(-1, 2)
        outputs = outputs.permute(0, 2, 1).contiguous().view(-1, output_size)
        ctr_scores = ctr_scores.permute(0, 2, 1).contiguous().view(-1)  # ★

        center_ref2 = center_ref2.permute(0, 2, 1).contiguous().view(-1, 3)

        cls_probs = F.softmax(cls_scores, -1)  # (B*N,2)
        ctr_probs = torch.sigmoid(ctr_scores).view(batch_size, num_out)  # ★ (B,N)

        fg_scores = cls_probs[:, 1].view(batch_size, num_out) * ctr_probs  # ★
        cls_fg = fg_scores  # (B,N)
        cls_bg = 1.0 - cls_fg
        cls_probs = torch.stack((cls_bg, cls_fg), dim=-1).view(-1, 2)  # ★ (B*N,2) 扁平化


        if center_label is None:
            assert not self.training, 'Please provide labels for training.'

            det_outputs = self._slice_output(outputs)

            center_boxnet, heading_scores, heading_res_norm, size_scores, size_res_norm = det_outputs

            # decode
            heading_probs = F.softmax(heading_scores, -1)
            size_probs = F.softmax(size_scores, -1)

            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)

            center_preds = center_boxnet + center_ref2

            heading_preds = angle_decode(heading_res_norm, heading_pred_label, num_bins=self.num_bins)
            size_preds = size_decode(size_res_norm, mean_size_array, size_pred_label)

            # corner_preds = get_box3d_corners_helper(center_preds, heading_preds, size_preds)

            cls_probs = cls_probs.view(batch_size, -1, 2)
            center_preds = center_preds.view(batch_size, -1, 3)

            size_preds = size_preds.view(batch_size, -1, 3)
            size_probs = size_probs.view(batch_size, -1, self.num_size_cluster)

            heading_preds = heading_preds.view(batch_size, -1)
            heading_probs = heading_probs.view(batch_size, -1, self.num_bins)

            outputs = (cls_probs, center_preds, heading_preds, size_preds, heading_probs, size_probs)
            # outputs = (cls_probs, center_preds, heading_preds, size_preds)
            return outputs

        fg_idx = (cls_label.view(-1) == 1).nonzero().view(-1)

        assert fg_idx.numel() != 0

        outputs = outputs[fg_idx, :]
        center_ref2 = center_ref2[fg_idx]

        det_outputs = self._slice_output(outputs)

        center_boxnet, heading_scores, heading_res_norm, size_scores, size_res_norm = det_outputs

        heading_probs = F.softmax(heading_scores, -1)
        size_probs = F.softmax(size_scores, -1)

        # cls_loss = F.cross_entropy(cls_scores, mask_label, ignore_index=-1)
        cls_loss = softmax_focal_loss_ignore(cls_probs, cls_label.view(-1), ignore_idx=-1)

        # prepare label
        center_label = center_label.unsqueeze(1).expand(-1, num_out, -1).contiguous().view(-1, 3)[fg_idx]
        heading_label = heading_label.expand(-1, num_out).contiguous().view(-1)[fg_idx]
        size_label = size_label.unsqueeze(1).expand(-1, num_out, -1).contiguous().view(-1, 3)[fg_idx]
        size_class_label = size_class_label.expand(-1, num_out).contiguous().view(-1)[fg_idx]

        # encode regression targets
        center_gt_offsets = center_encode(center_label, center_ref2)
        heading_class_label, heading_res_norm_label = angle_encode(heading_label, num_bins=self.num_bins)
        size_res_label_norm = size_encode(size_label, mean_size_array, size_class_label)

        # loss calculation

        # center_loss
        center_loss = self.get_center_loss(center_boxnet, center_gt_offsets)

        # heading loss
        heading_class_loss, heading_res_norm_loss = self.get_heading_loss(
            heading_scores, heading_res_norm, heading_class_label, heading_res_norm_label)

        # size loss
        size_class_loss, size_res_norm_loss = self.get_size_loss(
            size_scores, size_res_norm, size_class_label, size_res_label_norm)

        # corner loss regulation
        center_preds = center_decode(center_ref2, center_boxnet)
        heading = angle_decode(heading_res_norm, heading_class_label, num_bins=self.num_bins)
        size = size_decode(size_res_norm, mean_size_array, size_class_label)

        corners_loss, corner_gts = self.get_corner_loss(
            (center_preds, heading, size),
            (center_label, heading_label, size_label)
        )

        BOX_LOSS_WEIGHT = cfg.LOSS.BOX_LOSS_WEIGHT
        CORNER_LOSS_WEIGHT = cfg.LOSS.CORNER_LOSS_WEIGHT
        HEAD_REG_WEIGHT = cfg.LOSS.HEAD_REG_WEIGHT
        SIZE_REG_WEIGHT = cfg.LOSS.SIZE_REG_WEIGHT
        pred_ctr = ctr_scores[fg_idx]   # (num_fg,)

        offset = torch.norm(center_label - center_ref2, dim=-1)  # ★ (num_fg,)
        alpha = 1.5
        centerness_target = torch.exp(-alpha * offset).clamp(1e-4, 1 - 1e-4).detach()

        ctr_loss = F.binary_cross_entropy_with_logits(pred_ctr, centerness_target)

        # Weighted sum of all losses
        loss = cls_loss + \
               BOX_LOSS_WEIGHT * (center_loss + heading_class_loss + size_class_loss +
                                  HEAD_REG_WEIGHT * heading_res_norm_loss +
                                  SIZE_REG_WEIGHT * size_res_norm_loss +
                                  CORNER_LOSS_WEIGHT * corners_loss) + \
               0.25 * ctr_loss

        # some metrics to monitor training status

        with torch.no_grad():

            # accuracy
            cls_prec = get_accuracy(cls_probs, cls_label.view(-1), ignore=-1)
            heading_prec = get_accuracy(heading_probs, heading_class_label.view(-1))
            size_prec = get_accuracy(size_probs, size_class_label.view(-1))

            # iou metrics
            heading_pred_label = torch.argmax(heading_probs, -1)
            size_pred_label = torch.argmax(size_probs, -1)

            heading_preds = angle_decode(heading_res_norm, heading_pred_label, num_bins=self.num_bins)
            size_preds = size_decode(size_res_norm, mean_size_array, size_pred_label)

            corner_preds = get_box3d_corners_helper(center_preds, heading_preds, size_preds)
            overlap = rbbox_iou_3d_pair(corner_preds.detach().cpu().numpy(), corner_gts.detach().cpu().numpy())

            iou2ds, iou3ds = overlap[:, 0], overlap[:, 1]
            iou2d_mean = iou2ds.mean()
            iou3d_mean = iou3ds.mean()
            iou3d_gt_mean = (iou3ds >= cfg.IOU_THRESH).mean()
            iou2d_mean = torch.tensor(iou2d_mean).type_as(cls_prec)
            iou3d_mean = torch.tensor(iou3d_mean).type_as(cls_prec)
            iou3d_gt_mean = torch.tensor(iou3d_gt_mean).type_as(cls_prec)

        losses = {
            'total_loss': loss,
            'cls_loss': cls_loss,
            'center_loss': center_loss,
            'head_cls_loss': heading_class_loss,
            'head_res_loss': heading_res_norm_loss,
            'size_cls_loss': size_class_loss,
            'size_res_loss': size_res_norm_loss,
            'corners_loss': corners_loss,
            'ctr_loss': ctr_loss,
        }

        metrics = {
            'cls_acc': cls_prec,
            'head_acc': heading_prec,
            'size_acc': size_prec,
            'IoU_2D': iou2d_mean,
            'IoU_3D': iou3d_mean,
            'IoU_' + str(cfg.IOU_THRESH): iou3d_gt_mean
        }

        return losses, metrics
