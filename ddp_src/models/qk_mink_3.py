import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

import math
import time

from models.backbone_kpconv.kpconv import KPFEncoder, PreprocessorGPU, compute_overlaps
from models.generic_reg_model import GenericRegModel
from models.losses.corr_loss import CorrCriterion
from models.losses.feature_loss import InfoNCELossFull, CircleLossFull
from models.transformer.position_embedding import PositionEmbeddingCoordsSine, PositionEmbeddingLearned
from models.transformer.transformers import TransformerCrossEncoderLayer, TransformerCrossEncoder
from utils.se3_torch import compute_rigid_transform, se3_transform_list, se3_inv, se3_transform
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences

from utils.ME_layers import get_norm_layer, get_res_block
# from lib.utils import kabsch_transformation_estimation

_EPS = 1e-6

"""
The reason for the bad performance might be the large number of conv layers which leads to extreme washout of parameters
"""

class SparseEnoder(ME.MinkowskiNetwork):
    CHANNELS = [None, 64, 128, 256]

    def __init__(self,cfg):

        in_channels = cfg.in_channels
        out_channels = cfg.out_channels
        bn_momentum = cfg.bn_momentum
        conv1_kernel_size = cfg.conv1_kernel_size
        norm_type = cfg.norm_type
        D = cfg.D

        ME.MinkowskiNetwork.__init__(self, D)

        NORM_TYPE = norm_type
        BLOCK_NORM_TYPE = norm_type
        CHANNELS = cfg.CHANNELS

    
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm1 = get_norm_layer(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.block1 = get_res_block(
            BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)

        self.norm2 = get_norm_layer(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2 = get_res_block(
                BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)

        self.norm3 = get_norm_layer(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3 = get_res_block(
                BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

    def forward(self, x, tgt_feature=False):
        # print("x", x.shape)
        skip_features = []
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out = self.block1(out_s1)

        # print("1: ", out.shape)
        # skip_features.append(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out = self.block2(out_s2)
        # print("2: ", out.shape)

        out_s3 = self.conv3(out)
        out_s3 = self.norm3(out_s3)
        out = self.block3(out_s3)

        return out, skip_features


class RegTR(GenericRegModel):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.verbose = False
        self.time_verbose = False
        self.normalize_feature = True

        ################################
        # Sparse Minkowski Encoder
        ################################
        self.encoder = SparseEnoder(cfg)

        #######################
        # Embeddings (To be used in the Cross Attention layers) 
        #######################
        self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.d_embed,scale=cfg.get('pos_emb_scaling', 1.0))

        #######################
        # Attention propagation
        #######################
        encoder_layer = TransformerCrossEncoderLayer(
            cfg.d_embed, cfg.nhead, cfg.d_feedforward, cfg.dropout,
            activation=cfg.transformer_act,
            normalize_before=cfg.pre_norm,
            sa_val_has_pos_emb=cfg.sa_val_has_pos_emb,
            ca_val_has_pos_emb=cfg.ca_val_has_pos_emb,
            attention_type=cfg.attention_type,
        )
        encoder_norm = nn.LayerNorm(cfg.d_embed) if cfg.pre_norm else None
        self.transformer_encoder = TransformerCrossEncoder(
            encoder_layer, cfg.num_encoder_layers, encoder_norm,
            return_intermediate=False)

        #######################
        # Losses
        #######################
        self.feature_criterion = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
        self.overlap_criterion = nn.BCEWithLogitsLoss()

        self.weight_dict = {}
        self.weight_dict['feature'] = cfg.wt_feature
        self.weight_dict['T'] = 1.0

        self.logger.info('Loss weighting: {}'.format(self.weight_dict))
        self.logger.info(
            f'Config: d_embed:{cfg.d_embed}, nheads:{cfg.nhead}, pre_norm:{cfg.pre_norm}, '
            f'use_pos_emb:{cfg.transformer_encoder_has_pos_emb}, '
            f'sa_val_has_pos_emb:{cfg.sa_val_has_pos_emb}, '
            f'ca_val_has_pos_emb:{cfg.ca_val_has_pos_emb}'
        )

    def _get_unpooled_data(self, src_features, tgt_features):
        src_features_list = []
        tgt_features_list = []

        src_pts_list = []
        tgt_pts_list = []

        for b_idx in range(len(src_features.decomposed_coordinates)):
            feat_s = src_features.F[src_features.C[:,0] == b_idx]
            feat_t = tgt_features.F[tgt_features.C[:,0] == b_idx]

            coor_s = src_features.C[src_features.C[:,0] == b_idx,1:].to(self.device) * self.cfg.voxel_size
            coor_t = tgt_features.C[tgt_features.C[:,0] == b_idx,1:].to(self.device) * self.cfg.voxel_size

            src_features_list.append(feat_s)
            tgt_features_list.append(feat_t)

            src_pts_list.append(coor_s)
            tgt_pts_list.append(coor_t)

        # src_features = torch.stack(src_features_list, dim=0)
        # tgt_features = torch.stack(tgt_features_list, dim=0)

        return src_features_list, tgt_features_list, src_pts_list, tgt_pts_list

    def forward(self, batch):
        main_tic = time.time()
        B = len(batch['src_xyz'])
        outputs = {}

        # Make the sparse tensors
        src_input = ME.SparseTensor(features=batch['feats_src'], coordinates=batch['coords_src'])
        tgt_input = ME.SparseTensor(features=batch['feats_tgt'], coordinates=batch['coords_tgt'])

        if self.verbose:
            print(type(batch['src_xyz']))
            print(batch['src_xyz'].shape)

            print(type(batch['tgt_xyz']))
            print(batch['tgt_xyz'].shape)

        
        ###########################
        # Sparse Minkowski Encoder
        ###########################
        tic = time.time()
        src_features, skip_features_src = self.encoder(src_input)
        tgt_features, skip_features_tgt = self.encoder(tgt_input)

        if self.time_verbose:
            print(f"Time for Minkowski Encoder: {time.time()-tic}")
        if self.verbose:
            print(f"src_features shape is: {src_features.shape}")
            print(f"tgt_features shape is: {tgt_features.shape}")
        
        ##################
        # Unpool the data 
        ##################
        tic = time.time()
        src_feats_un, tgt_feats_un, src_pts_list, tgt_pts_list = self._get_unpooled_data(src_features, tgt_features)

        if self.time_verbose:
            print(f"Time for Unpooling features: {time.time()-tic}")
        if self.verbose:
            print(f"unpooled src_feats_un shape is: {src_feats_un[0].shape}")
            print(f"unpooled tgt_feats_un shape is: {tgt_feats_un[0].shape}")    
            print(f"unpooled tgt_pts_list 0 shape is: {tgt_pts_list[0].shape}")    

        ##################
        # Pad the data 
        ##################
        src_feats_padded, src_key_padding_mask, _ = pad_sequence(src_feats_un,
                                                                 require_padding_mask=True)
        tgt_feats_padded, tgt_key_padding_mask, _ = pad_sequence(tgt_feats_un,
                                                                 require_padding_mask=True)

        ######################
        # Position Embeddings
        ######################
        # Pad the pts list
        src_pts_padded, _, _ = pad_sequence(src_pts_list)
        tgt_pts_padded, _, _ = pad_sequence(tgt_pts_list)

        if self.verbose:
            print(f"src_pts_padded type is: {type(src_pts_padded)}")
            print(f"src_pts_padded shape is: {src_pts_padded.shape}")
            print(f"tgt_pts_padded shape is: {tgt_pts_padded.shape}")

        # Get the position embeddings
        src_pe_padded = self.pos_embed(src_pts_padded)
        tgt_pe_padded = self.pos_embed(tgt_pts_padded)

        if self.verbose:
            print(f"src_pe shape is: {src_pe_padded.shape}")
            print(f"tgt_pe shape is: {tgt_pe_padded.shape}")

        ##########################
        # Cross-Attention Encoder
        ##########################
        tic = time.time()
        
        src_feats_cond, tgt_feats_cond = self.transformer_encoder(
            src_feats_padded, tgt_feats_padded,
            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
            src_pos=src_pe_padded, tgt_pos=tgt_pe_padded
            )
        
        if self.time_verbose:
            print(f"Transformer encoder time: {time.time() - tic}")
        if self.verbose:
            print("type of src_feats_cond", type(src_feats_cond))
            print("src_feats_cond dimensions are", src_feats_cond.shape)
            print("tgt_feats_cond dimensions are", tgt_feats_cond.shape)

        ##########################
        # Unpad the sequences 
        ##########################
        src_slens_c = []
        tgt_slens_c = []
        for i in range(B):
            src_slens_c.append(src_pts_list[i].shape[0])
            tgt_slens_c.append(tgt_pts_list[i].shape[0])
        
        src_feats_cond_unpad = unpad_sequences(src_feats_cond, src_slens_c)
        tgt_feats_cond_unpad = unpad_sequences(tgt_feats_cond, tgt_slens_c)

        if self.verbose:
            print("src_feats_cond_unpad type is: ", type(src_feats_cond_unpad))
            print("src_feats_cond_unpad len is: ", len(src_feats_cond_unpad))
            print("src_feats_cond_unpad 0 dimensions are", src_feats_cond_unpad[0].shape)
            print("src_feats_cond_unpad 1 dimensions are", src_feats_cond_unpad[1].shape)
            print("src_feats_cond_unpad 2 dimensions are", src_feats_cond_unpad[2].shape)
            print("src_feats_cond_unpad 3 dimensions are", src_feats_cond_unpad[3].shape)
            print("tgt_feats_cond_unpad 0 dimensions are", tgt_feats_cond_unpad[0].shape)
            print("tgt_feats_cond_unpad 1 dimensions are", tgt_feats_cond_unpad[1].shape)
            print("tgt_feats_cond_unpad 2 dimensions are", tgt_feats_cond_unpad[2].shape)
            print("tgt_feats_cond_unpad 3 dimensions are", tgt_feats_cond_unpad[3].shape)

        ######################
        # Softmax Correlation
        ######################
        tic = time.time()
        pose_sfc, attn_list, overlap_prob_list, ind_list = self.softmax_correlation(src_feats_cond_unpad, tgt_feats_cond_unpad,
                                     src_pts_list, tgt_pts_list)
        
        if self.time_verbose:
            print(f"Softmax corr time: {time.time() - tic}")

        if self.verbose:
            print(f"type of pose_sfc is {type(pose_sfc)}")
            print(f"demensions of pose_sfc is {pose_sfc.shape}")
            print(f"type of attn_list is {type(attn_list)}")
            print(f"type of attn_list is {attn_list[0].shape}")

        if self.time_verbose:
            print(f"Total time: {time.time() - main_tic}")

        ######################
        # Output
        ######################
        outputs = {
            # Predictions
            'pose': pose_sfc,
            'attn': attn_list,
            'src_feat': src_feats_cond_unpad,  # List(B) of (N_pred, N_src, D)
            'tgt_feat': tgt_feats_cond_unpad,  # List(B) of (N_pred, N_tgt, D)
            'src_kp': src_pts_list,
            'tgt_kp': tgt_pts_list,
            'src_feat_un': src_feats_un,
            'tgt_feat_un': tgt_feats_un,
            
            'overlap_prob_list': overlap_prob_list,
            'ind_list': ind_list,
        }
        return outputs

    def softmax_correlation(self, src_feats, tgt_feats, src_xyz, tgt_xyz):
        """
        Args:
            src_feats_padded: Source features List of [1, N, D] tensors
            tgt_feats_padded: Target features List of [1, M, D] tensors
            src_xyz: List of ([N, 3])
            tgt_xyz: List of ([M, 3])

        Returns:

        """
        B = len(src_feats)

        if self.verbose:
            print("type of src_feats", type(src_feats))
            print([f"src_feats {i} dimensions are: {src_feats[i].shape}" for i in range(B)])
            print([f"tgt_feats {i} dimensions are: {tgt_feats[i].shape}" for i in range(B)])
            print([f"src_xyz {i} dimensions are: {src_xyz[i].shape}" for i in range(B)])
            print([f"tgt_xyz {i} dimensions are: {tgt_xyz[i].shape}" for i in range(B)])

        pose_list = []
        attn_list = []
        overlap_prob_list = []
        ind_list = []
        for i in range(B):
            _, N, D = src_feats[i].shape
            _, M, D = tgt_feats[i].shape

            # Correlation = [1, N, M]
            correlation = torch.matmul(src_feats[i], tgt_feats[i].permute(0, 2, 1)) / (D**0.5)
            attn = torch.nn.functional.softmax(correlation, dim=-1)
            attn_list.append(attn)
            if N>M:
                attn = torch.nn.functional.softmax(correlation, dim=-2)
                attn_list.append(attn)

                val, ind = torch.max(attn, dim=1)
                src_pts = torch.gather(src_xyz[i], 0, ind.permute(1,0).expand(-1,3))  # [N, 3] -> [M, 3]
                T = compute_rigid_transform(src_pts, tgt_xyz[i], weights=val.permute(1,0).squeeze())
                
                overlap_prob_list.append(val.squeeze())
                ind_list.append(ind.squeeze())
            else:
                attn = torch.nn.functional.softmax(correlation, dim=-1)
                attn_list.append(attn)

                val, ind = torch.max(attn, dim=2)
                tgt_pts = torch.gather(tgt_xyz[i], 0, ind.permute(1,0).expand(-1,3))  # [M, 3] -> [N, 3]
                T = compute_rigid_transform(src_xyz[i], tgt_pts, weights=val.permute(1,0).squeeze())
                
                overlap_prob_list.append(val.squeeze())
                ind_list.append(ind.squeeze())

            pose_list.append(T)

        pose_sfc = torch.stack(pose_list, dim=0)

        return pose_sfc, attn_list, overlap_prob_list, ind_list

    def compute_overlap(self, src_xyz, tgt_xyz, pose, ind, dim):
        """
        Args:
            src_xyz: [N, 3]
            tgt_xyz: [M, 3]
            pose: [4, 4]
        Returns:
            overlap: [N, 1]
        """
        if self.verbose:
            print(f"src_xyz shape is {src_xyz.shape}")
            print(f"tgt_xyz shape is {tgt_xyz.shape}")
            print(f"pose shape is {pose.shape}")
            print(f"ind shape is {ind.shape}")
            print(f"dim is {dim}")
        
        src_xyz_tf = se3_transform(pose, src_xyz)
        dist = torch.cdist(src_xyz_tf, tgt_xyz)
        
        if self.verbose:
            print(f"src_xyz_tf shape is {src_xyz_tf.shape}")
            print(f"dist shape is {dist.shape}")

        min_dist_val, _ = torch.min(dist, dim=dim)

        if self.verbose:
            print(f"min_dist_val shape is {min_dist_val.shape}")

        overlap = torch.where(min_dist_val < self.cfg.overlap_threshold,
                             torch.ones_like(min_dist_val), torch.zeros_like(min_dist_val))
        if self.verbose:
            print(f"overlap shape is {overlap.shape}")
        
        overlap = torch.gather(overlap, 0, ind)
        if self.verbose:
            print(f"overlap shape is {overlap.shape}")
        return overlap

    def compute_loss(self, pred, batch):

        losses = {}
        pose_gt = batch['pose']

        src = batch['src_xyz']
        tgt = batch['tgt_xyz']

        # # Compute overlap loss
        overlap_loss = 0
        # for i in range(len(pred['overlap_prob_list'])):
        #     overlap_pred = pred['overlap_prob_list'][i]
            
        #     if pred['src_kp'][i].shape[0] > pred['tgt_kp'][i].shape[0]:
        #         overlap_gt = self.compute_overlap(pred['src_kp'][i], pred['tgt_kp'][i], batch['pose'][i],
        #                                         pred['ind_list'][i], dim=1)
        #     else:
        #         overlap_gt = self.compute_overlap(pred['src_kp'][i], pred['tgt_kp'][i], batch['pose'][i],
        #                                         pred['ind_list'][i], dim=0)

        #     overlap_loss += self.overlap_criterion(overlap_pred, overlap_gt)

        # Feature criterion
        for i in self.cfg.feature_loss_on:
            feature_loss = self.feature_criterion(
                [s[i] for s in pred['src_feat']],
                [t[i] for t in pred['tgt_feat']],
                se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
            )

        pc_tf_gt = se3_transform_list(pose_gt, src)
        pc_tf_pred = se3_transform_list(pred['pose'], src)

        T_loss = 0
        for i in range(len(pc_tf_gt)):
            T_loss += torch.mean(torch.abs(pc_tf_gt[i] - pc_tf_pred[i])).requires_grad_()

        if self.verbose:
            print(f"Feature loss: {0.1*feature_loss}")
            print(f"Overlap loss: {0.1*overlap_loss}")
            print(f"T loss: {T_loss}")

        # losses['feature'] = 0.1*feature_loss
        # losses['T'] = T_loss
        # losses['overlap'] = 0.1*overlap_loss
        losses['total'] = T_loss + 0.1 * feature_loss + 0.1 * overlap_loss

        return losses



