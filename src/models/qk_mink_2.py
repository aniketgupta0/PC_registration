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
from utils.se3_torch import compute_rigid_transform, se3_transform_list, se3_inv
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences

from utils.ME_layers import get_norm_layer, get_res_block
# from lib.utils import kabsch_transformation_estimation

_EPS = 1e-6

"""
The reason for the bad performance might be the large number of conv layers which leads to extreme washout of parameters
"""

class SparseEnoder(ME.MinkowskiNetwork):
    CHANNELS = [None, 64, 128]

    def __init__(self,
                in_channels=3,
                out_channels=128,
                bn_momentum=0.1,
                conv1_kernel_size=9,
                norm_type='IN',
                D=3):

        ME.MinkowskiNetwork.__init__(self, D)

        NORM_TYPE = norm_type
        BLOCK_NORM_TYPE = norm_type
        CHANNELS = self.CHANNELS

    
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



    def forward(self, x, tgt_feature=False):
        # print("x", x.shape)
        skip_features = []
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out = self.block1(out_s1)

        # print("1: ", out.shape)
        skip_features.append(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out = self.block2(out_s2)
        # print("2: ", out.shape)

        return out, skip_features

class SparseDecoder(ME.MinkowskiNetwork):
    TR_CHANNELS = [None, 64, 128]
    CHANNELS = [None, 64, 128]

    def __init__(self,
                out_channels=128,
                bn_momentum=0.1,
                norm_type='IN',
                D=3):

        ME.MinkowskiNetwork.__init__(self, D)

        NORM_TYPE = norm_type
        BLOCK_NORM_TYPE = norm_type
        TR_CHANNELS = self.TR_CHANNELS
        CHANNELS = self.CHANNELS


        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2],
            out_channels=TR_CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)

        self.norm2_tr = get_norm_layer(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2_tr = get_res_block(
                BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)


        self.conv1_tr = ME.MinkowskiConvolutionTranspose(
                in_channels=CHANNELS[1] + TR_CHANNELS[2],
                out_channels=TR_CHANNELS[1],
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=False,
                dimension=D)
        # self.norm1_tr = get_norm_layer(NORM_TYPE, TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)


    def forward(self, x, skip_features):
        
        out = self.conv2_tr(x)
        out = self.norm2_tr(out)
        
        out_s2_tr = self.block2_tr(out)

        out = ME.cat(out_s2_tr, skip_features[-1])
        # print(f"out shape is: {out.shape}")
        out = self.conv1_tr(out)
        # out = self.norm1_tr(out)

        return out


class RegTR(GenericRegModel):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.verbose = False
        self.time_verbose = False
        self.normalize_feature = True

        ################################
        # Sparse Minkowski Encoder
        ################################
        self.encoder = SparseEnoder()
        self.decoder = SparseDecoder()

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
        # Normalize the features
        if self.normalize_feature:
            src_features = ME.SparseTensor(
                        src_features.F / torch.norm(src_features.F, p=2, dim=1, keepdim=True),
                        coordinate_map_key=src_features.coordinate_map_key,
                        coordinate_manager=src_features.coordinate_manager)

            tgt_features = ME.SparseTensor(
                        tgt_features.F / torch.norm(tgt_features.F, p=2, dim=1, keepdim=True),
                        coordinate_map_key=tgt_features.coordinate_map_key,
                        coordinate_manager=tgt_features.coordinate_manager)

        src_features_list = []
        tgt_features_list = []

        # src_pts_list = []
        # tgt_pts_list = []

        for b_idx in range(len(src_features.decomposed_coordinates)):
            feat_s = src_features.F[src_features.C[:,0] == b_idx]
            feat_t = tgt_features.F[tgt_features.C[:,0] == b_idx]

            # coor_s = src_features.C[src_features.C[:,0] == b_idx,1:].to(self.device) * self.voxel_size
            # coor_t = tgt_features.C[tgt_features.C[:,0] == b_idx,1:].to(self.device) * self.voxel_size

            src_features_list.append(feat_s)
            tgt_features_list.append(feat_t)

            # src_pts_list.append(coor_s)
            # tgt_pts_list.append(coor_t)

        src_features = torch.stack(src_features_list, dim=0)
        tgt_features = torch.stack(tgt_features_list, dim=0)

        return src_features, tgt_features

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
        enc_src_features, skip_features_src = self.encoder(src_input)
        enc_tgt_features, skip_features_tgt = self.encoder(tgt_input)

        if self.time_verbose:
            print(f"Time for Minkowski Encoder: {time.time()-tic}")
        if self.verbose:
            print(f"enc_src_features shape is: {enc_src_features.shape}")
            print(f"enc_tgt_features shape is: {enc_tgt_features.shape}")

        ###########################
        # Sparse Minkowski Decoder
        ###########################
        tic = time.time()
        src_features = self.decoder(enc_src_features, skip_features_src)
        tgt_features = self.decoder(enc_tgt_features, skip_features_tgt)
        
        if self.time_verbose:
            print(f"Time for Minkowski Decoder: {time.time()-tic}")
        if self.verbose:
            print(f"src_features shape is: {src_features.shape}")
            print(f"tgt_features shape is: {tgt_features.shape}")
        
        ##################
        # Unpool the data 
        ##################
        tic = time.time()
        src_feats_un, tgt_feats_un = self._get_unpooled_data(src_features, tgt_features)

        if self.time_verbose:
            print(f"Time for Unpooling features: {time.time()-tic}")
        if self.verbose:
            print(f"unpooled src_feats_un shape is: {src_feats_un.shape}")
            print(f"unpooled tgt_feats_un shape is: {tgt_feats_un.shape}")    

        ######################
        # Position Embeddings
        ######################
        src_pe = self.pos_embed(batch['src_xyz'])
        tgt_pe = self.pos_embed(batch['tgt_xyz'])

        if self.verbose:
            print(f"src_pe shape is: {src_pe.shape}")
            print(f"tgt_pe shape is: {tgt_pe.shape}")

        ##########################
        # Cross-Attention Encoder
        ##########################
        tic = time.time()
        
        src_feats_cond, tgt_feats_cond = self.transformer_encoder(src_feats_un, tgt_feats_un, 
                                                                    src_pos=src_pe, tgt_pos=tgt_pe)
        
        if self.time_verbose:
            print(f"Transformer encoder time: {time.time() - tic}")
        if self.verbose:
            print("type of src_feats_cond", type(src_feats_cond))
            print("src_feats_cond dimensions are", src_feats_cond.shape)
            print("tgt_feats_cond dimensions are", tgt_feats_cond.shape)


        ######################
        # Softmax Correlation
        ######################
        tic = time.time()
        pose_sfc, attn_list = self.softmax_correlation(src_feats_cond, tgt_feats_cond,
                                     batch['src_xyz'], batch['tgt_xyz'])
        
        if self.time_verbose:
            print(f"Softmax corr time: {time.time() - tic}")

        if self.verbose:
            print(f"type of pose_sfc is {type(pose_sfc)}")
            print(f"demensions of pose_sfc is {pose_sfc.shape}")
            print(f"type of attn_list is {type(attn_list)}")
            print(f"type of attn_list is {attn_list[0].shape}")

        if self.time_verbose:
            print(f"Total time: {time.time() - main_tic}")

        outputs = {
            # Predictions
            'pose': pose_sfc,
            'attn': attn_list,
            'src_feat': src_feats_cond,  # List(B) of (N_pred, N_src, D)
            'tgt_feat': tgt_feats_cond,  # List(B) of (N_pred, N_tgt, D)

            'src_feat_un': src_feats_un,
            'tgt_feat_un': tgt_feats_un,
        }
        return outputs

    def compute_loss(self, pred, batch):

        losses = {}
        pose_gt = batch['pose']

        src = batch['src_xyz']
        tgt = batch['tgt_xyz']

        # Feature criterion
        for i in self.cfg.feature_loss_on:
            feature_loss = self.feature_criterion(
                [s[i] for s in pred['src_feat']],
                [t[i] for t in pred['tgt_feat']],
                se3_transform_list(pose_gt, src), tgt,
            )

        pc_tf_gt = se3_transform_list(pose_gt, src)
        pc_tf_pred = se3_transform_list(pred['pose'], src)

        T_loss = 0
        for i in range(len(pc_tf_gt)):
            T_loss += torch.mean(torch.abs(pc_tf_gt[i] - pc_tf_pred[i])).requires_grad_()

        if self.verbose:
            # print(f"Feature loss: {feature_loss}")
            print(f"T loss: {T_loss}")

        # losses['feature'] = feature_loss
        # losses['T'] = T_loss
        losses['total'] = T_loss + 0.1 * feature_loss
        # losses['total'] = T_loss

        # print(losses)
        return losses

    def softmax_correlation(self, src_feats, tgt_feats, src_xyz, tgt_xyz):
        """
        Args:
            src_feats: Source features [B,N,D]
            tgt_feats: Target features [B,N,D]
            src_xyz: List of ([B,N,3])
            tgt_xyz: List of ([B,N,3])

        Returns:

        """
        src_feats = torch.squeeze(src_feats)
        tgt_feats = torch.squeeze(tgt_feats)

        if self.verbose:
            print("type of src_feats", type(src_feats))
            print("src_feats dimensions are", src_feats.shape)
            print("tgt_feats dimensions are", tgt_feats.shape)
            print("src_xyz dimensions are", src_xyz.shape)
            print("tgt_xyz dimensions are", tgt_xyz.shape)

        attn_list = []
        pose_list = []

        # print(f"src_feats max: {src_feats.max()}")
        # print(f"src_feats min: {src_feats.min()}")

        # print(f"tgt_feats max: {tgt_feats.max()}")
        # print(f"tgt_feats min: {tgt_feats.min()}")

        B, N, D = src_feats.shape
        correlation = torch.matmul(src_feats, tgt_feats.permute(0,2,1)) / (D**0.5)

        # print(f"correlation max: {correlation.max()}")
        # print(f"correlation min: {correlation.min()}")
        attn = torch.nn.functional.softmax(correlation, dim=-1)
        # print(f"attn max: {attn.max()}")
        # print(f"attn min: {attn.min()}")
        attn_list.append(attn)

        val, ind = torch.max(attn, dim=2)
        # print(val.shape)
        # print(ind.shape)
        # print(ind.unsqueeze(2).expand(-1,-1,3).shape)
        # print(tgt_xyz.shape)


        tgt_pts = torch.gather(tgt_xyz, 1, ind.unsqueeze(2).expand(-1,-1,3))
        for i in range(B):
            # print(src_xyz[i].shape)
            # print(tgt_pts[i].shape)

            # raise ValueError

            pose_list.append(compute_rigid_transform(src_xyz[i], tgt_pts[i], weights=val[i]))

        pose_sfc = torch.stack(pose_list, dim=0)

        return pose_sfc, attn_list
