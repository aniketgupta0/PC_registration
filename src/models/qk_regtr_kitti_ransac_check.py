"""REGTR network architecture
"""
import math
import time
import torch
import torch.nn as nn

from models.backbone_kpconv.kpconv import KPFEncoder, PreprocessorGPU, compute_overlaps
from models.generic_reg_model import GenericRegModel
from models.losses.corr_loss import CorrCriterion
from models.losses.feature_loss import InfoNCELossFull, CircleLossFull
from models.transformer.position_embedding import PositionEmbeddingCoordsSine, \
    PositionEmbeddingLearned
from models.transformer.transformers import \
    TransformerCrossEncoderLayer, TransformerCrossEncoder
from utils.se3_torch import compute_rigid_transform, se3_transform_list, se3_inv
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences
# from utils.viz import visualize_registration
_TIMEIT = False


class RegTR(GenericRegModel):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.verbose = False
        self.time_verbose = False
        #######################
        # Preprocessor
        #######################
        self.preprocessor = PreprocessorGPU(cfg)

        #######################
        # KPConv Encoder/decoder
        #######################
        self.kpf_encoder = KPFEncoder(cfg, cfg.d_embed)
        # Bottleneck layer to shrink KPConv features to a smaller dimension for running attention
        self.feat_proj = nn.Linear(self.kpf_encoder.encoder_skip_dims[-1], cfg.d_embed, bias=True)

        #######################
        # Embeddings
        #######################
        if cfg.get('pos_emb_type', 'sine') == 'sine':
            self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.d_embed,
                                                         scale=cfg.get('pos_emb_scaling', 1.0))
        elif cfg['pos_emb_type'] == 'learned':
            self.pos_embed = PositionEmbeddingLearned(3, cfg.d_embed)
        else:
            raise NotImplementedError

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

        ######################
        # Output layers
        ######################
        # if cfg.get('direct_regress_coor', False):
        #     self.correspondence_decoder = CorrespondenceRegressor(cfg.d_embed)
        # else:
        #     self.correspondence_decoder = CorrespondenceDecoder(cfg.d_embed,
        #                                                         cfg.corr_decoder_has_pos_emb,
        #                                                         self.pos_embed)

        #######################
        # Losses
        #######################
        self.overlap_criterion = nn.BCEWithLogitsLoss()
        if self.cfg.feature_loss_type == 'infonce':
            self.feature_criterion = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
        elif self.cfg.feature_loss_type == 'circle':
            self.feature_criterion = CircleLossFull(dist_type='euclidean', r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = self.feature_criterion
        else:
            raise NotImplementedError

        self.corr_criterion = CorrCriterion(metric='mae')

        self.weight_dict = {}
        for k in ['overlap', 'feature', 'corr']:
            for i in cfg.get(f'{k}_loss_on', [cfg.num_encoder_layers - 1]):
                self.weight_dict[f'{k}_{i}'] = cfg.get(f'wt_{k}')
        self.weight_dict['feature_un'] = cfg.wt_feature_un

        self.logger.info('Loss weighting: {}'.format(self.weight_dict))
        self.logger.info(
            f'Config: d_embed:{cfg.d_embed}, nheads:{cfg.nhead}, pre_norm:{cfg.pre_norm}, '
            f'use_pos_emb:{cfg.transformer_encoder_has_pos_emb}, '
            f'sa_val_has_pos_emb:{cfg.sa_val_has_pos_emb}, '
            f'ca_val_has_pos_emb:{cfg.ca_val_has_pos_emb}'
        )

    def forward(self, batch):
        main_tic = time.time()
        B = len(batch['src_xyz'])
        outputs = {}

        if self.verbose:
            print(type(batch['src_xyz']))
            print(batch['src_xyz'][0].shape)
            print(batch['src_xyz'][1].shape)

            print(type(batch['tgt_xyz']))
            print(batch['tgt_xyz'][0].shape)
            print(batch['tgt_xyz'][1].shape)

        tic = time.time()
        # Preprocess
        kpconv_meta = self.preprocessor(batch['src_xyz'] + batch['tgt_xyz'])
        batch['kpconv_meta'] = kpconv_meta
        slens = [s.tolist() for s in kpconv_meta['stack_lengths']]
        slens_c = slens[-1]
        src_slens_c, tgt_slens_c = slens_c[:B], slens_c[B:]
        feats0 = torch.ones_like(kpconv_meta['points'][0][:, 0:1])

        if self.time_verbose:
            print(f"\n Kpconv Preprocess time: {time.time()-tic}")

        ####################
        # REGTR Encoder
        ####################
        # KPConv encoder (downsampling) to obtain unconditioned features
        tic = time.time()
        feats_un, skip_x = self.kpf_encoder(feats0, kpconv_meta)
        
        if self.time_verbose:
            print(f"KPConv Encoder time: {time.time()-tic}")
        if self.verbose:
            print(f"feats_un: {type(feats_un)}")
            print(f"feats_un dimensions are: {feats_un.shape}")

        tic = time.time()
        both_feats_un = self.feat_proj(feats_un)
        
        if self.time_verbose:
            print(f"Feat projection time: {time.time()-tic}")
        if self.verbose:
            print(f"both_feats_un: {type(both_feats_un)}")
            print(f"both_feats_un dimensions are: {both_feats_un.shape}")

        tic = time.time()
        src_feats_un, tgt_feats_un = split_src_tgt(both_feats_un, slens_c)
        
        if self.time_verbose:
            print(f"Split time: {time.time()-tic}")

        if self.verbose:
            print(f"src_feats_un: {type(src_feats_un)}")
            print(f"src_feats_un dimensions are: {src_feats_un[0].shape}")
            print(f"src_feats_un dimensions are: {src_feats_un[1].shape}")

            print(f"tgt_feats_un: {type(tgt_feats_un)}")
            print(f"tgt_feats_un dimensions are: {tgt_feats_un[0].shape}")
            print(f"tgt_feats_un dimensions are: {tgt_feats_un[1].shape}")

        tic = time.time()
        # Position embedding for downsampled points
        src_xyz_c, tgt_xyz_c = split_src_tgt(kpconv_meta['points'][-1], slens_c)
        src_pe, tgt_pe = split_src_tgt(self.pos_embed(kpconv_meta['points'][-1]), slens_c)
        src_pe_padded, _, _ = pad_sequence(src_pe)
        tgt_pe_padded, _, _ = pad_sequence(tgt_pe)
        
        if self.time_verbose:
            print(f"Position embedding time: {time.time()-tic}")

        if self.verbose:
            print(f"src_xyz_c: {type(src_xyz_c)}")
            print(f"src_xyz_c dimensions are: {src_xyz_c[0].shape}")
            print(f"src_xyz_c dimensions are: {src_xyz_c[1].shape}")

            print(f"tgt_xyz_c: {type(tgt_xyz_c)}")
            print(f"tgt_xyz_c dimensions are: {tgt_xyz_c[0].shape}")
            print(f"tgt_xyz_c dimensions are: {tgt_xyz_c[1].shape}")

        # Performs padding, then apply attention (REGTR "encoder" stage) to condition on the other
        # point cloud

        tic = time.time()
        src_feats_padded, src_key_padding_mask, _ = pad_sequence(src_feats_un,
                                                                 require_padding_mask=True)
        tgt_feats_padded, tgt_key_padding_mask, _ = pad_sequence(tgt_feats_un,
                                                                 require_padding_mask=True)
        src_feats_cond, tgt_feats_cond = self.transformer_encoder(
            src_feats_padded, tgt_feats_padded,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_pos=src_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
            tgt_pos=tgt_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
        )
        
        if self.time_verbose:
            print(f"Transformer encoder time: {time.time() - tic}")
        if self.verbose:
            print("type of src_feats_cond", type(src_feats_cond))
            print("src_feats_cond dimensions are", src_feats_cond.shape)
            print("tgt_feats_cond dimensions are", tgt_feats_cond.shape)
            print("len of src_xyz_c", len(src_xyz_c))
            print("src_xyz_c dimensions are", src_xyz_c[0].shape)
        
        tic = time.time()
        src_feats_cond_unpad = unpad_sequences(src_feats_cond, src_slens_c)
        tgt_feats_cond_unpad = unpad_sequences(tgt_feats_cond, tgt_slens_c)
        
        if self.time_verbose:
            print(f"Unpad time: {time.time()-tic}")

        if self.verbose:
            print("src_feats_cond_unpad type is: ", type(src_feats_cond_unpad))
            print("src_feats_cond_unpad len is: ", len(src_feats_cond_unpad))
            print("src_feats_cond_unpad dimensions are", src_feats_cond_unpad[0].shape)
            print("src_feats_cond_unpad dimensions are", src_feats_cond_unpad[1].shape)

        # Softmax Correlation
        tic = time.time()
        pose_sfc, attn_list, src_pts_list, tgt_pts_list = self.softmax_correlation(src_feats_cond_unpad, tgt_feats_cond_unpad,
                                     src_xyz_c, tgt_xyz_c)
        
        if self.time_verbose:
            print(f"Softmax corr time: {time.time() - tic}")

        if self.verbose:
            print(f"type of pose_sfc is {type(pose_sfc)}")
            print(f"demensions of pose_sfc is {pose_sfc.shape}")
            print(f"type of attn_list is {type(attn_list)}")
            print(f"type of attn_list is {attn_list[0].shape}")
            print(f"type of attn_list is {attn_list[1].shape}")


        
        if self.time_verbose:
            print(f"Total time: {time.time() - main_tic}")
        # raise ValueError

        outputs = {
            # Predictions
            'pose': pose_sfc,
            'attn': attn_list,
            'src_feat': src_feats_cond_unpad,  # List(B) of (N_pred, N_src, D)
            'tgt_feat': tgt_feats_cond_unpad,  # List(B) of (N_pred, N_tgt, D)

            'src_kp': src_xyz_c,
            'tgt_kp': tgt_xyz_c,

            'src_feat_un': src_feats_un,
            'tgt_feat_un': tgt_feats_un,

            'src_pts_after_sfc': src_pts_list,
            'tgt_pts_after_sfc': tgt_pts_list,
        }
        return outputs

    def compute_loss(self, pred, batch):

        losses = {}
        kpconv_meta = batch['kpconv_meta']
        pose_gt = batch['pose']
        p = len(kpconv_meta['stack_lengths']) - 1  # coarsest level

        # print(f"Datatype of pred['src_kp']: {pred['src_kp'][0].dtype}")
        # print(f"Datatype of pred['src_kp']: {pred['src_kp'][0].dtype}")
        # print(f"Datatype of pose_gt: {pose_gt.dtype}")
        # Feature criterion
        for i in self.cfg.feature_loss_on:
            feature_loss = self.feature_criterion(
                [s[i] for s in pred['src_feat']],
                [t[i] for t in pred['tgt_feat']],
                se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
            )
        # losses['feature_un'] = self.feature_criterion_un(
        #     pred['src_feat_un'],
        #     pred['tgt_feat_un'],
        #     se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
        # )

        pc_tf_gt = se3_transform_list(pose_gt, pred['src_kp'])
        pc_tf_pred = se3_transform_list(pred['pose'], pred['src_kp'])

        # print("Loss =====")
        # print(pose_gt.shape)
        # print(pred['pose'].shape)
        # print(pred['src_kp'][0].shape)
        # print(pc_tf_gt[0].shape)
        # print(pc_tf_pred[0].shape)
        T_loss = 0
        for i in range(len(pc_tf_gt)):
            T_loss += torch.mean(torch.abs(pc_tf_gt[i] - pc_tf_pred[i])).requires_grad_()
        # losses['total'] = torch.sum(
        #     torch.stack([(losses[k] * self.weight_dict[k]) for k in losses]))

        if self.verbose:
            print(f"Feature loss: {feature_loss}")
            print(f"T loss: {T_loss}")

        # losses['feature'] = feature_loss
        # losses['T'] = T_loss
        # losses['total'] = T_loss + 0.1 * feature_loss
        losses['total'] = T_loss
        return losses

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
        pose_list = []
        attn_list = []
        src_pts_list = []
        tgt_pts_list = []

        for i in range(B):
            _, N, D = src_feats[i].shape
            _, M, D = tgt_feats[i].shape 

            # Correlation = [1, N, M]
            correlation = torch.matmul(src_feats[i], tgt_feats[i].permute(0, 2, 1)) / (D**0.5)
            attn = torch.nn.functional.softmax(correlation, dim=-1)
            attn_list.append(attn)
            if N>M:
                # val, ind = torch.max(attn, dim=1)
                val2, ind2 = torch.topk(attn,2, dim=1)
                val2_ratio = val2[:,1,:]/val2[:,0,:]
                val2 = torch.where(val2_ratio<0.9, val2, 0)
                val = val2[:,0,:]
                ind = ind2[:,0,:]


                # print(ind.shape)
                # print(src_xyz[i].shape)
                # print(val.shape)
                # print(src_xyz[i][:-1].shape)
                src_pts = torch.gather(src_xyz[i], 0, ind.permute(1,0).expand(-1,3))  # [N, 3] -> [M, 3]
                # print("src_pts shape: ", src_pts.shape)
                # print("tgt_pts.shape", tgt_xyz[i].shape)
                T = compute_rigid_transform(src_pts, tgt_xyz[i], weights=val.permute(1,0).squeeze())

                src_pts_list.append(src_pts)
                tgt_pts_list.append(tgt_xyz[i])
            else:
                # val, ind = torch.max(attn, dim=2)

                val2, ind2 = torch.topk(attn,2, dim=2)
                val2_ratio = val2[:,:,1]/val2[:,:,0]
                val2 = torch.where(val2_ratio<0.9, val2[:,:,0], 0)
                val = val2
                ind = ind2[:,:,0]


                
                # print(ind.shape)
                # print(tgt_xyz[i].shape)
                # print(val.shape)
                # print(tgt_xyz[i][:-1].shape)

                tgt_pts = torch.gather(tgt_xyz[i], 0, ind.permute(1,0).expand(-1,3))  # [M, 3] -> [N, 3]
                # print("src_pts shape: ", src_xyz[i].shape)
                # print("tgt_pts.shape", tgt_pts.shape)
                T = compute_rigid_transform(src_xyz[i], tgt_pts, weights=val.permute(1,0).squeeze())

                src_pts_list.append(src_xyz[i])
                tgt_pts_list.append(tgt_pts)


            pose_list.append(T)

        pose_sfc = torch.stack(pose_list, dim=0)

        return pose_sfc, attn_list, src_pts_list, tgt_pts_list



class CorrespondenceDecoder(nn.Module):
    def __init__(self, d_embed, use_pos_emb, pos_embed=None, num_neighbors=0):
        super().__init__()

        assert use_pos_emb is False or pos_embed is not None, \
            'Position encoder must be supplied if use_pos_emb is True'

        self.use_pos_emb = use_pos_emb
        self.pos_embed = pos_embed
        self.q_norm = nn.LayerNorm(d_embed)

        self.q_proj = nn.Linear(d_embed, d_embed)
        self.k_proj = nn.Linear(d_embed, d_embed)
        self.conf_logits_decoder = nn.Linear(d_embed, 1)
        self.num_neighbors = num_neighbors

        # nn.init.xavier_uniform_(self.q_proj.weight)
        # nn.init.xavier_uniform_(self.k_proj.weight)

    def simple_attention(self, query, key, value, key_padding_mask=None):
        """Simplified single-head attention that does not project the value:
        Linearly projects only the query and key, compute softmax dot product
        attention, then returns the weighted sum of the values

        Args:
            query: ([N_pred,] Q, B, D)
            key: ([N_pred,] S, B, D)
            value: (S, B, E), i.e. dimensionality can be different
            key_padding_mask: (B, S)

        Returns:
            Weighted values (B, Q, E)
        """

        q = self.q_proj(query) / math.sqrt(query.shape[-1])
        k = self.k_proj(key)

        attn = torch.einsum('...qbd,...sbd->...bqs', q, k)  # (B, N_query, N_src)

        if key_padding_mask is not None:
            attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(key_padding_mask, float('-inf'))
            attn = attn + attn_mask[:, None, :]  # ([N_pred,] B, Q, S)

        if self.num_neighbors > 0:
            neighbor_mask = torch.full_like(attn, fill_value=float('-inf'))
            haha = torch.topk(attn, k=self.num_neighbors, dim=-1).indices
            neighbor_mask[:, :, haha] = 0
            attn = attn + neighbor_mask

        attn = torch.softmax(attn, dim=-1)

        attn_out = torch.einsum('...bqs,...sbd->...qbd', attn, value)

        return attn_out

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3)
            tgt_xyz: List of ([N_pred,] N_tgt, 3)

        Returns:

        """

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)
        assert src_xyz_padded.shape[:-1] == src_feats_padded.shape[-3:-1] and \
               tgt_xyz_padded.shape[:-1] == tgt_feats_padded.shape[-3:-1]

        if self.use_pos_emb:
            both_xyz_packed = torch.cat(src_xyz + tgt_xyz)
            slens = list(map(len, src_xyz)) + list(map(len, tgt_xyz))
            src_pe, tgt_pe = split_src_tgt(self.pos_embed(both_xyz_packed), slens)
            src_pe_padded, _, _ = pad_sequence(src_pe)
            tgt_pe_padded, _, _ = pad_sequence(tgt_pe)

        # Decode the coordinates
        src_feats2 = src_feats_padded + src_pe_padded if self.use_pos_emb else src_feats_padded
        tgt_feats2 = tgt_feats_padded + tgt_pe_padded if self.use_pos_emb else tgt_feats_padded
        src_corr = self.simple_attention(src_feats2, tgt_feats2, pad_sequence(tgt_xyz)[0],
                                         tgt_key_padding_mask)
        tgt_corr = self.simple_attention(tgt_feats2, src_feats2, pad_sequence(src_xyz)[0],
                                         src_key_padding_mask)

        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list


class CorrespondenceRegressor(nn.Module):

    def __init__(self, d_embed):
        super().__init__()

        self.coor_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 3)
        )
        self.conf_logits_decoder = nn.Linear(d_embed, 1)

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3). Ignored
            tgt_xyz: List of ([N_pred,] N_tgt, 3). Ignored

        Returns:

        """

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)

        # Decode the coordinates
        src_corr = self.coor_mlp(src_feats_padded)
        tgt_corr = self.coor_mlp(tgt_feats_padded)

        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list
