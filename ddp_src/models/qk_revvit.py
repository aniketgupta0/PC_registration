
import math
import time
import torch
import torch.nn as nn
import numpy as np

from models.backbone_kpconv.kpconv import KPFEncoder, PreprocessorGPU, compute_overlaps
from models.generic_reg_model import GenericRegModel
from models.losses.corr_loss import CorrCriterion
from models.losses.feature_loss import InfoNCELossFull, CircleLossFull
from models.transformer.position_embedding import PositionEmbeddingCoordsSine, PositionEmbeddingLearned
from models.transformer.transformers import TransformerCrossEncoderLayer, TransformerCrossEncoder
from utils.se3_torch import compute_rigid_transform, se3_transform_list, se3_inv
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences

# Needed to implement custom backward pass
from torch.autograd import Function as Function

# We use the standard pytorch multi-head attention module
from torch.nn import MultiheadAttention as MHA


class RevBackProp(Function):

    """
    Custom Backpropagation function to allow (A) flusing memory in foward
    and (B) activation recomputation reversibly in backward for gradient
    calculation. Inspired by
    https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(ctx,x,layers):
        """
        Reversible Forward pass.
        Each reversible layer implements its own forward pass logic.
        """

        # obtaining X_1 and X_2 from the concatenated input
        X_1, X_2 = torch.chunk(x, 2, dim=-1)

        for layer in layers:
            X_1, X_2 = layer(X_1, X_2)
            all_tensors = [X_1.detach(), X_2.detach()]

        # saving only the final activations of the last reversible block
        # for backward pass, no intermediate activations are needed.
        ctx.save_for_backward(*all_tensors)
        ctx.layers = layers

        return torch.cat([X_1, X_2], dim=-1)

    @staticmethod
    def backward(ctx, dx):
        """
        Reversible Backward pass.
        Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        # obtaining gradients dX_1 and dX_2 from the concatenated input
        dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve the last saved activations, to start rev recomputation
        X_1, X_2 = ctx.saved_tensors
        # layer weights
        layers = ctx.layers

        for _, layer in enumerate(layers[::-1]):
            # this is recomputing both the activations and the gradients wrt
            # those activations.
            X_1, X_2, dX_1, dX_2 = layer.backward_pass(Y_1=X_1,Y_2=X_2,dY_1=dX_1,dY_2=dX_2)

        # final input gradient to be passed backward to the patchification layer
        dx = torch.cat([dX_1, dX_2], dim=-1)

        del dX_1, dX_2, X_1, X_2

        return dx, None, None


class ReversibleBlock(nn.Module):
    """
    Reversible Blocks for Reversible Vision Transformer.
    See Section 3.3.2 in paper for details.
    """

    def __init__(self,dim,num_heads):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()
        # F and G can be arbitrary functions, here we use
        # simple attwntion and MLP sub-blocks using vanilla attention.
        self.F = AttentionSubBlock(dim=dim, num_heads=num_heads)

        self.G = MLPSubblock(dim=dim)

        # note that since all functions are deterministic, and we are
        # not using any stochastic elements such as dropout, we do
        # not need to control seeds for the random number generator.
        # To see usage with controlled seeds and dropout, see pyslowfast.

    def forward(self, X_1, X_2):
        """
        forward pass equations:
        Y_1 = X_1 + Attention(X_2), F = Attention
        Y_2 = X_2 + MLP(Y_1), G = MLP
        """

        # Y_1 : attn_output
        f_X_2 = self.F(X_2)

        # Y_1 = X_1 + f(X_2)
        Y_1 = X_1 + f_X_2

        # free memory since X_1 is now not needed
        del X_1

        g_Y_1 = self.G(Y_1)

        # Y_2 = X_2 + g(Y_1)
        Y_2 = X_2 + g_Y_1

        # free memory since X_2 is now not needed
        del X_2

        return Y_1, Y_2

    def backward_pass(self, Y_1, Y_2, dY_1, dY_2):
        """
        equation for activation recomputation:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention
        And we use pytorch native logic carefully to
        calculate gradients on F and G.
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():

            Y_1.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            g_Y_1 = self.G(Y_1)

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass. Hence we do not
        # need to record it in the computation graph.
        with torch.no_grad():

            # recomputing X_2 from the rev equation
            X_2 = Y_2 - g_Y_1

            # free memory since g_Y_1 is now not needed
            del g_Y_1

            # the gradients for the previous block
            # note that it is called dY_1 but it in fact dX_1 in math.
            # reusing same variable to save memory
            dY_1 = dY_1 + Y_1.grad

            # free memory since Y_1.grad is now not needed
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            # reconstrucating the intermediate activations
            # and the computational graph for F.
            f_X_2 = self.F(X_2)

            # using pytorch native logic to differentiate through
            # gradients in G in backward pass.
            f_X_2.backward(dY_1, retain_graph=True)

        # propagate reverse computed acitvations at the start of
        # the previou block for backprop.s
        with torch.no_grad():

            # recomputing X_1 from the rev equation
            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            # the gradients for the previous block
            # note that it is called dY_2 but it in fact dX_2 in math.
            # reusing same variable to save memory
            dY_2 = dY_2 + X_2.grad

            # free memory since X_2.grad is now not needed
            X_2.grad = None

            X_2 = X_2.detach()

        # et voila~
        return X_1, X_2, dY_1, dY_2


class MLPSubblock(nn.Module):
    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(self, dim, mlp_ratio=4):

        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        return self.mlp(self.norm(x))


class AttentionSubBlock(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(self, dim, num_heads):

        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)

        # using vanilla attention for simplicity. To support adanced attention
        # module see pyslowfast.
        # Note that the complexity of the attention module is not a concern
        # since it is used blackbox as F block in the reversible logic and
        # can be arbitrary.
        self.attn = MHA(dim, num_heads, batch_first=True)

    def forward(self, x):
        x = self.norm(x)
        out, _ = self.attn(x, x, x)
        return out


class RegTR(GenericRegModel):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.verbose = False
        self.time_verbose = False


        #####################################
        # Linear layers to populate features
        #####################################
        # self.linear_layers = nn.Sequential(
        #     nn.Linear(3, 8),
        #     # nn.BatchNorm1d(8, eps=1e-5, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Linear(8, 32),
        #     # nn.BatchNorm1d(32, eps=1e-5, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Linear(32, 128),
        #     # nn.BatchNorm1d(128, eps=1e-5, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     # nn.BatchNorm1d(256, eps=1e-5, momentum=0.01),
        # )

        self.linear_layers = nn.Sequential(
            nn.Conv1d(3, 8, 1),
            nn.BatchNorm1d(8, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv1d(8, 32, 1),
            nn.BatchNorm1d(32, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv1d(32, 128, 1),
            nn.BatchNorm1d(128, eps=1e-5, momentum=0.01),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256, eps=1e-5, momentum=0.01),
        )
        ################################
        # Reversible Transformer Encoder
        ################################
        self.rev_T_layers = nn.ModuleList([
                ReversibleBlock(dim=cfg.rev_dim, num_heads=cfg.rev_head)
                for _ in range(cfg.rev_num)
            ])

        #######################
        # Embeddings # Not using them because we want the model to be permutation invariant? 
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
            batch_first=True
        )
        print("Using batch first"*10)
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

    def forward(self, batch):
        main_tic = time.time()
        B = len(batch['src_xyz'])
        # print("Input shape: ", batch['src_xyz'].shape)
        # print(batch['tgt_xyz'].shape)
        outputs = {}

        if self.verbose:
            print(type(batch['src_xyz']))
            print(batch['src_xyz'].shape)

            print(type(batch['tgt_xyz']))
            print(batch['tgt_xyz'].shape)

            print(f"batch['src_xyz'] max: {batch['src_xyz'].max()}")
            print(f"batch['src_xyz'] min: {batch['src_xyz'].min()}")
            print(f"batch['tgt_xyz'] max: {batch['tgt_xyz'].max()}")
            print(f"batch['tgt_xyz'] min: {batch['tgt_xyz'].min()}")
        
        ################
        # Linear layers
        ################
        tic = time.time()
        src_features = self.linear_layers(batch['src_xyz'])
        tgt_features = self.linear_layers(batch['tgt_xyz'])

        # Permute the dimensions (short term hack to get results)
        src_features = torch.permute(src_features, (0,2,1))
        tgt_features = torch.permute(tgt_features, (0,2,1))
        
        if self.time_verbose:
            print(f"Time for linear layers: {time.time()-tic}")
        if self.verbose:
            print(f"src_features shape is: {src_features.shape}")
            print(f"tgt_features shape is: {tgt_features.shape}")

            print(f"src_features max: {src_features.max()}")
            print(f"src_features min: {src_features.min()}")
            print(f"tgt_features max: {tgt_features.max()}")
            print(f"tgt_features min: {tgt_features.min()}")

        ######################
        # Position Embeddings
        ######################
        # src_pe, tgt_pe = self.pos_embed(kpconv_meta['points'][-1])
        # src_pe_padded, _, _ = pad_sequence(src_pe)
        # tgt_pe_padded, _, _ = pad_sequence(tgt_pe)
        src_pe = self.pos_embed(torch.permute(batch['src_xyz'], (0,2,1)))
        tgt_pe = self.pos_embed(torch.permute(batch['tgt_xyz'], (0,2,1)))

        if self.verbose:
            print(f"src_pe shape is: {src_pe.shape}")
            print(f"tgt_pe shape is: {tgt_pe.shape}")
        
        # raise ValueError

        


        #################################
        # Reversible Transformer Encoder
        #################################
        tic = time.time()
        src_feats_un = RevBackProp.apply(src_features, self.rev_T_layers)
        tgt_feats_un = RevBackProp.apply(tgt_features, self.rev_T_layers)

        if self.time_verbose:
            print(f"Time for Reversible transformer layers: {time.time()-tic}")
        if self.verbose:
            print(f"src_feats_un shape is: {src_feats_un.shape}")
            print(f"tgt_feats_un shape is: {tgt_feats_un.shape}")

            print(f"src_feats_un max: {src_feats_un.max()}")
            print(f"src_feats_un min: {src_feats_un.min()}")
            print(f"tgt_feats_un max: {tgt_feats_un.max()}")
            print(f"tgt_feats_un min: {tgt_feats_un.min()}")

        ##########################
        # Cross-Attention Encoder
        ##########################
        # Performs padding, then apply attention (REGTR "encoder" stage) to condition on the other
        # point cloud

        tic = time.time()
        
        src_feats_cond, tgt_feats_cond = self.transformer_encoder(src_feats_un, tgt_feats_un, 
                                                                    src_pos=src_pe, tgt_pos=tgt_pe)
        
        if self.time_verbose:
            print(f"Transformer encoder time: {time.time() - tic}")
        if self.verbose:
            print("type of src_feats_cond", type(src_feats_cond))
            print("src_feats_cond dimensions are", src_feats_cond.shape)
            print("tgt_feats_cond dimensions are", tgt_feats_cond.shape)
            
            print(f"src_feats_cond max: {src_feats_cond.max()}")
            print(f"src_feats_cond min: {src_feats_cond.min()}")
            print(f"tgt_feats_cond max: {tgt_feats_cond.max()}")
            print(f"tgt_feats_cond min: {tgt_feats_cond.min()}")


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

        

        src = torch.permute(batch['src_xyz'], (0,2,1))
        tgt = torch.permute(batch['tgt_xyz'], (0,2,1))


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
        src_feats = src_feats.squeeze(0)
        tgt_feats = tgt_feats.squeeze(0)

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
        tgt_xyz = torch.permute(tgt_xyz, (0,2,1))

        # print(val.shape)
        # print(ind.shape)
        # print(ind.unsqueeze(2).expand(-1,-1,3).shape)
        # print(tgt_xyz.shape)


        tgt_pts = torch.gather(tgt_xyz, 1, ind.unsqueeze(2).expand(-1,-1,3))
        for i in range(B):
            # print(src_xyz[i].shape)
            # print(tgt_pts[i].shape)

            # raise ValueError

            pose_list.append(compute_rigid_transform(src_xyz[i].T, tgt_pts[i], weights=val[i]))

        pose_sfc = torch.stack(pose_list, dim=0)

        return pose_sfc, attn_list


def main():
    # instantiating and fixing the model.
    model = RevViT()

    # random input, instaintiate and fixing.
    # no need for GPU for unit test, runs fine on CPU.
    pcd1 = torch.rand((1000, 3))
    pcd2 = torch.rand((1200, 3))
    pcd = torch.nn.utils.rnn.pad_sequence([pcd1, pcd2], batch_first=True)
    # pcd = torch.rand((1,1000,3))


    x = torch.rand((1, 3, 32, 32))
    
    print(model)
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total model params: {total_params}")
    # output of the model under reversible backward logic
    output = model(pcd)
    print(output.shape)
    # loss is just the norm of the output
    loss = output.norm()
    print(loss)

    # computatin gradients with reversible backward logic
    # using retain_graph=True to keep the computation graph.
    loss.backward()


if __name__ == "__main__":
    main()