import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

from utils.ME_layers import get_norm_layer, get_res_block
# from lib.utils import kabsch_transformation_estimation

_EPS = 1e-6

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
        print("x", x.shape)
        skip_features = []
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out = self.block1(out_s1)

        print("1: ", out.shape)
        skip_features.append(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out = self.block2(out_s2)
        print("2: ", out.shape)

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
        print(f"out shape is: {out.shape}")
        out = self.conv1_tr(out)
        # out = self.norm1_tr(out)

        return out

if __name__=="__main__":
    model = SparseEnoder()
    model2 = SparseDecoder()

    pc = np.fromfile("/work/nufr/aniket/dataset/sequences/00/velodyne/000000.bin", dtype=np.float32).reshape((-1,4))[:,:3]
    coords = torch.Tensor(np.floor(pc/0.1))#.unsqueeze(0)
    feats_train = [pc]
    feats = torch.Tensor(np.hstack(feats_train))#.unsqueeze(0)
    print(type(coords))
    print(coords.shape)

    coords_batch, feats_batch = ME.utils.sparse_collate(coords=[coords], feats=[feats])
    print(type(coords_batch))
    print(type(feats_batch))
    sinput = ME.SparseTensor(features=feats_batch.float(), coordinates=coords_batch)
    # sinput = ME.SparseTensor(features=feats, coordinates=coords)
    print(sinput.shape)
    out, skip = model(sinput)
    # out = model2(out, skip)
    print(out.shape)

