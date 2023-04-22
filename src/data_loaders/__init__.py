import torch

import data_loaders.transforms
import data_loaders.modelnet as modelnet
from data_loaders.collate_functions import collate_pair, collate_tensors, collate_sparse_tensors
from data_loaders.threedmatch import ThreeDMatchDataset
from data_loaders.mkitti import KittiDataset

import torchvision


def get_dataloader(cfg, phase, num_workers=0, num_gpus=1):

    assert phase in ['train', 'val', 'test']

    if cfg.dataset == '3dmatch':
        if phase == 'train':
            # Apply training data augmentation (Pose perturbation and jittering)
            transforms_aug = torchvision.transforms.Compose([
                data_loaders.transforms.RigidPerturb(perturb_mode=cfg.perturb_pose),
                data_loaders.transforms.Jitter(scale=cfg.augment_noise),
                data_loaders.transforms.ShufflePoints(),
                data_loaders.transforms.RandomSwap(),
            ])
        else:
            transforms_aug = None

        dataset = ThreeDMatchDataset(
            cfg=cfg,
            phase=phase,
            transforms=transforms_aug,
        )

    elif cfg.dataset == 'modelnet':
        if phase == 'train':
            dataset = modelnet.get_train_datasets(cfg)[0]
        elif phase == 'val':
            dataset = modelnet.get_train_datasets(cfg)[1]
        elif phase == 'test':
            dataset = modelnet.get_test_datasets(cfg)

    elif cfg.dataset == "kitti":
        dataset = KittiDataset(config=cfg, phase=phase)

    else:
        raise AssertionError('Invalid dataset')

    # # For calibrating the number of neighbors (set in config file)
    # from models.backbone_kpconv.kpconv import calibrate_neighbors
    # neighborhood_limits = calibrate_neighbors(dataset, cfg)

    batch_size = cfg[f'{phase}_batch_size']
    # shuffle = phase == 'train'
    shuffle = False
    print(cfg.model)
    if cfg.model in ["qk_regtr.RegTR", "qk_regtr_old.RegTR"]:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if num_gpus == 1 else False,
            num_workers=num_workers,
            collate_fn=collate_pair,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset) if num_gpus > 1 else None,
        )
    elif cfg.model in ["qk_revvit.RegTR", "qk_revvit_2.RegTR", "qk_ce.RegTR"]:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if num_gpus == 1 else False,
            num_workers=num_workers,
            collate_fn=collate_tensors,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset) if num_gpus > 1 else None,
        )
    elif cfg.model in ["qk_mink.RegTR", "qk_mink_2.RegTR", "qk_mink_3.RegTR"]:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if num_gpus == 1 else False,
            num_workers=num_workers,
            collate_fn=collate_sparse_tensors,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset) if num_gpus > 1 else None,
        )
    
    return data_loader


