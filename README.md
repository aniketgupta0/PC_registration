# PC_registration
Run training with one gpu
```
python train.py --config conf/qk_regtr_modelnet.yaml
```
Run training with multiple gpus (e.g. 8)
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --config ./conf/qk_regtr_modelnet.yaml
```