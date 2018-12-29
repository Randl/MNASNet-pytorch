# MNASNet in PyTorch

An implementation of `MNASNet` in PyTorch. `MNASNet` is an efficient
convolutional neural network architecture for mobile devices,
developed with architectural search. For more information check the paper:
[MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626)

The model is is implemented by
[billhhh](https://github.com/billhhh/MnasNet-pytorch-pretrained)
and the initial idea of reproducing MNASNet is by
[snakers4](https://github.com/snakers4/mnasnet-pytorch)

## Usage

Clone the repo:
```bash
git clone https://github.com/Randl/MNASNet-pytorch
pip install -r requirements.txt
```

Use the model defined in `model.py` to run ImageNet example:
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 imagenet.py --dataroot "/path/to/imagenet/" --warmup 5 --sched cosine -lr 0.2 -b 128 -d 5e-5 --world-size 8 --seed 42
```

To continue training from checkpoint
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "/path/to/checkpoint/folder"
```
## Results
Initially I've got 72+% top-1 accuracy, but the checkpointing didn't
work properly. I believe the results are reproducable.

|Classification Checkpoint| MACs (M)   | Parameters (M)| Top-1 Accuracy| Top-5 Accuracy|  Claimed top-1|  Claimed top-5|
|-------------------------|------------|---------------|---------------|---------------|---------------|---------------|

You can test it with
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "results/shufflenet_v2_0.5/model_best.pth.tar" -e
```

## Other implementations

- [Mnasnet.MXNet](https://github.com/chinakook/Mnasnet.MXNet) --  A Gluon implementation of Mnasnet, 73.6% top-1 and 91.52% top-5
- [MnasNet-pytorch-pretrained](https://github.com/billhhh/MnasNet-pytorch-pretrained) --  A PyTorch implementation of Mnasnet, 70.132% top-1 and 89.434% top-5