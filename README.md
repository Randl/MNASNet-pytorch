# MNASNet in PyTorch

An implementation of `MNASNet` in PyTorch. `MNASNet` is an efficient convolutional neural network architecture for mobile devices, developed with architectural search. For more information check the paper:
[MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626)

## Usage

Clone the repo:
```bash
git clone https://github.com/Randl/ShuffleNetV2-pytorch
pip install -r requirements.txt
```

Use the model defined in `model.py` to run ImageNet example:
```bash
python3 -m torch.distributed.launch --nproc_per_node=7 imagenet.py --dataroot /home/chaimb/ILSVRC/Data/CLS-LOC --seed 42 -j 2 -b 2048
```

To continue training from checkpoint
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "/path/to/checkpoint/folder"
```
## Results


For x0.5 model I achieved 0.4% lower top-1 accuracy than claimed.

|Classification Checkpoint| MACs (M)   | Parameters (M)| Top-1 Accuracy| Top-5 Accuracy|  Claimed top-1|  Claimed top-5|
|-------------------------|------------|---------------|---------------|---------------|---------------|---------------|

You can test it with
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "results/shufflenet_v2_0.5/model_best.pth.tar" -e --scaling 0.5
```
