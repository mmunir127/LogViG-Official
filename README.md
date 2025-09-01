# Multi-Scale High-Resolution Logarithmic Grapher Module for Efficient Vision GNNs (LOG 2024)
Mustafa Munir, Alex Zhang, and Radu Marculescu

[PDF](https://openreview.net/pdf/fec03b23750738481c96ecb8a446da0590b1d727.pdf)

# Overview
This repository contains the source code for Multi-Scale High-Resolution Logarithmic Grapher Module for Efficient Vision GNNs

# Usage

## Installation Image Classification

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install mpi4py
pip install -r requirements.txt
```

### Train image classification:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --use_env main.py --data-path /path/to/imagenet --model Model --output_dir Results
```

## Installation Semantic Segmentation
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
```
pip install -U openmim
mim install mmengine
mim install mmcv-full
```
```
mim install "mmsegmentation <=0.30.0"
```

### Train semantic segmentation:

Semantic segmentation on ADE20K is implemented based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We follow settings and hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/segmentation), [PoolFormer](https://github.com/sail-sg/poolformer), and [EfficientFormer](https://github.com/snap-research/EfficientFormer) for comparison. 

```
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 train.py configs/sem_fpn/fpn_logvig_s_ade20k_40k.py --model logvig_s --work-dir semantic_results/ --launcher pytorch > semantic_results/logvig_s_semantic.txt
```


### Acknowledgement
We are very grateful for these excellent works [timm](https://github.com/huggingface/pytorch-image-models), [EfficientFormer](https://github.com/snap-research/EfficientFormer), [EfficientFormerV2](https://github.com/snap-research/EfficientFormer), [MobileViG](https://github.com/SLDGroup/MobileViG), [MobileViGV2](https://github.com/SLDGroup/MobileViGv2), [GreedyViG](https://github.com/SLDGroup/GreedyViG), and [MobileOne](https://github.com/apple/ml-mobileone), which have provided the basis for our framework.

### Citation

```
@inproceedings{logvig,
title={Multi-Scale High-Resolution Logarithmic Grapher Module for Efficient Vision GNNs},
author={Mustafa Munir and Alex Zhang and Radu Marculescu},
booktitle={The Third Learning on Graphs Conference},
year={2024}
}
```
