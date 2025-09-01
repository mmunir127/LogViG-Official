_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='logvig_b_feat',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../../../Results/B_LogViG/B_LogViG.pth',
        ),
    ),
    neck=dict(in_channels=[48, 96, 192, 384]),
    decode_head=dict(num_classes=150))

gpu_multiples = 2
# optimizer
optimizer = dict(type='AdamW', lr=0.0001 * gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000 // gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000 // gpu_multiples)
evaluation = dict(interval=8000 // gpu_multiples, metric='mIoU')
