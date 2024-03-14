_base_ = 'mmcls::_base_/default_runtime.py'

model = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    data_preprocessor=dict(
        num_classes=33,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    backbone=dict(
        type='BEiT',
        arch='base',
        img_size=224,
        patch_size=16,
        drop_path_rate=0.1,
        avg_token=True,
        output_cls_token=False,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=False),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=33,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
)


dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
extract_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmcls.ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='mmcls.PackClsInputs'),
]
extract_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='/dssg/home/acct-medftn/medftn/BEPT',
        ann_file='./meta/valid_shuffle.txt',
        data_prefix='./',
        pipeline=extract_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
