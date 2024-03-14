_base_ = 'mmcls::_base_/default_runtime.py'

model = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    data_preprocessor=dict(
        num_classes=33,
        mean=(189.78929696867084, 128.24523960459152, 175.1775062798947),
        std=(38.39398676131772, 45.25197621706431, 32.21307843285958),
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
        num_classes=2,
        in_channels=768,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=2e-5)]),
)

dataset_type = 'mmcls.ImageNet'
data_root = './BEPT'
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
