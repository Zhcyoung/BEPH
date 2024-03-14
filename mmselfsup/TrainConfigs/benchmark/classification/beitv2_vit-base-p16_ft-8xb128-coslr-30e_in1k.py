auto_scale_lr = dict(base_batch_size=1024)
bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=1000,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmcls', interval=1, max_keep_ckpts=2, type='CheckpointHook'),
    logger=dict(_scope_='mmcls', interval=100, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmcls', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmcls', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmcls', type='IterTimerHook'),
    visualization=dict(
        _scope_='mmcls', enable=False, type='VisualizationHook'))
default_scope = 'mmcls'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = "./TCGA_Checkpoints/beitv2_backbone_cancer.pth"
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='base',
        avg_token=True,
        drop_path_rate=0.1,
        img_size=224,
        init_cfg=dict(
            checkpoint='./TCGA_Checkpoints/beitv2_backbone_cancer.pth',
            prefix='backbone.',
            type='Pretrained'),
        output_cls_token=False,
        patch_size=16,
        type='BEiT',
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        use_shared_rel_pos_bias=False),
    head=dict(
        in_channels=768,
        init_cfg=[
            dict(layer='Linear', std=0.02, type='TruncNormal'),
        ],
        loss=dict(
            label_smooth_val=0.1, mode='original', type='SurvivalAnalysisLoss'),
        num_classes=2,
        type='SurvivalClsHead'),
    neck=None,
    type='ImageClassifier')
optim_wrapper = dict(
    constructor='mmselfsup.LearningRateDecayOptimWrapperConstructor',
    optimizer=dict(
        _scope_='mmcls',
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        layer_decay_rate=0.75,
        lr=5e-05,
        model_type='vit',
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            '.bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.gamma': dict(decay_mult=0.0),
            '.ln': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            'q_bias': dict(decay_mult=0.0),
            'v_bias': dict(decay_mult=0.0)
        })))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        start_factor=0.0001,
        type='LinearLR'),
    dict(
        begin=20,
        by_epoch=True,
        convert_to_iter_based=True,
        end=30,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=0)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=128,
    dataset=dict(
        _scope_='mmcls',
        ann_file='meta/val.txt',
        data_prefix='val',
        data_root='data/imagenet',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=5,
    sampler=dict(_scope_='mmcls', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmcls', topk=(
        1,
        2,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        edge='short',
        interpolation='bicubic',
        scale=256,
        type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackClsInputs'),
]
data_root = '/dssg/home/acct-medftn/medftn/BEPT/downstreamData/Cancer/'
train_cfg = dict(by_epoch=True, max_epochs=30, val_interval=1)
train_dataloader = dict(
    batch_size=128,
    dataset=dict(
        _scope_='mmcls',
        ann_file='meta/train.txt',
        data_prefix='train',
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=224,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                hparams=dict(
                    interpolation='bicubic', pad_val=[
                        104,
                        116,
                        124,
                    ]),
                magnitude_level=9,
                magnitude_std=0.5,
                num_policies=2,
                policies='timm_increasing',
                total_level=10,
                type='RandAugment'),
            dict(
                erase_prob=0.25,
                fill_color=[
                    103.53,
                    116.28,
                    123.675,
                ],
                fill_std=[
                    57.375,
                    57.12,
                    58.395,
                ],
                max_area_ratio=0.3333333333333333,
                min_area_ratio=0.02,
                mode='rand',
                type='RandomErasing'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=5,
    sampler=dict(_scope_='mmcls', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=224,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        hparams=dict(interpolation='bicubic', pad_val=[
            104,
            116,
            124,
        ]),
        magnitude_level=9,
        magnitude_std=0.5,
        num_policies=2,
        policies='timm_increasing',
        total_level=10,
        type='RandAugment'),
    dict(
        erase_prob=0.25,
        fill_color=[
            103.53,
            116.28,
            123.675,
        ],
        fill_std=[
            57.375,
            57.12,
            58.395,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasing'),
    dict(type='PackClsInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=128,
    dataset=dict(
        _scope_='mmcls',
        ann_file='meta/val.txt',
        data_prefix='./train',
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackClsInputs'),
        ],
        type='ImageNet'),
    num_workers=5,
    sampler=dict(_scope_='mmcls', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmcls', topk=(
        1,
        2,
    ), type='Accuracy')
vis_backends = [
    dict(_scope_='mmcls', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmcls',
    type='ClsVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/TrainConfigs/benchmark/classification/vit_survival'
