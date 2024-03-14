dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=2,
    mean=(189.78929696867084, 128.24523960459152, 175.1775062798947),
    std=(38.39398676131772, 45.25197621706431, 32.21307843285958),
    to_rgb=True)
bgr_mean = [175.1775062798947,128.24523960459152,189.78929696867084]
bgr_std = [32.21307843285958,45.25197621706431,38.39398676131772]
data_root = '/dssg/home/acct-medftn/medftn/BEPT/downstreamData/CAMELYON16'
# data_root ='/dssg/home/acct-medftn/medftn/BEPT/Cancer_patches'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color= [175.1775062798947,128.24523960459152,189.78929696867084],
        fill_std=[32.21307843285958,45.25197621706431,38.39398676131772]),
    dict(type='PackClsInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]
train_dataloader = dict(
    batch_size=224,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root= data_root,
        ann_file='/dssg/home/acct-medftn/medftn/BEPT/downstreamData/CAMELYON16/meta/train_shuffle.txt',
        data_prefix='./',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                scale=224,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='RandAugment',
                policies='timm_increasing',
                num_policies=2,
                total_level=10,
                magnitude_level=9,
                magnitude_std=0.5,
                hparams=dict(pad_val=[104, 116, 124],
                             interpolation='bicubic')),
            dict(
                type='RandomErasing',
                erase_prob=0.25,
                mode='rand',
                min_area_ratio=0.02,
                max_area_ratio=0.3333333333333333,
        fill_color= [175.1775062798947,128.24523960459152,189.78929696867084],
        fill_std=[32.21307843285958,45.25197621706431,38.39398676131772]),
            dict(type='PackClsInputs')
        ],
        _scope_='mmcls'),
    sampler=dict(type='DefaultSampler', shuffle=True, _scope_='mmcls'))
val_dataloader = dict(
    batch_size=224,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        ann_file='/dssg/home/acct-medftn/medftn/BEPT/downstreamData/CAMELYON16/meta/test_shuffle.txt',
        data_prefix='./',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeEdge',
                scale=256,
                edge='short',
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs')
        ],
        _scope_='mmcls'),
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmcls'))
val_evaluator = dict(type='Accuracy', topk=(1, 2), _scope_='mmcls')
test_dataloader = dict(
    batch_size=224,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root= data_root,
        ann_file='/dssg/home/acct-medftn/medftn/BEPT/downstreamData/CAMELYON16/meta/test_shuffle.txt',
        data_prefix='./',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeEdge',
                scale=256,
                edge='short',
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs')
        ],
        _scope_='mmcls'),
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmcls'))
test_evaluator = dict(type='Accuracy', topk=(1, 2), _scope_='mmcls')
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0015,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999),
        _scope_='mmcls',
        model_type='vit',
        layer_decay_rate=0.75),
    paramwise_cfg=dict(
        custom_keys=dict({
            '.ln': dict(decay_mult=0.0),
            '.bias': dict(decay_mult=0.0),
            'q_bias': dict(decay_mult=0.0),
            'v_bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            '.gamma': dict(decay_mult=0.0)
        })),
    constructor='mmselfsup.LearningRateDecayOptimWrapperConstructor')
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=80,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=10,
        end=80,
        eta_min=1e-06,
        convert_to_iter_based=True)
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=1024)
default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmcls'),
    logger=dict(type='LoggerHook', interval=100, _scope_='mmcls'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmcls'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, _scope_='mmcls', max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmcls'),
    visualization=dict(
        type='VisualizationHook', enable=False, _scope_='mmcls'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='WandbVisBackend', _scope_='mmcls')]
visualizer = dict(
    type='ClsVisualizer',
    vis_backends=[dict(type='WandbVisBackend')],
    _scope_='mmcls')
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=0, deterministic=False)
model = dict(
    type='ImageClassifier',
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
        init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.02)]))
launcher = 'none'
work_dir = './FineTuningWorkDirs/Beitv2_cancer_2000patch_2'
