data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=(
189.78929696867084, 128.24523960459152, 175.1775062798947
    ),
    second_mean=(
  109.80125002590874, 107.85080245137108, 109.8301356923625
    ),
    second_std=(
 74.46909924205002, 74.91884943904161, 74.8734708147090
    ),
    std=(
38.39398676131772, 45.25197621706431, 32.21307843285958
    ),
    type='TwoNormDataPreprocessor')
data_root = '/dssg/home/acct-medftn/medftn/BEPT/'
dataset_type = 'mmcls.ImageNet'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmselfsup'
drop_path_rate = 0.0
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
launcher = 'none'
layer_scale_init_value = 0.1
load_from = None
log_level = 'INFO'
log_processor = dict(
    custom_cfg=[
        dict(data_src='', method='mean', window_size='global'),
    ],
    window_size=10)
model = dict(
    backbone=dict(
        arch='base',
        drop_path_rate=0.0,
        final_norm=False,
        init_cfg=[
            dict(layer='Linear', std=0.02, type='TruncNormal'),
            dict(layer='Conv2d', std=0.02, type='TruncNormal'),
            dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
        ],
        layer_scale_init_value=0.1,
        out_indices=[
            -4,
            -1,
        ],
        patch_size=16,
        type='BEiTViT'),
    head=dict(
        embed_dims=768,
        loss=dict(type='BEiTLoss'),
        num_embed=8192,
        type='BEiTV2Head'),
    neck=dict(
        backbone_arch='base',
        drop_path_rate=0.0,
        early_layers=9,
        layer_scale_init_value=0.1,
        num_layers=2,
        type='BEiTV2Neck'),
    target_generator=dict(
        encoder_config=dict(
            arch='base',
            avg_token=False,
            drop_path_rate=0.0,
            drop_rate=0.0,
            final_norm=True,
            frozen_stages=-1,
            img_size=224,
            in_channels=3,
            init_cfg=None,
            interpolate_mode='bicubic',
            layer_cfgs=dict(),
            layer_scale_init_value=0.0,
            norm_cfg=dict(eps=1e-06, type='LN'),
            out_indices=-1,
            output_cls_token=False,
            patch_cfg=dict(),
            patch_size=16,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            with_cls_token=True),
        init_cfg=dict(
            checkpoint=
            './points/beitv2_vit-backbone.pth',
            type='Pretrained'),
        type='VQKD'),
    type='BEiT')
optim_wrapper = dict(
    clip_grad=dict(max_norm=3.0),
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.98,
        ), lr=0.0015, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            '.bias': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.gamma': dict(decay_mult=0.0),
            '.ln': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            'q_bias': dict(decay_mult=0.0),
            'v_bias': dict(decay_mult=0.0)
        })),
    type='AmpOptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.98,
    ), lr=0.0015, type='AdamW', weight_decay=0.05)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=10,
        start_factor=0.0001,
        type='LinearLR'),
    dict(
        begin=10,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=1e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(diff_rank_seed=True, seed=0)
resume = False
train_cfg = dict(max_epochs=300, type='EpochBasedTrainLoop')
train_dataloader = dict(
    batch_size=256,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='meta/train_shuffle.txt',
        data_prefix=dict(img_path='Cancer_patches/'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                brightness=0.4,
                contrast=0.4,
                hue=0.0,
                saturation=0.4,
                type='ColorJitter'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                interpolation='bicubic',
                scale=(
                    0.2,
                    1.0,
                ),
                second_interpolation='bicubic',
                second_size=224,
                size=224,
                type='RandomResizedCropAndInterpolationWithTwoPic'),
            dict(
                input_size=(
                    14,
                    14,
                ),
                max_num_patches=75,
                min_num_patches=16,
                num_masking_patches=75,
                type='BEiTMaskGenerator'),
            dict(
                algorithm_keys=[
                    'mask',
                ],
                meta_keys=[
                    'img_path',
                ],
                type='PackSelfSupInputs'),
        ],
        type='mmcls.ImageNet'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        brightness=0.4,
        contrast=0.4,
        hue=0.0,
        saturation=0.4,
        type='ColorJitter'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        interpolation='bicubic',
        scale=(
            0.2,
            1.0,
        ),
        second_interpolation='bicubic',
        second_size=224,
        size=224,
        type='RandomResizedCropAndInterpolationWithTwoPic'),
    dict(
        input_size=(
            14,
            14,
        ),
        max_num_patches=75,
        min_num_patches=16,
        num_masking_patches=75,
        type='BEiTMaskGenerator'),
    dict(
        algorithm_keys=[
            'mask',
        ],
        meta_keys=[
            'img_path',
        ],
        type='PackSelfSupInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SelfSupVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
vqkd_encoder = dict(
    arch='base',
    avg_token=False,
    drop_path_rate=0.0,
    drop_rate=0.0,
    final_norm=True,
    frozen_stages=-1,
    img_size=224,
    in_channels=3,
    init_cfg=None,
    interpolate_mode='bicubic',
    layer_cfgs=dict(),
    layer_scale_init_value=0.0,
    norm_cfg=dict(eps=1e-06, type='LN'),
    out_indices=-1,
    output_cls_token=False,
    patch_cfg=dict(),
    patch_size=16,
    use_abs_pos_emb=True,
    use_rel_pos_bias=False,
    use_shared_rel_pos_bias=False,
    with_cls_token=True)
work_dir = './work_dirs/selfsup/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k'
