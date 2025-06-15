auto_scale_lr = dict(enable=False, base_batch_size=16)
backend_args = None
default_scope = 'mmdet'
work_dir = './work_dirs/mask2former_sam'
custom_imports = dict(imports=['mmdet.rsprompter'], allow_failed_imports=False)

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=3,
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends= vis_backends)
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)

log_level = 'INFO'
load_from = None
resume = False

num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

crop_size = (1024, 1024)

batch_augments = [
    dict(
        type='BatchFixedSizePad',
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False)
]

data_preprocessor = dict(
    batch_augments=batch_augments,
    bgr_to_rgb=True,
    mask_pad_value=0,
    mean=[123.675, 116.28, 103.53],
    pad_mask=True,
    pad_seg=False,
    pad_size_divisor=32,
    seg_pad_value=255,
    std=[58.395, 57.12, 57.375],
    type='DetDataPreprocessor')

hf_sam_pretrain_name = "checkpoints/sam_vit_base"
hf_sam_pretrain_ckpt_path = "checkpoints/sam_vit_base/pytorch_model.bin"

num_queries = 100

model = dict(
    type='SAMSegMask2Former',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='RSSamVisionEncoder',
        # peft_config = None,
        hf_pretrain_name=hf_sam_pretrain_name,
        extra_config=dict(output_hidden_states=True),
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)),
    neck=dict(
        type='RSFPN',
        feature_aggregator=dict(
            type='RSFeatureAggregator',
            in_channels=hf_sam_pretrain_name,
            out_channels=256,
            hidden_channels=32,
            select_layers=range(1, 13, 2),
        ),
        feature_spliter=dict(
            type='RSSimpleFPN',
            backbone_channel=256,
            in_channels=[64, 128, 256, 256],
            out_channels=256,
            num_outs=5,
            norm_cfg=dict(type='LN2d', requires_grad=True)
        ),
    ),
    panoptic_head=dict(
        type='Mask2FormerHead',
        in_channels=[256, 256, 256, 256, 256],  # pass to pixel_decoder inside
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=num_queries,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            strides=[4, 8, 16, 32, 64],
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=3,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    test_cfg=dict(
        filter_low_score=True,
        instance_on=True,
        iou_thr=0.8,
        max_per_image=100,
        panoptic_on=False,
        semantic_on=False),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', use_sigmoid=True, weight=5.0),
                dict(eps=1.0, pred_act=True, type='DiceCost', weight=5.0),
            ],
            type='HungarianAssigner'),
        importance_sample_ratio=0.75,
        num_points=12544,
        oversample_ratio=3.0,
        sampler=dict(type='MaskPseudoSampler')),
)

train_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(prob=0.5, type='RandomFlip'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        resize_type='Resize',
        scale=(
            1024,
            1024,
        ),
        type='RandomResize'),
    dict(
        allow_negative_crop=True,
        crop_size=(
            1024,
            1024,
        ),
        crop_type='absolute',
        recompute_bbox=True,
        type='RandomCrop'),
    dict(
        by_mask=True,
        min_gt_bbox_wh=(
            1e-05,
            1e-05,
        ),
        type='FilterAnnotations'),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    # dict(type='Pad', size=crop_size, pad_val=dict(img=(0.406 * 255, 0.456 * 255, 0.485 * 255), masks=0)),
    dict(type='Pad', size=crop_size),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]

# dataset settings
data_root = '../data/cross/coco'
dataset_type = 'CocoDataset'


num_workers = 4
persistent_workers = True
indices = None

train_dataloader = dict(
    batch_size=2,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        metainfo = {
            'classes': ('whistle', ),
            'palette': [
                (220, 20, 60),
            ]
        },
        type=dataset_type,
        data_root=data_root + '/train',
        ann_file='labels.json',
        data_prefix=dict(img='data/'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))


val_dataloader = dict(
    batch_size=4,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        metainfo = {
            'classes': ('whistle', ),
            'palette': [
                (220, 20, 60),
            ]
        },
        type=dataset_type,
        indices=indices,
        data_root=data_root + '/val',
        ann_file='labels.json',
        data_prefix=dict(img='data/', ),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)


test_dataloader = dict(
    batch_size=8,
    drop_last=False,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        metainfo = {
            'classes': ('whistle', ),
            'palette': [
                (220, 20, 60),
            ]
        },
        type=dataset_type,
        ann_file='labels.json',
        backend_args=None,
        data_prefix=dict(img='data/'),
        data_root=data_root + '/val',
        pipeline=test_pipeline,
        test_mode=True),
)


val_evaluator = dict(
    ann_file=data_root + '/val/labels.json',
    backend_args=None,
    format_only=False,
    metric=[
        'segm',
    ],
    type='WhistleMetric2')

test_evaluator = dict(
    ann_file= data_root + '/val/labels.json',
    # ann_file='../data/cross/coco/val/labels.json',
    backend_args=None,
    format_only=False,
    metric=[
        'segm',
    ],
    type='WhistleMetric2')

train_cfg = dict(
    dynamic_intervals=[
        (
            365001,
            368750,
        ),
    ],
    max_iters=368750,
    type='IterBasedTrainLoop',
    val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = dict(
    begin=0,
    by_epoch=False,
    end=368750,
    gamma=0.1,
    milestones=[
        327778,
        355092,
    ],
    type='MultiStepLR')

custom_keys=dict(
        backbone=dict(decay_mult=1.0, lr_mult=0.1),
        level_embed=dict(decay_mult=0.0, lr_mult=1.0),
        query_embed=dict(decay_mult=0.0, lr_mult=1.0),
        query_feat=dict(decay_mult=0.0, lr_mult=1.0))

for i in range(12):
    custom_keys[f'backbone.vision_encoder.layers.{i}.layer_norm1'] = dict(
        decay_mult=0.0, lr_mult=0.1)
    custom_keys[f'backbone.vision_encoder.layers.{i}.layer_norm2'] = dict(
        decay_mult=0.0, lr_mult=0.1)
custom_keys['backbone.vision_encoder.neck.layer_norm1'] = dict(
    decay_mult=0.0, lr_mult=1)
custom_keys['backbone.vision_encoder.neck.layer_norm2'] = dict(
    decay_mult=0.0, lr_mult=1)
    

optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=custom_keys,
        norm_decay_mult=0.0),
    type='OptimWrapper')

