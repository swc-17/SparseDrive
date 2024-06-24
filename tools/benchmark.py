# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
import sys
sys.path.append('.')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets import custom_build_dataset
from mmdet.models import build_detector
from mmcv.cnn.utils.flops_counter import add_flops_counting_methods
from mmcv.parallel import scatter


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--samples', default=1000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def get_max_memory(model):
    device = getattr(model, 'output_device', None)
    mem = torch.cuda.max_memory_allocated(device=device)
    mem_mb = torch.tensor([mem / (1024 * 1024)],
        dtype=torch.int,
        device=device)
    return mem_mb.item()


def main():
    args = parse_args()
    get_flops_params(args)
    get_mem_fps(args)

def get_mem_fps(args):
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    print(cfg.data.test)
    dataset = custom_build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if args.fuse_conv_bn:
    #     model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    max_memory = 0
    for i, data in enumerate(data_loader):
        # torch.cuda.synchronize()
        with torch.no_grad():
            start_time = time.perf_counter()
            model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            max_memory = max(max_memory, get_max_memory(model))

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} img / s, '
                      f"gpu mem: {max_memory} M")

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} img / s')
            break


def get_flops_params(args):
    gpu_id = 0
    cfg = Config.fromfile(args.config)
    dataset = custom_build_dataset(cfg.data.val)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    data_iter = dataloader.__iter__()
    data = next(data_iter)
    data = scatter(data, [gpu_id])[0]

    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda(gpu_id)
    model.eval()

    bilinear_flops = 11
    num_key_pts_det = (
        cfg.model["head"]['det_head']["deformable_model"]["kps_generator"]["num_learnable_pts"]
        + len(cfg.model["head"]['det_head']["deformable_model"]["kps_generator"]["fix_scale"])
    )
    deformable_agg_flops_det = (
        cfg.num_decoder
        * cfg.embed_dims
        * cfg.num_levels
        * cfg.model["head"]['det_head']["instance_bank"]["num_anchor"]
        * cfg.model["head"]['det_head']["deformable_model"]["num_cams"]
        * num_key_pts_det
        * bilinear_flops
    )
    num_key_pts_map = (
        cfg.model["head"]['map_head']["deformable_model"]["kps_generator"]["num_learnable_pts"]
        + len(cfg.model["head"]['map_head']["deformable_model"]["kps_generator"]["fix_height"])
    ) * cfg.model["head"]['map_head']["deformable_model"]["kps_generator"]["num_sample"]
    deformable_agg_flops_map = (
        cfg.num_decoder
        * cfg.embed_dims
        * cfg.num_levels
        * cfg.model["head"]['map_head']["instance_bank"]["num_anchor"]
        * cfg.model["head"]['map_head']["deformable_model"]["num_cams"]
        * num_key_pts_map
        * bilinear_flops
    )
    deformable_agg_flops = deformable_agg_flops_det + deformable_agg_flops_map

    for module in ["total", "img_backbone", "img_neck", "head"]:
        if module != "total":
            flops_model = add_flops_counting_methods(getattr(model, module))
        else:
            flops_model = add_flops_counting_methods(model)
        flops_model.eval()
        flops_model.start_flops_count()
        
        if module == "img_backbone":
            flops_model(data["img"].flatten(0, 1))
        elif module == "img_neck":
            flops_model(model.img_backbone(data["img"].flatten(0, 1)))
        elif module == "head":
            flops_model(model.extract_feat(data["img"], metas=data), data)
        else:
            flops_model(**data)
        flops_count, params_count = flops_model.compute_average_flops_cost()
        flops_count *= flops_model.__batch_counter__
        flops_model.stop_flops_count()
        if module == "head" or module == "total":
            flops_count += deformable_agg_flops
        if module == "total":
            total_flops = flops_count
            total_params = params_count
        print(
            f"{module:<13} complexity: "
            f"FLOPs={flops_count/ 10.**9:>8.4f} G / {flops_count/total_flops*100:>6.2f}%, "
            f"Params={params_count/10**6:>8.4f} M / {params_count/total_params*100:>6.2f}%."
        )

if __name__ == '__main__':
    main()
