"""SparseDrive per-scenario inference: bank reset + 4-frame history replay.

Runs in the SparseDrive environment (sparsedrive310). Loads a config +
checkpoint once, then, per NAVSIM scenario token:

1. resets ALL temporal state (detection/map ``InstanceBank``s and the
   motion/planning ``InstanceQueue``) so no state leaks between scenario
   tokens or two-stage frames;
2. replays the four history frames oldest-to-current through the exact
   test pipeline of the training config (undistort -> resize/crop ->
   normalize -> adaptor);
3. returns the current frame's ``final_planning`` (ego_fut_ts, 2)
   cumulative xy in the SparseDrive BEV frame.
"""
import copy

import numpy as np
import torch

from .geometry_gate import GENERATED_GEOMETRY_SOURCES, PLANNER_GEOMETRY_SOURCES


def _load_plugin(cfg, repo_root):
    import importlib
    import os
    import sys

    if cfg.get("plugin", False):
        plugin_dir = cfg.get("plugin_dir", "projects/mmdet3d_plugin/")
        module_path = os.path.normpath(plugin_dir).replace(os.sep, ".")
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        importlib.import_module(module_path.rstrip("."))


def deterministic_test_aug(data_aug_conf):
    """The deterministic test-mode branch of
    ``NavSim3DDataset.get_augmentation`` (resize to cover, center/bottom
    crop, no flip/rotate)."""
    H, W = data_aug_conf["H"], data_aug_conf["W"]
    fH, fW = data_aug_conf["final_dim"]
    resize = max(fH / H, fW / W)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.mean(data_aug_conf["bot_pct_lim"])) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    return {
        "resize": resize,
        "resize_dims": resize_dims,
        "crop": crop,
        "flip": False,
        "rotate": 0,
        "rotate_3d": 0,
    }


class SparseDriveRunner:
    def __init__(
        self,
        config_path,
        checkpoint_path,
        device="cuda:0",
        repo_root=None,
        geometry_source=None,
        geometry_producer=None,
        ego_query_only=False,
        reset_perception_each_frame=False,
    ):
        import os

        from mmcv import Config
        from mmcv.runner import load_checkpoint, wrap_fp16_model
        from mmdet.models import build_detector
        from mmdet.datasets.pipelines import Compose

        repo_root = repo_root or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        cfg = Config.fromfile(config_path)
        _load_plugin(cfg, repo_root)

        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        if cfg.get("fp16", None) is not None:
            wrap_fp16_model(model)
        checkpoint = None
        if checkpoint_path is not None:
            checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
        model.to(device)
        model.eval()

        self.cfg = cfg
        self.model = model
        self.device = device
        self.geometry_source = geometry_source
        self.geometry_producer = geometry_producer
        self.ego_query_only = ego_query_only
        self.reset_perception_each_frame = reset_perception_each_frame
        self.planner = model.head.motion_plan_head
        self.configure_frame_independent_perception()
        if geometry_source is None and geometry_producer is not None:
            raise ValueError("a geometry producer requires a generated geometry source")
        if geometry_source is not None or ego_query_only:
            if checkpoint is None:
                raise RuntimeError("the geometry gate requires a trained checkpoint")
            checkpoint_state = checkpoint.get("state_dict", checkpoint)
            model_state = model.state_dict()
            prefixes = (
                "head.motion_plan_head.agent_geo_encoder.",
                "head.motion_plan_head.map_geo_encoder.",
            )
            critical = {
                name: value.shape
                for name, value in model_state.items()
                if name.startswith(prefixes)
            }
            invalid = [
                name
                for name, shape in critical.items()
                if name not in checkpoint_state
                or checkpoint_state[name].shape != shape
            ]
            if len(critical) != 12 or invalid:
                raise RuntimeError(
                    f"checkpoint lacks compatible V6 geometry encoders: {invalid}"
                )
        if geometry_source is not None:
            if geometry_source not in PLANNER_GEOMETRY_SOURCES:
                raise ValueError(f"unknown planner geometry source {geometry_source!r}")
            if (
                geometry_source in GENERATED_GEOMETRY_SOURCES
                and geometry_producer is None
            ):
                raise RuntimeError("EMPERROR geometry source requires a producer")
            if (
                geometry_source not in GENERATED_GEOMETRY_SOURCES
                and geometry_producer is not None
            ):
                raise ValueError(
                    "a geometry producer requires a generated geometry source"
                )
            if geometry_source in GENERATED_GEOMETRY_SOURCES and not ego_query_only:
                raise RuntimeError("EMPERROR geometry requires ego_query_only=True")
            if not self.planner.geometric_inputs:
                raise RuntimeError("planner geometry gate requires a V6 checkpoint")
            if self.planner.training:
                raise RuntimeError("planner geometry gate is evaluation-only")
            if (self.planner.num_det, self.planner.num_map) != (50, 10):
                raise RuntimeError("planner geometry gate requires num_det/map=50/10")
            if self.planner.instance_queue.use_cam_ego_feature:
                raise RuntimeError("planner geometry gate forbids camera ego features")
            self.planner._planner_geometry_source = geometry_source
            self.planner._planner_geometry_gate_calls = 0
            self.planner._planner_geometry_gate_max_error = None
            self.planner._planner_geometry_gate_max_source_delta = None
            # Avoid registering the already-owned map encoder a second time.
            object.__setattr__(
                self.planner,
                "_planner_map_anchor_encoder",
                model.head.map_head.anchor_encoder,
            )
        if ego_query_only:
            if not self.planner.geometric_inputs:
                raise RuntimeError("ego_query_only requires a V6 checkpoint")
            if self.planner.planning_decoder.use_rescore:
                raise RuntimeError("ego_query_only requires use_rescore=False")
            expected_classes = [
                "vehicle",
                "pedestrian",
                "bicycle",
                "traffic_cone",
                "barrier",
                "czone_sign",
                "generic_object",
            ]
            checkpoint_meta = checkpoint.get("meta") or {}
            if checkpoint_meta.get("CLASSES") != expected_classes:
                raise RuntimeError("ego_query_only requires the NAVSIM classes")
            if list(cfg.get("map_class_names", [])) != [
                "ped_crossing",
                "divider",
                "boundary",
            ]:
                raise RuntimeError("ego_query_only requires the NAVSIM map classes")
            query_separable_ops = {
                "temp_gnn",
                "gnn",
                "norm",
                "ffn",
                "cross_gnn",
                "refine",
            }
            if not set(self.planner.operation_order) <= query_separable_ops:
                raise RuntimeError("ego_query_only encountered an unverified planner op")
            incompatible = (
                "_inject_planner_inputs",
                "_gt_oracle_enabled",
                "ablate_gnn_zero",
                "ablate_gnn_anchor_mean",
                "ablate_gnn_both_mean",
                "ablate_gnn_permute",
                "ablate_gnn_ego_only",
                "ablate_gnn_instance_features",
                "ablate_map_zero",
                "ablate_query_zero",
                "cjepa_cfg",
                "cjepa_inject_features",
                "cjepa_withcam_data",
            )
            active = []
            for name in incompatible:
                value = getattr(self.planner, name, None)
                if value is not None and value is not False:
                    active.append(name)
            if active:
                raise RuntimeError(
                    f"ego_query_only is incompatible with planner overrides: {active}"
                )
            # Every verified planner op is query-separable, and the output
            # consumes only the ego query; agents remain available as K/V.
            self.planner.drop_agent_queries = True
        # exact test pipeline of the training config, minus the file loader
        # when images are handed over in memory
        self.pipeline = Compose(
            [copy.deepcopy(p) for p in cfg.data.test.pipeline]
        )
        self.aug_config = deterministic_test_aug(cfg.data.test.data_aug_conf)
        self.ego_fut_ts = cfg.get("ego_fut_ts", 8)

    # ------------------------------------------------------------------ #
    def reset_temporal_state(self):
        """Reset every temporal bank/queue in the model (per scenario)."""
        n = 0
        for module in self.model.modules():
            if type(module).__name__ in ("InstanceBank", "InstanceQueue"):
                module.reset()
                n += 1
        assert n > 0, "no temporal banks found to reset"
        return n

    def configure_frame_independent_perception(self):
        """Bridge an absent detector mask to a continuous ego planner queue."""
        self.planner.instance_queue._allow_missing_detection_mask = bool(
            self.reset_perception_each_frame
        )

    def reset_perception_temporal_state(self):
        """Reset detection/map banks while preserving planner history."""
        n = 0
        for name in ("det_head", "map_head"):
            bank = getattr(
                getattr(self.model.head, name, None), "instance_bank", None
            )
            if bank is None:
                raise RuntimeError(f"SparseDrive {name} has no instance bank")
            bank.reset()
            n += 1
        return n

    # ------------------------------------------------------------------ #
    def _prepare(self, input_dict):
        """Pre-pipeline input dict -> collated, device-scattered batch."""
        from mmcv.parallel import collate, scatter

        input_dict = dict(input_dict)
        images = input_dict.pop("_images_rgb", None)
        input_dict.pop("_frame_token", None)
        input_dict.pop("_geometry_scenario_token", None)
        input_dict.pop("_geometry_frame_index", None)
        if images is not None and any(im is not None for im in images):
            # in-memory RGB frames (e.g. synthetic renders): convert to the
            # BGR layout mmcv.imread would have produced so the rest of the
            # pipeline (undistort/resize/normalize to_rgb=True) is identical
            assert all(im is not None for im in images)
            imgs = [
                np.ascontiguousarray(
                    np.asarray(im)[..., ::-1].astype(np.float32)
                )
                for im in images
            ]
            input_dict["img"] = imgs
            input_dict["img_shape"] = [im.shape for im in imgs]
            input_dict["ori_shape"] = [im.shape for im in imgs]
            input_dict["pad_shape"] = [im.shape for im in imgs]
            input_dict["scale_factor"] = 1.0
            input_dict["img_norm_cfg"] = dict(
                mean=np.zeros(3, dtype=np.float32),
                std=np.ones(3, dtype=np.float32),
                to_rgb=False,
            )
            input_dict["filename"] = input_dict.get("img_filename")
            pipeline = self.pipeline.transforms[1:]  # skip the file loader
        else:
            pipeline = self.pipeline.transforms

        input_dict["aug_config"] = copy.deepcopy(self.aug_config)
        for t in pipeline:
            input_dict = t(input_dict)
            assert input_dict is not None
        batch = collate([input_dict], samples_per_gpu=1)
        if "cuda" in str(self.device):
            batch = scatter(batch, [torch.device(self.device).index])[0]
        else:
            batch = {
                k: (v.data[0] if hasattr(v, "data") else v)
                for k, v in batch.items()
            }
        return batch

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_scenario(self, frames):
        """Replay ``frames`` (oldest..current pre-pipeline input dicts) and
        return the current frame's plan.

        :return: (ego_fut_ts, 2) float64 cumulative xy in the SparseDrive
            BEV frame.
        """
        assert len(frames) >= 1
        self.reset_temporal_state()
        gate_calls = getattr(self.planner, "_planner_geometry_gate_calls", 0)
        result = None
        for input_dict in frames:
            if self.reset_perception_each_frame:
                self.reset_perception_temporal_state()
            if self.geometry_source in GENERATED_GEOMETRY_SOURCES:
                self.planner._planner_geometry_for_frame = (
                    self.geometry_producer(input_dict)
                )
            try:
                batch = self._prepare(input_dict)
                img = batch.pop("img")
                result = self.model(img=img, **batch)
            finally:
                if self.geometry_source in GENERATED_GEOMETRY_SOURCES:
                    self.planner._planner_geometry_for_frame = None
        if self.geometry_source is not None:
            calls = getattr(self.planner, "_planner_geometry_gate_calls", 0)
            if calls - gate_calls != len(frames):
                raise RuntimeError("planner geometry gate did not run once per frame")
        final_planning = result[0]["img_bbox"]["final_planning"]
        plan = np.asarray(final_planning.detach().cpu().numpy(),
                          dtype=np.float64)
        assert plan.shape == (self.ego_fut_ts, 2), plan.shape
        assert np.isfinite(plan).all(), "non-finite plan"
        return plan

    def geometry_gate_stats(self):
        if self.geometry_source is None:
            return None
        return {
            "calls": getattr(self.planner, "_planner_geometry_gate_calls", 0),
            "max_reencode_error": getattr(
                self.planner, "_planner_geometry_gate_max_error", None
            ),
            "max_source_delta": getattr(
                self.planner, "_planner_geometry_gate_max_source_delta", None
            ),
            "ego_query_only": bool(self.ego_query_only),
            "perception_temporal_reset": (
                "every_frame"
                if self.reset_perception_each_frame
                else "sequence_boundary"
            ),
        }
