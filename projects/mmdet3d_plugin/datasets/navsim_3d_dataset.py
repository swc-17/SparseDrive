import copy
import math
import os

import numpy as np
from torch.utils.data import Dataset
from shapely.geometry import LineString

import mmcv
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose


@DATASETS.register_module()
class NavSim3DDataset(Dataset):
    """Thin NAVSIM dataset over infos produced by navsim_converter.py.

    One item per official scenario token. Mirrors the NuScenes3DDataset info
    contract (boxes/velocity/futures/ego status already in the SparseDrive
    virtual BEV frame), but:

    - 7 NAVSIM classes in fixed order (front camera CAM_F0 at index 0 of the
      fixed 8-camera order);
    - sequence grouping from the converter's connected-run ``sequence_id``
      (never the nuScenes empty-sweeps rule);
    - future collision boxes precomputed by the converter (scenario rows are
      not consecutive frames, so neighbor-index traversal is invalid);
    - no nuScenes SDK anywhere (evaluation is NAVSIM-side, later phases).
    """

    CLASSES = (
        "vehicle",
        "pedestrian",
        "bicycle",
        "traffic_cone",
        "barrier",
        "czone_sign",
        "generic_object",
    )
    MAP_CLASSES = (
        "ped_crossing",
        "divider",
        "boundary",
    )

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        map_classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        test_mode=False,
        use_valid_flag=True,
        data_aug_conf=None,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        work_dir=None,
        eval_config=None,
        tokens=None,
        max_samples=None,
        metric_cache_root=None,
    ):
        super().__init__()
        self.data_root = data_root
        # NAVSIM metric-cache tree ({log_name}/unknown/{token}/
        # metric_cache.pkl) for live PDM metric supervision; None disables.
        self.metric_cache_root = metric_cache_root
        self.ann_file = ann_file
        self.load_interval = load_interval
        # Optional frozen token manifest (e.g. the Phase-1 overfit gate's
        # first-8-sorted-tokens subset). Filtering keeps converter order.
        self.tokens = set(tokens) if tokens is not None else None
        # Optional head-of-split truncation for quick/subset evals
        # (converter order is log-major, so this keeps whole logs together).
        self.max_samples = max_samples
        self.use_valid_flag = use_valid_flag
        self.test_mode = test_mode
        self.with_velocity = with_velocity
        self.modality = modality or dict(
            use_camera=True,
            use_lidar=False,
            use_radar=False,
            use_map=False,
            use_external=False,
        )
        if classes is not None:
            self.CLASSES = classes
        if map_classes is not None:
            self.MAP_CLASSES = map_classes
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        self.data_infos = self.load_annotations(self.ann_file)
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.data_aug_conf = data_aug_conf
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        if with_seq_flag:
            self._set_sequence_group_flag()

        self.work_dir = work_dir
        self.eval_config = eval_config

    def __len__(self):
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format="pkl")
        # Keep converter order (log-major, chronological within a log). Do
        # NOT globally sort all logs by timestamp.
        data_infos = data["infos"][:: self.load_interval]
        if self.tokens is not None:
            data_infos = [i for i in data_infos if i["token"] in self.tokens]
            assert len(data_infos) == len(self.tokens), (
                f"token manifest mismatch: wanted {len(self.tokens)}, "
                f"found {len(data_infos)}"
            )
        if self.max_samples is not None:
            data_infos = data_infos[: self.max_samples]
        self.frame_table = data.get("frames", {})
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        print(self.metadata.get("version"), ":",
              self.metadata.get("num_samples"), "samples")
        return data_infos

    def _set_sequence_group_flag(self):
        """Group by the converter's connected-run sequence_id."""
        seq_ids = [info["sequence_id"] for info in self.data_infos]
        # densify to consecutive group indices
        remap = {}
        flag = []
        for sid in seq_ids:
            if sid not in remap:
                remap[sid] = len(remap)
            flag.append(remap[sid])
        self.flag = np.array(flag, dtype=np.int64)

        if self.sequences_split_num == "all":
            self.flag = np.arange(len(self.data_infos), dtype=np.int64)
        elif self.sequences_split_num not in (1, None):
            bin_counts = np.bincount(self.flag)
            new_flags = []
            curr_new_flag = 0
            for curr_flag in range(len(bin_counts)):
                curr_sequence_length = np.array(
                    list(
                        range(
                            0,
                            bin_counts[curr_flag],
                            math.ceil(
                                bin_counts[curr_flag]
                                / self.sequences_split_num
                            ),
                        )
                    )
                    + [bin_counts[curr_flag]]
                )
                for sub_seq_idx in (
                    curr_sequence_length[1:] - curr_sequence_length[:-1]
                ):
                    for _ in range(sub_seq_idx):
                        new_flags.append(curr_new_flag)
                    curr_new_flag += 1
            assert len(new_flags) == len(self.flag)
            self.flag = np.array(new_flags, dtype=np.int64)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        return {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data = self.pipeline(data)
        return data

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        mask = info["valid_flag"]
        gt_names = set(info["gt_names"][mask])
        return [self.cat2id[name] for name in gt_names if name in self.CLASSES]

    def anno2geom(self, annos):
        map_geoms = {}
        for label, anno_list in annos.items():
            map_geoms[label] = [LineString(anno) for anno in anno_list]
        return map_geoms

    def _img_path(self, data_path):
        if self.data_root is None:
            return data_path
        return os.path.join(self.data_root, data_path)

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            token=info["token"],
            sequence_id=info["sequence_id"],
            map_location=info["map_location"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            ego_status=info["ego_status"].astype(np.float32),
            map_infos=info["map_annos"],
        )
        import pyquaternion

        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(
            info["ego2global_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        # lidar2ego is identity for NAVSIM (ego rear-axle frame)
        input_dict["lidar2global"] = ego2global

        input_dict["map_geoms"] = self.anno2geom(info["map_annos"])

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsic = []
            cam_distortion = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(self._img_path(cam_info["data_path"]))
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = (
                    cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                )
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = copy.deepcopy(cam_info["cam_intrinsic"])
                cam_intrinsic.append(intrinsic)
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt)
                cam_distortion.append(cam_info["distortion"].copy())

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    lidar2cam=lidar2cam_rts,
                    cam_intrinsic=cam_intrinsic,
                    cam_distortion=cam_distortion,
                )
            )

        input_dict.update(self.get_ann_info(index))
        if self.metric_cache_root is not None:
            from mmcv.parallel import DataContainer

            # cpu_only DC: mmcv collate would otherwise recurse into the
            # string char-by-char (str is a Sequence) and blow the stack.
            # Scatter unwraps this to a per-GPU list of B path strings.
            input_dict["metric_cache_path"] = DataContainer(
                os.path.join(
                    self.metric_cache_root,
                    info["log_name"],
                    "unknown",
                    info["token"],
                    "metric_cache.pkl",
                ),
                cpu_only=True,
            )
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        mask = (
            info["valid_flag"]
            if self.use_valid_flag
            else np.ones(len(info["gt_boxes"]), dtype=bool)
        )
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_labels_3d = np.array(
            [
                self.CLASSES.index(cat) if cat in self.CLASSES else -1
                for cat in gt_names_3d
            ]
        )

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            instance_inds=np.array(info["instance_inds"], dtype=np.int64)[mask],
            gt_agent_fut_trajs=info["gt_agent_fut_trajs"][mask],
            gt_agent_fut_masks=info["gt_agent_fut_masks"][mask],
            gt_ego_fut_trajs=info["gt_ego_fut_trajs"],
            gt_ego_fut_masks=info["gt_ego_fut_masks"],
            gt_ego_fut_cmd=info["gt_ego_fut_cmd"],
            # scalar -> shape (1,) so the pipeline's to_tensor/DC-stack works
            gt_ego_fut_cmd_valid=np.asarray(
                info["gt_ego_fut_cmd_valid"], dtype=np.float32
            ).reshape(1),
            # future collision boxes precomputed in the current SD frame
            fut_boxes=[fb.copy() for fb in info["fut_boxes"]],
        )
        return anns_results

    def _format_det_eval_annos(self, results):
        """Build (gt_annos, pred_annos) for the devkit-free detection eval.

        Both sides live in the same SparseDrive virtual BEV frame the model
        is trained in, so no global-frame round trip is needed. GT applies
        the converter validity policy (valid_flag; all retained annotations
        are within the 55 m range by construction).
        """
        gt_annos, pred_annos = [], []
        for i in range(len(results)):
            info = self.data_infos[i]
            mask = (
                info["valid_flag"]
                if self.use_valid_flag
                else np.ones(len(info["gt_boxes"]), dtype=bool)
            )
            velocity = info["gt_velocity"][mask].copy()
            velocity[np.isnan(velocity)] = 0.0
            gt_annos.append(
                dict(
                    boxes=info["gt_boxes"][mask],
                    names=info["gt_names"][mask],
                    velocity=velocity,
                )
            )

            det = results[i].get("img_bbox", results[i])
            boxes = det["boxes_3d"]
            scores = det["scores_3d"]
            labels = det["labels_3d"]
            if hasattr(boxes, "numpy"):
                boxes = boxes.numpy()
            if hasattr(scores, "numpy"):
                scores = scores.numpy()
            if hasattr(labels, "numpy"):
                labels = labels.numpy()
            boxes = np.asarray(boxes, dtype=np.float64).reshape(len(boxes), -1)
            pred_annos.append(
                dict(
                    boxes=boxes[:, :7],
                    scores=np.asarray(scores, dtype=np.float64),
                    labels=np.asarray(labels, dtype=np.int64),
                    velocity=boxes[:, 7:9]
                    if boxes.shape[1] >= 9
                    else np.zeros((len(boxes), 2)),
                )
            )
        return gt_annos, pred_annos

    def format_map_results(self, results, prefix=None):
        """Dump map predictions to the VectorEvaluate submission format
        (same schema as NuScenes3DDataset.format_map_results)."""
        submissions = {"results": {}}
        for j, pred in enumerate(results):
            if pred is None:
                continue
            pred = pred.get("img_bbox", pred)
            single_case = {"vectors": [], "scores": [], "labels": []}
            token = self.data_infos[j]["token"]
            for i in range(len(pred["scores"])):
                vector = pred["vectors"][i]
                if len(vector) < 2:  # a line needs >= 2 points
                    continue
                single_case["vectors"].append(vector)
                single_case["scores"].append(pred["scores"][i])
                single_case["labels"].append(pred["labels"][i])
            submissions["results"][token] = single_case

        out_path = os.path.join(prefix, "submission_vector.json")
        print(f"saving map submission to {out_path}")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mmcv.dump(submissions, out_path)
        return out_path

    def _map_eval_config(self):
        """eval_config copy for VectorEvaluate's GT dataset, with non-dataset
        keys stripped and this dataset's subset selection propagated so gt
        and predictions cover the same samples."""
        eval_config = copy.deepcopy(self.eval_config)
        eval_config.pop("planning_metric", None)
        eval_config.pop("map_eval_workers", None)
        if self.tokens is not None:
            eval_config["tokens"] = sorted(self.tokens)
        if self.max_samples is not None:
            eval_config["max_samples"] = self.max_samples
        if self.load_interval != 1:
            eval_config["load_interval"] = self.load_interval
        return eval_config

    def evaluate(
        self,
        results,
        metric=None,
        logger=None,
        eval_mode=None,
        jsonfile_prefix=None,
        **kwargs,
    ):
        """Offline evaluation mirroring NuScenes3DDataset.evaluate.

        Tasks are selected by ``--eval`` metrics from tools/test.py (any of
        ``bbox``/``det``, ``map``, ``planning``) or, when no metric is
        given, by the config's ``evaluation.eval_mode`` dict (``with_det``,
        ``with_map``, ``with_planning``).

        - detection: devkit-free nuScenes-style mAP/ATE/ASE/AOE/AVE over the
          7 NAVSIM classes (center-distance thresholds 0.5/1/2/4 m, 55 m
          range) — see evaluation/detection/det_eval.py;
        - map: the repo's chamfer-distance vector-map mAP (0.5/1.0/1.5 m
          over ped_crossing/divider/boundary) via VectorEvaluate;
        - planning: local L2/collision diagnostics (Phase 4a). The official
          NAVSIM scores (v1 PDMS, v2 EPDMS) come from the separate
          ``navsim_agent/`` evaluation stack.
        """
        if self.work_dir is not None:
            os.makedirs(self.work_dir, exist_ok=True)
            res_path = os.path.join(self.work_dir, "results_navsim.pkl")
            print("All Results write to", res_path)
            mmcv.dump(results, res_path)

        if metric is not None:
            metrics = [metric] if isinstance(metric, str) else list(metric)
            eval_mode = dict(
                with_det=("bbox" in metrics) or ("det" in metrics),
                with_map="map" in metrics,
                with_planning="planning" in metrics,
            )
        else:
            eval_mode = eval_mode or dict(with_planning=True)

        results_dict = dict()

        if eval_mode.get("with_det", False):
            from .evaluation.detection.det_eval import evaluate_detection

            assert len(results) == len(self.data_infos), (
                f"got {len(results)} results for {len(self.data_infos)} "
                "samples; detection eval needs full split coverage"
            )
            gt_annos, pred_annos = self._format_det_eval_annos(results)
            det_metrics = evaluate_detection(
                gt_annos,
                pred_annos,
                class_names=list(self.CLASSES),
                logger=logger,
            )
            if self.work_dir is not None:
                mmcv.dump(
                    det_metrics,
                    os.path.join(self.work_dir, "metrics_det_navsim.json"),
                )
            prefix = "img_bbox_NavSim"
            for name in self.CLASSES:
                for k, v in det_metrics["label_aps"][name].items():
                    results_dict[f"{prefix}/{name}_AP_dist_{k}"] = round(v, 4)
                for k, v in det_metrics["label_tp_errors"][name].items():
                    results_dict[f"{prefix}/{name}_{k}"] = round(v, 4)
            from .evaluation.detection.det_eval import ERR_NAME_MAPPING

            for k, v in det_metrics["tp_errors"].items():
                results_dict[f"{prefix}/{ERR_NAME_MAPPING[k]}"] = round(v, 4)
            results_dict[f"{prefix}/mAP"] = det_metrics["mean_ap"]
            results_dict[f"{prefix}/NDS"] = det_metrics["nd_score"]

        if eval_mode.get("with_map", False):
            first = results[0].get("img_bbox", results[0]) if results else {}
            if "vectors" not in first:
                from mmcv.utils import print_log

                print_log(
                    "[NavSim3DDataset] map eval requested but results carry "
                    "no 'vectors' (checkpoint/config without a map head) — "
                    "skipping map metrics.",
                    logger=logger,
                )
            else:
                assert self.eval_config is not None, (
                    "NavSim3DDataset.evaluate needs eval_config (dataset "
                    "config with a VectorizeMap eval pipeline) for map eval"
                )
                from .evaluation.map.vector_eval import VectorEvaluate

                # VectorEvaluate's fork-based Pool deadlocks when created
                # inside a CUDA-initialized test process (observed on the
                # local 5090 harness), so default to serial chamfer eval;
                # override via eval_config["map_eval_workers"].
                n_workers = (self.eval_config or {}).get(
                    "map_eval_workers", 0
                )
                map_evaluator = VectorEvaluate(
                    self._map_eval_config(), n_workers=n_workers
                )
                result_path = self.format_map_results(
                    results, prefix=self.work_dir
                )
                map_results_dict = map_evaluator.evaluate(
                    result_path, logger=logger
                )
                results_dict.update(map_results_dict)

        if eval_mode.get("with_planning", False):
            assert self.eval_config is not None, (
                "NavSim3DDataset.evaluate needs eval_config (dataset config "
                "with an eval pipeline collecting gt ego futures/fut_boxes)"
            )
            from .evaluation.planning.planning_eval import planning_eval

            eval_config = copy.deepcopy(self.eval_config)
            # planning-metric horizon/geometry knobs (n_future, ego_width,
            # ego_length, ego_height, ego_center_offset) travel inside
            # eval_config but are not dataset-constructor args.
            metric_kwargs = eval_config.pop("planning_metric", {})
            results_dict.update(
                planning_eval(
                    results, eval_config, logger=logger, **metric_kwargs
                )
            )

        if self.work_dir is not None:
            out = os.path.join(self.work_dir, "metrics_navsim.json")
            mmcv.dump(
                {
                    k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in results_dict.items()
                },
                out,
            )
            print(f"metrics written to {out}")
        return results_dict
