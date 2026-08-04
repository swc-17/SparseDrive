import hashlib
import importlib.util
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import yaml
import torch

import lilypad_entrypoint_navsim as entrypoint
from navsim_agent.score_pdm_v1 import _validate_complete_scores
from navsim_agent.score_epdms_two_stage import _validate_required_epdms
from navsim_agent.run_inference_infos import (
    _required_geometry_occurrences as _required_info_occurrences,
    _validate_eval_contract,
)
from navsim_agent.runner import SparseDriveRunner
from navsim_agent.run_inference_frames import (
    _emperror_index_sha256,
    _required_geometry_occurrences as _required_frame_occurrences,
    _required_geometry_tokens,
    _validate_eval_contract as _validate_frames_contract,
)


YAML_PATH = (
    "lilypad_config/navsim_eval/"
    "eval_navtest_sd15_metric_teacher_geometry_ego_v1.yaml"
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_instance_queue_for_reset_test():
    """Load InstanceQueue without importing the CUDA/MMCV plugin package."""

    class Registry:
        @staticmethod
        def register_module():
            return lambda cls: cls

    modules = {
        "mmcv": ModuleType("mmcv"),
        "mmcv.utils": ModuleType("mmcv.utils"),
        "mmcv.cnn": ModuleType("mmcv.cnn"),
        "mmcv.cnn.bricks": ModuleType("mmcv.cnn.bricks"),
        "mmcv.cnn.bricks.registry": ModuleType("mmcv.cnn.bricks.registry"),
        "projects": ModuleType("projects"),
        "projects.mmdet3d_plugin": ModuleType("projects.mmdet3d_plugin"),
        "projects.mmdet3d_plugin.ops": ModuleType(
            "projects.mmdet3d_plugin.ops"
        ),
        "projects.mmdet3d_plugin.core": ModuleType(
            "projects.mmdet3d_plugin.core"
        ),
        "projects.mmdet3d_plugin.core.box3d": ModuleType(
            "projects.mmdet3d_plugin.core.box3d"
        ),
    }
    modules["mmcv.utils"].build_from_cfg = lambda *args, **kwargs: None
    modules["mmcv.cnn.bricks.registry"].PLUGIN_LAYERS = Registry()
    modules["projects.mmdet3d_plugin.ops"].feature_maps_format = lambda value, **_: value
    box3d = modules["projects.mmdet3d_plugin.core.box3d"]
    for index, name in enumerate(
        ("X", "Y", "Z", "W", "L", "H", "SIN_YAW", "COS_YAW", "VX", "VY", "VZ")
    ):
        setattr(box3d, name, index)
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "projects/mmdet3d_plugin/models/motion/instance_queue.py",
    )
    spec = importlib.util.spec_from_file_location(
        "_sd15_instance_queue_reset_test", path
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.InstanceQueue


class Sd15GeometryEvalTest(unittest.TestCase):
    def test_instance_queue_missing_mask_bridge_is_opt_in(self):
        InstanceQueue = load_instance_queue_for_reset_test()

        def second_frame(allow_missing_mask):
            queue = InstanceQueue(
                embed_dims=4,
                queue_length=2,
                tracking_threshold=0,
                feature_map_scale=(2, 2),
                use_cam_ego_feature=False,
            )
            queue._allow_missing_detection_mask = allow_missing_mask

            def output():
                return {
                    "instance_feature": torch.zeros(1, 2, 4),
                    "prediction": [torch.zeros(1, 2, 11)],
                    "classification": [torch.zeros(1, 2, 1)],
                    "instance_id": torch.tensor([[0, 1]]),
                }

            first = output()
            ego_feature, *_ = queue.get(
                first, [], {"img_metas": [{}]}, 1,
                torch.ones(1, dtype=torch.bool), None,
            )
            queue.cache_motion(
                first["instance_feature"], first, {"img_metas": [{}]}
            )
            ego_status = torch.zeros(1, 1, 10)
            ego_status[..., 6] = 4.25
            queue.cache_planning(ego_feature, ego_status)

            observed_masks = []
            original = queue.prepare_planning

            def capture(feature_maps, mask, batch_size, ego_status=None):
                observed_masks.append(mask)
                return original(
                    feature_maps, mask, batch_size, ego_status=ego_status
                )

            queue.prepare_planning = capture
            result = queue.get(
                output(), [], {"img_metas": [{}]}, 1, None, None
            )
            return queue, result, observed_masks

        with self.assertRaises(TypeError):
            second_frame(False)

        queue, result, observed_masks = second_frame(True)
        self.assertEqual(observed_masks[0].dtype, torch.bool)
        self.assertTrue(observed_masks[0].all())
        self.assertEqual(queue.ego_period.item(), 2)
        self.assertAlmostEqual(result[1][0, 0, 9].item(), 4.25)

    def test_geometry_occurrence_keys_do_not_reach_model_pipeline(self):
        seen = {}

        def capture(value):
            seen.update(value)
            return value

        runner = SparseDriveRunner.__new__(SparseDriveRunner)
        runner.pipeline = SimpleNamespace(transforms=[capture])
        runner.aug_config = {}
        runner.device = "cpu"
        parallel = ModuleType("mmcv.parallel")
        parallel.collate = lambda values, samples_per_gpu: values[0]
        parallel.scatter = lambda value, devices: [value]
        mmcv = ModuleType("mmcv")
        mmcv.parallel = parallel
        with mock.patch.dict(
            sys.modules, {"mmcv": mmcv, "mmcv.parallel": parallel}
        ):
            runner._prepare(
                {
                    "_frame_token": "frame",
                    "_geometry_scenario_token": "current",
                    "_geometry_frame_index": 2,
                    "value": 1,
                }
            )
        self.assertEqual(seen["value"], 1)
        self.assertNotIn("_frame_token", seen)
        self.assertNotIn("_geometry_scenario_token", seen)
        self.assertNotIn("_geometry_frame_index", seen)

    def test_frame_reset_preserves_planner_history(self):
        class Bank:
            def __init__(self):
                self.resets = 0

            def reset(self):
                self.resets += 1

        det, maps, planner = Bank(), Bank(), Bank()
        runner = SparseDriveRunner.__new__(SparseDriveRunner)
        runner.reset_perception_each_frame = True
        runner.planner = SimpleNamespace(instance_queue=planner)
        runner.model = SimpleNamespace(
            head=SimpleNamespace(
                det_head=SimpleNamespace(instance_bank=det),
                map_head=SimpleNamespace(instance_bank=maps),
                motion_plan_head=SimpleNamespace(instance_queue=planner),
            )
        )
        runner.configure_frame_independent_perception()
        self.assertEqual(runner.reset_perception_temporal_state(), 2)
        self.assertEqual((det.resets, maps.resets, planner.resets), (1, 1, 0))
        self.assertTrue(planner._allow_missing_detection_mask)

    def test_score_only_trajectory_gate_is_exact_and_finite(self):
        payload = {
            "trajectories": {
                "token": [[0.0, 0.0, 0.0] for _ in range(8)]
            }
        }
        with tempfile.NamedTemporaryFile(suffix=".pkl") as handle:
            pickle.dump(payload, handle)
            handle.flush()
            loaded = entrypoint._validate_scoring_trajectories(handle.name, 1)
            self.assertEqual(set(loaded["trajectories"]), {"token"})
            with self.assertRaisesRegex(RuntimeError, "count differs"):
                entrypoint._validate_scoring_trajectories(handle.name, 2)

        payload["trajectories"]["token"][0][0] = float("nan")
        with tempfile.NamedTemporaryFile(suffix=".pkl") as handle:
            pickle.dump(payload, handle)
            handle.flush()
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                entrypoint._validate_scoring_trajectories(handle.name, 1)

    def test_required_pdms_fails_closed(self):
        _validate_complete_scores(12146, 12146, 12146)
        with self.assertRaisesRegex(RuntimeError, "count differs"):
            _validate_complete_scores(12146, 12145, 12146)

    def test_required_epdms_fails_closed(self):
        scores = {"combined": 0.3, "stage_one": 0.6, "stage_two": 0.4}
        _validate_required_epdms(scores, True, 5912, 5912, 5912)
        with self.assertRaisesRegex(RuntimeError, "aggregation is invalid"):
            _validate_required_epdms(scores, False, 5912, 5912, 5912)
        with self.assertRaisesRegex(RuntimeError, "count differs"):
            _validate_required_epdms(scores, True, 5912, 5911, 5912)

    def test_navhard_uses_real_deduplicated_frame_tokens(self):
        frames = {
            "scene-a": [{"_frame_token": "shared"}, {"_frame_token": "a"}],
            "scene-b": [{"_frame_token": "shared"}],
        }
        self.assertEqual(
            _required_geometry_tokens(frames),
            {"shared", "a"},
        )
        with self.assertRaisesRegex(ValueError, "requires a real _frame_token"):
            _required_geometry_tokens({"scene-a": [{}]})
        contract_frames = {"scene-a": [{}, {}, {}, {}]}
        self.assertEqual(
            _validate_frames_contract(contract_frames, 1, 4),
            {"scenarios": 1, "replay_calls": 4},
        )
        self.assertEqual(
            _emperror_index_sha256({"emperror_index_sha256": "a" * 64}),
            "a" * 64,
        )
        with self.assertRaisesRegex(ValueError, "emperror_index_sha256"):
            _emperror_index_sha256({})

        ordered_frames = {
            "current": [
                {"_frame_token": token}
                for token in ("h0", "h1", "h2", "current")
            ]
        }
        expected = {"current": ("h0", "h1", "h2", "current")}
        self.assertEqual(_required_frame_occurrences(ordered_frames), expected)
        self.assertEqual(
            _required_info_occurrences(
                [
                    {
                        "token": "current",
                        "history_tokens": ["h0", "h1", "h2", "current"],
                    }
                ]
            ),
            expected,
        )

    def test_sd15_perception_and_planner_are_eval_defaults(self):
        config = entrypoint._with_sd15_eval_defaults(
            {"geometry_source": "emperror"}
        )
        self.assertEqual(config["config_file"], entrypoint.SD15_METRIC_CONFIG_FILE)
        self.assertEqual(
            config["config_sha256"], entrypoint.SD15_METRIC_CONFIG_SHA256
        )
        self.assertEqual(
            config["checkpoint_s3"], entrypoint.SD15_METRIC_CHECKPOINT_S3
        )
        self.assertEqual(
            config["checkpoint_sha256"],
            entrypoint.SD15_METRIC_CHECKPOINT_SHA256,
        )
        self.assertEqual(len(config["stage_files"]), 6)
        self.assertEqual(
            config["geometry_teacher_config_sha256"],
            entrypoint.SD15_METRIC_CONFIG_SHA256,
        )
        self.assertEqual(
            config["geometry_teacher_checkpoint_sha256"],
            entrypoint.SD15_METRIC_CHECKPOINT_SHA256,
        )
        explicit = entrypoint._with_sd15_eval_defaults(
            {"config_file": "legacy.py", "checkpoint_s3": "s3://legacy/model.pth"}
        )
        self.assertEqual(explicit["config_file"], "legacy.py")
        self.assertNotIn("stage_files", explicit)
        with self.assertRaisesRegex(ValueError, "provided together"):
            entrypoint._with_sd15_eval_defaults({"config_file": "partial.py"})

    def test_teacher_yaml_pins_exact_navtest_assets(self):
        with open(YAML_PATH) as handle:
            payload = yaml.safe_load(handle)
        self.assertEqual(payload["cluster_resources"]["num_gpus"], 8)
        config = payload["workload_variant_config"]["entrypoint_fn_config"]
        self.assertEqual((config["num_gpus"], config["num_shards"]), (8, 8))
        self.assertEqual(config["geometry_source"], "teacher")
        self.assertTrue(config["ego_query_only"])
        self.assertEqual(
            (config["expected_scenarios"], config["expected_replay_calls"]),
            (12146, 48584),
        )
        self.assertEqual(sha256(config["config_file"]), config["config_sha256"])
        self.assertEqual(
            sha256("work_dirs/navsim_stage2_vocab_metric_16g/iter_8703.pth"),
            config["checkpoint_sha256"],
        )
        self.assertEqual(
            sha256("/home/tejan/lwm-rl/SparseDrive/data/infos/"
                   "navsim_infos_navtest.pkl"),
            config["infos_sha256"],
        )
        self.assertEqual(len(config["stage_files"]), 6)
        for entry in config["stage_files"]:
            self.assertEqual(
                set(entry), {"s3_rel", "repo_rel", "sha256"}
            )
            self.assertEqual(sha256(entry["repo_rel"]), entry["sha256"])

    def test_infos_contract_requires_four_frames_and_exact_counts(self):
        infos = [
            {"history_tokens": [f"{index}-{step}" for step in range(4)]}
            for index in range(2)
        ]
        self.assertEqual(
            _validate_eval_contract(infos, 2, 8),
            {"scenarios": 2, "replay_calls": 8},
        )
        infos[0]["history_tokens"].pop()
        with self.assertRaisesRegex(ValueError, "exactly four"):
            _validate_eval_contract(infos, 2, 7)

    def test_two_shard_teacher_merge_totals_gate_and_replays(self):
        run_tag = "test_sd15_geometry_merge"
        out_dir = f"/tmp/eval_out/{run_tag}"
        shutil.rmtree(out_dir, ignore_errors=True)
        config_sha = "2" * 64
        checkpoint_sha = "3" * 64
        infos_sha = "1" * 64
        config = {
            "mode": "pdm_v1",
            "split": "navtest",
            "run_tag": run_tag,
            "geometry_source": "teacher",
            "ego_query_only": True,
            "config_file": "projects/configs/navsim/"
                           "sparsedrive_navsim_stage2_vocab_metric_full.py",
            "config_sha256": config_sha,
            "checkpoint_s3": "s3://bucket/checkpoint.pth",
            "checkpoint_sha256": checkpoint_sha,
            "infos_s3": "s3://bucket/infos.pkl",
            "infos_sha256": infos_sha,
            "expected_scenarios": 2,
            "expected_replay_calls": 8,
            "num_gpus": 2,
            "num_shards": 2,
            "results_s3_output": "s3://bucket/results/",
            "stage_files": [
                {
                    "s3_rel": "hashed/source.npy",
                    "repo_rel": "hashed/destination.npy",
                    "sha256": config_sha,
                },
                "legacy/same-path.npy",
                ["legacy/source.npy", "legacy/destination.npy"],
            ],
            "frame_tars_s3": "s3://bucket/frames/",
            "metric_cache_s3": "s3://bucket/cache/",
            "navsim_logs_s3": "s3://bucket/logs/test/",
            "sdv2_snapshot_s3": "s3://bucket/sdv2.tar.gz",
            "eval_env_s3": "s3://bucket/eval.tar.gz",
            "maps_s3": "s3://bucket/maps/",
        }

        def fake_sha(path):
            if path.endswith(".pth"):
                return checkpoint_sha
            if "/tmp/eval_infos/" in path:
                return infos_sha
            return config_sha

        def fake_run(command, tag=None, **_kwargs):
            if not tag or not tag.startswith("inference-shard"):
                return
            shard = int(command[command.index("--shard") + 1])
            output = command[command.index("--output") + 1]
            with open(output, "wb") as handle:
                pickle.dump(
                    {
                        "trajectories": {f"token-{shard}": [[0.0, 0.0, 0.0]]},
                        "config_sha256": config_sha,
                        "checkpoint": f"/tmp/eval_ckpt_{run_tag}.pth",
                        "checkpoint_sha256": checkpoint_sha,
                        "infos_sha256": infos_sha,
                        "expected_scenarios": 2,
                        "expected_replay_calls": 8,
                        "geometry_source": "teacher",
                        "ego_query_only": True,
                        "geometry_gate": {
                            "calls": 4,
                            "max_reencode_error": (shard + 1) * 1e-6,
                            "max_source_delta": None,
                            "ego_query_only": True,
                        },
                        "geometry_artifact": None,
                        "replay_stats": {"scenarios": 1, "replay_calls": 4},
                    },
                    handle,
                )

        with mock.patch.object(entrypoint, "_sha256_file", side_effect=fake_sha), \
                mock.patch.object(entrypoint, "_once"), \
                mock.patch.object(entrypoint, "_s3_client", return_value=object()), \
                mock.patch.object(entrypoint, "_run_cmd", side_effect=fake_run), \
                mock.patch.object(entrypoint, "_put_file_nonchunked"):
            entrypoint._run_navsim_eval(config)

        with open(os.path.join(out_dir, f"trajs_{run_tag}.pkl"), "rb") as handle:
            merged = pickle.load(handle)
        self.assertEqual(set(merged["trajectories"]), {"token-0", "token-1"})
        self.assertEqual(
            merged["replay_stats"], {"scenarios": 2, "replay_calls": 8}
        )
        self.assertEqual(merged["geometry_gate"]["calls"], 8)
        self.assertEqual(merged["geometry_gate"]["max_reencode_error"], 2e-6)
        shutil.rmtree(out_dir, ignore_errors=True)

    def test_navhard_emperror_is_wired_to_frame_inference(self):
        run_tag = "test_sd15_navhard_emperror"
        config_sha = "2" * 64
        checkpoint_sha = "3" * 64
        geometry_sha = "4" * 64
        producer_sha = "5" * 64
        teacher_config_sha = "6" * 64
        teacher_checkpoint_sha = "7" * 64
        frames_sha = "8" * 64
        config = {
            "mode": "epdms_two_stage",
            "split": "navhard_two_stage",
            "run_tag": run_tag,
            "geometry_source": "emperror",
            "geometry_s3": "s3://bucket/geometry.pkl",
            "geometry_sha256": geometry_sha,
            "geometry_producer_checkpoint_sha256": producer_sha,
            "geometry_teacher_config_sha256": teacher_config_sha,
            "geometry_teacher_checkpoint_sha256": teacher_checkpoint_sha,
            "ego_query_only": True,
            "config_file": "planner.py",
            "config_sha256": config_sha,
            "checkpoint_s3": "s3://bucket/checkpoint.pth",
            "checkpoint_sha256": checkpoint_sha,
            "frames_pkl_s3": "s3://bucket/frames.pkl",
            "frames_sha256": frames_sha,
            "expected_scenarios": 5912,
            "expected_replay_calls": 23648,
            "num_gpus": 1,
            "num_shards": 1,
            "results_s3_output": "s3://bucket/results/",
            "stage_files": [],
            "metric_cache_s3": "s3://bucket/cache/",
            "navsim_logs_s3": "s3://bucket/logs/test/",
            "sdv2_snapshot_s3": "s3://bucket/sdv2.tar.gz",
            "eval_env_s3": "s3://bucket/eval.tar.gz",
        }
        commands = []

        def fake_sha(path):
            if "/tmp/eval_geometry/" in path:
                return geometry_sha
            if "/tmp/eval_frames/" in path:
                return frames_sha
            if path.endswith(".pth"):
                return checkpoint_sha
            return config_sha

        def fake_run(command, **_kwargs):
            commands.append(command)

        with mock.patch.object(entrypoint, "_sha256_file", side_effect=fake_sha), \
                mock.patch.object(entrypoint, "_once"), \
                mock.patch.object(entrypoint, "_s3_client", return_value=object()), \
                mock.patch.object(entrypoint, "_run_cmd", side_effect=fake_run), \
                mock.patch.object(entrypoint, "_put_file_nonchunked"):
            entrypoint._run_navsim_eval(config)

        inference = next(
            command for command in commands
            if "navsim_agent.run_inference_frames" in command
        )
        expected = {
            "--geometry-source": "emperror",
            "--geometry-sha256": geometry_sha,
            "--geometry-producer-checkpoint-sha256": producer_sha,
            "--geometry-teacher-config-sha256": teacher_config_sha,
            "--geometry-teacher-checkpoint-sha256": teacher_checkpoint_sha,
            "--checkpoint-sha256": checkpoint_sha,
            "--config-sha256": config_sha,
            "--frames-sha256": frames_sha,
            "--expected-scenarios": "5912",
            "--expected-replay-calls": "23648",
        }
        for flag, value in expected.items():
            self.assertEqual(inference[inference.index(flag) + 1], value)
        self.assertIn("--geometry-file", inference)
        self.assertIn("--geometry-uri", inference)
        self.assertIn("--ego-query-only", inference)

    def test_navmini_occurrence_file_uses_pinned_artifact_without_model_flags(self):
        run_tag = "test_sd15_navmini_occurrence"
        config_sha = "2" * 64
        checkpoint_sha = "3" * 64
        geometry_sha = "4" * 64
        infos_sha = "5" * 64
        config = {
            "mode": "pdm_v1",
            "split": "navmini",
            "run_tag": run_tag,
            "geometry_source": "emperror_occurrence",
            "geometry_s3": "s3://bucket/geometry.pkl",
            "geometry_sha256": geometry_sha,
            "ego_query_only": True,
            "config_file": "planner.py",
            "config_sha256": config_sha,
            "checkpoint_s3": "s3://bucket/checkpoint.pth",
            "checkpoint_sha256": checkpoint_sha,
            "infos_s3": "s3://bucket/infos.pkl",
            "infos_sha256": infos_sha,
            "expected_scenarios": 396,
            "expected_replay_calls": 1584,
            "num_gpus": 1,
            "num_shards": 1,
            "results_s3_output": "s3://bucket/results/",
            "stage_files": [],
            "frame_tars_s3": "s3://bucket/frames/",
            "metric_cache_s3": "s3://bucket/cache/",
            "navsim_logs_s3": "s3://bucket/logs/test/",
            "sdv2_snapshot_s3": "s3://bucket/sdv2.tar.gz",
            "eval_env_s3": "s3://bucket/eval.tar.gz",
            "maps_s3": "s3://bucket/maps/",
        }
        commands = []

        def fake_sha(path):
            if "/tmp/eval_geometry/" in path:
                return geometry_sha
            if "/tmp/eval_infos/" in path:
                return infos_sha
            if path.endswith(".pth"):
                return checkpoint_sha
            return config_sha

        with mock.patch.object(entrypoint, "_sha256_file", side_effect=fake_sha), \
                mock.patch.object(entrypoint, "_once"), \
                mock.patch.object(entrypoint, "_s3_client", return_value=object()), \
                mock.patch.object(
                    entrypoint, "_run_cmd", side_effect=lambda command, **_: commands.append(command)
                ), \
                mock.patch.object(entrypoint, "_put_file_nonchunked"):
            entrypoint._run_navsim_eval(config)

        inference = next(
            command
            for command in commands
            if "navsim_agent.run_inference_infos" in command
        )
        self.assertEqual(
            inference[inference.index("--geometry-source") + 1],
            "emperror_occurrence",
        )
        self.assertEqual(
            inference[inference.index("--geometry-sha256") + 1], geometry_sha
        )
        self.assertNotIn("--geometry-producer-checkpoint-sha256", inference)
        self.assertNotIn("--geometry-teacher-config-sha256", inference)
        self.assertNotIn("--geometry-teacher-checkpoint-sha256", inference)
        self.assertIn("--ego-query-only", inference)

        invalid = dict(config, geometry_producer_checkpoint_sha256="6" * 64)
        with self.assertRaisesRegex(ValueError, "immutable artifact"):
            entrypoint._run_navsim_eval(invalid)


if __name__ == "__main__":
    unittest.main()
