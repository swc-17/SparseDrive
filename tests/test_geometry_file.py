import hashlib
import pickle
import tempfile
import unittest

import numpy as np

from navsim_agent.geometry_file import GeometryFileProducer


def artifact():
    return {
        "schema_version": 1,
        "geometry": {
            "a": {
                "anchor_bbox": np.zeros((50, 11), dtype=np.float32),
                "map_anchor": np.zeros((10, 40), dtype=np.float32),
                "detection_valid": np.ones(50, dtype=np.bool_),
                "map_valid": np.ones(10, dtype=np.bool_),
            }
        },
        "metadata": {
            "infos_sha256": "infos-hash",
            "model": {
                "checkpoint": {
                    "sha256": "c" * 64,
                    "manifest_contract": {
                        "config_sha256": "a" * 64,
                        "checkpoint_sha256": "b" * 64,
                    }
                }
            },
        },
    }


def occurrence_artifact():
    records = []
    for index, token in enumerate(("h0", "h1", "h2", "current")):
        values = artifact()["geometry"]["a"]
        values["anchor_bbox"].fill(index)
        values["map_anchor"][:] = np.tile(
            np.arange(40, dtype=np.float32), (10, 1)
        )
        records.append(
            {
                "sequence_id": "sequence",
                "token": token,
                "geometry": values,
            }
        )
    return {
        "schema_version": 2,
        "geometry_index": "scenario_occurrence",
        "occurrences": {"current": records},
        "metadata": {
            **artifact()["metadata"],
            "history_frames": 4,
            "scenarios": 1,
            "occurrence_contract": {
                "index": "current_token",
                "order": "oldest_to_current",
                "frames_per_scenario": 4,
                "sequence_id_field": (
                    "occurrences[current_token][i].sequence_id"
                ),
            },
            "model": {
                "sampling_mode": "prior_mean",
                "sampling_is_stochastic": False,
                "config": {
                    "num_detection_classes": 7,
                    "num_map_classes": 3,
                    "detection_anchor_dim": 11,
                    "map_line_dim": 40,
                    "num_output_detections": 50,
                    "num_output_maps": 10,
                },
            },
        },
    }


class GeometryFileTest(unittest.TestCase):
    def write(self, payload):
        handle = tempfile.NamedTemporaryFile(suffix=".pkl")
        pickle.dump(payload, handle)
        handle.flush()
        return handle

    def test_routes_validated_artifact_by_token(self):
        handle = self.write(artifact())
        with open(handle.name, "rb") as source:
            digest = hashlib.sha256(source.read()).hexdigest()
        producer = GeometryFileProducer(
            handle.name, "cpu", {"a"}, "infos-hash", digest
        )
        geometry = producer({"_frame_token": "a"})
        self.assertEqual(geometry["anchor_bbox"].shape, (1, 50, 11))
        self.assertEqual(geometry["map_anchor"].shape, (1, 10, 40))
        self.assertEqual(producer.metadata()["calls"], 1)
        handle.close()

    def test_routes_schema_v2_by_scenario_occurrence(self):
        handle = self.write(occurrence_artifact())
        producer = GeometryFileProducer(
            handle.name,
            "cpu",
            required_tokens={"h0", "h1", "h2", "current"},
            infos_sha256="infos-hash",
            required_occurrences={
                "current": ("h0", "h1", "h2", "current")
            },
        )
        for frame_index, token in enumerate(("h0", "h1", "h2", "current")):
            geometry = producer(
                {
                    "_frame_token": token,
                    "_geometry_scenario_token": "current",
                    "_geometry_frame_index": frame_index,
                }
            )
            self.assertEqual(geometry["anchor_bbox"].shape, (1, 50, 11))
            self.assertEqual(geometry["anchor_bbox"][0, 0, 0], frame_index)
        metadata = producer.metadata()
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["calls"], 4)
        self.assertEqual(metadata["unique_occurrences_used"], 4)
        handle.close()

    def test_occurrence_file_mode_requires_deterministic_schema_v2(self):
        required = {"current": ("h0", "h1", "h2", "current")}
        handle = self.write(occurrence_artifact())
        producer = GeometryFileProducer(
            handle.name,
            "cpu",
            required_tokens={"h0", "h1", "h2", "current"},
            infos_sha256="infos-hash",
            required_occurrences=required,
            require_occurrence_file=True,
        )
        self.assertEqual(producer.schema_version, 2)
        handle.close()

        handle = self.write(artifact())
        with self.assertRaisesRegex(ValueError, "schema version 2"):
            GeometryFileProducer(
                handle.name, "cpu", require_occurrence_file=True
            )
        handle.close()

        payload = occurrence_artifact()
        payload["metadata"]["model"]["sampling_is_stochastic"] = True
        handle = self.write(payload)
        with self.assertRaisesRegex(ValueError, "model contract"):
            GeometryFileProducer(
                handle.name,
                "cpu",
                required_occurrences=required,
                require_occurrence_file=True,
            )
        handle.close()

        payload = occurrence_artifact()
        payload["occurrences"]["current"][0]["geometry"]["map_anchor"] = (
            np.zeros((10, 40), dtype=np.float32)
        )
        handle = self.write(payload)
        with self.assertRaisesRegex(ValueError, "map_anchor is empty"):
            GeometryFileProducer(
                handle.name,
                "cpu",
                required_occurrences=required,
                require_occurrence_file=True,
            )
        handle.close()

    def test_invalid_schema_v2_occurrences_fail_closed(self):
        required = {"current": ("h0", "h1", "h2", "current")}
        payload = occurrence_artifact()
        payload["occurrences"]["current"].pop()
        handle = self.write(payload)
        with self.assertRaisesRegex(ValueError, "four records"):
            GeometryFileProducer(
                handle.name, "cpu", required_occurrences=required
            )
        handle.close()

        payload = occurrence_artifact()
        payload["occurrences"]["current"].reverse()
        handle = self.write(payload)
        with self.assertRaisesRegex(ValueError, "not oldest-to-current"):
            GeometryFileProducer(
                handle.name, "cpu", required_occurrences=required
            )
        handle.close()

        payload = occurrence_artifact()
        payload["occurrences"]["current"][1]["sequence_id"] = "other"
        handle = self.write(payload)
        with self.assertRaisesRegex(ValueError, "multiple sequences"):
            GeometryFileProducer(
                handle.name, "cpu", required_occurrences=required
            )
        handle.close()

        handle = self.write(occurrence_artifact())
        producer = GeometryFileProducer(
            handle.name, "cpu", required_occurrences=required
        )
        with self.assertRaisesRegex(ValueError, "frame token differs"):
            producer(
                {
                    "_frame_token": "wrong",
                    "_geometry_scenario_token": "current",
                    "_geometry_frame_index": 0,
                }
            )
        with self.assertRaisesRegex(ValueError, "frame index"):
            producer(
                {
                    "_frame_token": "h0",
                    "_geometry_scenario_token": "current",
                    "_geometry_frame_index": 4,
                }
            )
        handle.close()

    def test_optional_class_fields_are_validated_but_not_forwarded(self):
        payload = artifact()
        payload["geometry"]["a"].update(
            {
                "detection_logits": np.zeros((50, 7), dtype=np.float32),
                "map_logits": np.zeros((10, 3), dtype=np.float32),
                "detection_scores": np.full(50, 0.5, dtype=np.float32),
                "map_scores": np.full(10, 0.5, dtype=np.float32),
            }
        )
        handle = self.write(payload)
        producer = GeometryFileProducer(handle.name, "cpu", {"a"}, "infos-hash")
        geometry = producer({"_frame_token": "a"})
        self.assertEqual(
            set(geometry),
            {"anchor_bbox", "map_anchor", "detection_valid", "map_valid"},
        )
        handle.close()

        payload = artifact()
        payload["geometry"]["a"]["detection_logits"] = np.zeros(
            (50, 6), dtype=np.float32
        )
        handle = self.write(payload)
        with self.assertRaisesRegex(ValueError, "detection_logits"):
            GeometryFileProducer(handle.name, "cpu")
        handle.close()

        payload = artifact()
        payload["geometry"]["a"]["unknown_field"] = np.zeros(1, dtype=np.float32)
        handle = self.write(payload)
        with self.assertRaisesRegex(KeyError, "incorrect fields"):
            GeometryFileProducer(handle.name, "cpu")
        handle.close()

    def test_invalid_artifacts_fail_closed(self):
        payload = artifact()
        payload["geometry"]["a"]["anchor_bbox"][0, 0] = np.nan
        handle = self.write(payload)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            GeometryFileProducer(handle.name, "cpu")
        handle.close()

        handle = self.write(artifact())
        with self.assertRaisesRegex(ValueError, "token mismatch"):
            GeometryFileProducer(handle.name, "cpu", {"a", "b"})
        with self.assertRaisesRegex(ValueError, "different infos"):
            GeometryFileProducer(handle.name, "cpu", {"a"}, "other")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            GeometryFileProducer(
                handle.name, "cpu", {"a"}, "infos-hash", "wrong"
            )
        handle.close()

    def test_teacher_identity_is_pinned(self):
        handle = self.write(artifact())
        producer = GeometryFileProducer(
            handle.name,
            "cpu",
            {"a"},
            "infos-hash",
            None,
            "a" * 64,
            "b" * 64,
        )
        self.assertEqual(producer.metadata()["calls"], 0)
        handle.close()

        handle = self.write(artifact())
        with self.assertRaisesRegex(ValueError, "different teacher"):
            GeometryFileProducer(
                handle.name,
                "cpu",
                {"a"},
                "infos-hash",
                None,
                "c" * 64,
                "b" * 64,
            )
        handle.close()

        handle = self.write(artifact())
        with self.assertRaisesRegex(ValueError, "both.*teacher"):
            GeometryFileProducer(
                handle.name,
                "cpu",
                {"a"},
                "infos-hash",
                None,
                "a" * 64,
            )
        handle.close()

    def test_producer_checkpoint_identity_is_pinned(self):
        handle = self.write(artifact())
        producer = GeometryFileProducer(
            handle.name,
            "cpu",
            expected_producer_checkpoint_sha256="c" * 64,
        )
        self.assertEqual(producer.metadata()["calls"], 0)
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            GeometryFileProducer(
                handle.name,
                "cpu",
                expected_producer_checkpoint_sha256="d" * 64,
            )
        handle.close()


if __name__ == "__main__":
    unittest.main()
