"""Token- or scenario-occurrence-indexed EMPERROR geometry producer."""

import hashlib
import pickle
from collections.abc import Mapping

import numpy as np
import torch


class GeometryFileProducer:
    def __init__(
        self,
        path,
        device,
        required_tokens=None,
        infos_sha256=None,
        expected_sha256=None,
        expected_teacher_config_sha256=None,
        expected_teacher_checkpoint_sha256=None,
        expected_producer_checkpoint_sha256=None,
        required_occurrences=None,
        require_occurrence_file=False,
    ):
        self.path = path
        self.device = torch.device(device)
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        self.schema_version = payload.get("schema_version")
        self.geometry = None
        self.occurrences = None
        if self.schema_version == 1:
            self._load_token_geometry(payload, required_tokens)
        elif self.schema_version == 2:
            self._load_occurrence_geometry(
                payload, required_tokens, required_occurrences
            )
        else:
            raise ValueError("unsupported EMPERROR geometry artifact schema")
        self.source_metadata = payload.get("metadata", {})
        if require_occurrence_file:
            self._validate_occurrence_file()
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        self.sha256 = digest.hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("EMPERROR geometry artifact SHA256 mismatch")
        if (
            infos_sha256 is not None
            and self.source_metadata.get("infos_sha256") != infos_sha256
        ):
            raise ValueError("EMPERROR geometry artifact was built from different infos")
        expected_teacher = (
            expected_teacher_config_sha256,
            expected_teacher_checkpoint_sha256,
        )
        if any(expected_teacher) and not all(expected_teacher):
            raise ValueError("both EMPERROR teacher SHA256 values are required")
        if all(expected_teacher):
            model = self.source_metadata.get("model", {})
            checkpoint = model.get("checkpoint", {}) if isinstance(model, Mapping) else {}
            contract = (
                checkpoint.get("manifest_contract", {})
                if isinstance(checkpoint, Mapping)
                else {}
            )
            actual_teacher = (
                contract.get("config_sha256")
                if isinstance(contract, Mapping)
                else None,
                contract.get("checkpoint_sha256")
                if isinstance(contract, Mapping)
                else None,
            )
            if actual_teacher != expected_teacher:
                raise ValueError(
                    "EMPERROR geometry artifact was trained on a different teacher"
                )
        if expected_producer_checkpoint_sha256 is not None:
            model = self.source_metadata.get("model", {})
            checkpoint = model.get("checkpoint", {}) if isinstance(model, Mapping) else {}
            actual_producer = (
                checkpoint.get("sha256") if isinstance(checkpoint, Mapping) else None
            )
            if actual_producer != expected_producer_checkpoint_sha256:
                raise ValueError(
                    "EMPERROR geometry artifact was produced by a different checkpoint"
                )
        self.calls = 0
        self.used_tokens = set()
        self.used_occurrences = set()

    def _load_token_geometry(self, payload, required_tokens):
        self.geometry = payload.get("geometry")
        if not isinstance(self.geometry, Mapping) or not self.geometry:
            raise ValueError("EMPERROR geometry artifact is empty")
        if required_tokens is not None and set(self.geometry) != set(required_tokens):
            missing = set(required_tokens) - set(self.geometry)
            extra = set(self.geometry) - set(required_tokens)
            raise ValueError(
                f"EMPERROR geometry token mismatch: {len(missing)} missing, "
                f"{len(extra)} extra"
            )
        for token, values in self.geometry.items():
            self._validate(token, values)

    def _validate_occurrence_file(self):
        if self.schema_version != 2:
            raise ValueError("occurrence geometry source requires schema version 2")
        model = self.source_metadata.get("model")
        config = model.get("config") if isinstance(model, Mapping) else None
        if (
            not isinstance(model, Mapping)
            or model.get("sampling_mode") != "prior_mean"
            or model.get("sampling_is_stochastic") is not False
            or not isinstance(config, Mapping)
            or (
                config.get("num_detection_classes"),
                config.get("num_map_classes"),
                config.get("detection_anchor_dim"),
                config.get("map_line_dim"),
                config.get("num_output_detections"),
                config.get("num_output_maps"),
            )
            != (7, 3, 11, 40, 50, 10)
        ):
            raise ValueError("occurrence geometry model contract is invalid")
        if (
            self.source_metadata.get("scenarios") != len(self.occurrences)
            or self.source_metadata.get("history_frames")
            != 4 * len(self.occurrences)
        ):
            raise ValueError("occurrence geometry metadata count mismatch")

        sequence_ids = set()
        for scenario_token, records in self.occurrences.items():
            sequence_id = records[0]["sequence_id"]
            if sequence_id in sequence_ids:
                raise ValueError("duplicate occurrence geometry sequence_id")
            sequence_ids.add(sequence_id)
            for frame_index, record in enumerate(records):
                maps = np.asarray(record["geometry"]["map_anchor"]).reshape(
                    10, 20, 2
                )
                if np.any(np.ptp(maps, axis=1).max(axis=1) == 0):
                    raise ValueError(
                        "occurrence geometry map_anchor is empty for "
                        f"{scenario_token}[{frame_index}]"
                    )

    def _load_occurrence_geometry(
        self, payload, required_tokens, required_occurrences
    ):
        if payload.get("geometry_index") != "scenario_occurrence":
            raise ValueError("schema-v2 geometry_index must be scenario_occurrence")
        expected_contract = {
            "index": "current_token",
            "order": "oldest_to_current",
            "frames_per_scenario": 4,
            "sequence_id_field": "occurrences[current_token][i].sequence_id",
        }
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get(
            "occurrence_contract"
        ) != expected_contract:
            raise ValueError("schema-v2 occurrence contract is invalid")
        self.occurrences = payload.get("occurrences")
        if not isinstance(self.occurrences, Mapping) or not self.occurrences:
            raise ValueError("EMPERROR occurrence geometry artifact is empty")
        if not isinstance(required_occurrences, Mapping):
            raise ValueError("schema-v2 geometry requires replay occurrences")
        if set(self.occurrences) != set(required_occurrences):
            missing = set(required_occurrences) - set(self.occurrences)
            extra = set(self.occurrences) - set(required_occurrences)
            raise ValueError(
                f"EMPERROR geometry occurrence mismatch: {len(missing)} missing, "
                f"{len(extra)} extra"
            )

        occurrence_tokens = set()
        for scenario_token, records in self.occurrences.items():
            if not isinstance(scenario_token, str) or not scenario_token:
                raise ValueError("geometry occurrence key must be a scenario token")
            expected_tokens = tuple(required_occurrences[scenario_token])
            if len(expected_tokens) != 4 or expected_tokens[-1] != scenario_token:
                raise ValueError(
                    f"replay occurrence {scenario_token} must contain four "
                    "oldest-to-current tokens"
                )
            if not isinstance(records, list) or len(records) != 4:
                raise ValueError(
                    f"geometry occurrence {scenario_token} must contain four records"
                )
            actual_tokens = []
            sequence_ids = set()
            for frame_index, record in enumerate(records):
                label = f"{scenario_token}[{frame_index}]"
                if not isinstance(record, Mapping) or set(record) != {
                    "sequence_id",
                    "token",
                    "geometry",
                }:
                    raise ValueError(
                        f"geometry occurrence record {label} has incorrect fields"
                    )
                if not isinstance(record["sequence_id"], str) or not record[
                    "sequence_id"
                ]:
                    raise ValueError(
                        f"geometry occurrence record {label} has invalid sequence_id"
                    )
                sequence_ids.add(record["sequence_id"])
                token = record["token"]
                if not isinstance(token, str) or not token:
                    raise ValueError(
                        f"geometry occurrence record {label} has invalid token"
                    )
                self._validate(label, record["geometry"])
                actual_tokens.append(token)
                occurrence_tokens.add(token)
            if len(sequence_ids) != 1:
                raise ValueError(
                    f"geometry occurrence {scenario_token} spans multiple sequences"
                )
            if tuple(actual_tokens) != expected_tokens:
                raise ValueError(
                    f"geometry occurrence {scenario_token} is not oldest-to-current"
                )
        if required_tokens is not None and occurrence_tokens != set(required_tokens):
            missing = set(required_tokens) - occurrence_tokens
            extra = occurrence_tokens - set(required_tokens)
            raise ValueError(
                f"EMPERROR geometry token mismatch: {len(missing)} missing, "
                f"{len(extra)} extra"
            )

    # The planner consumes only these four fields; newer artifacts may also
    # carry class logits and max-sigmoid confidences for inspection.
    PLANNER_FIELDS = {
        "anchor_bbox": ((50, 11), np.float32),
        "map_anchor": ((10, 40), np.float32),
        "detection_valid": ((50,), np.bool_),
        "map_valid": ((10,), np.bool_),
    }
    OPTIONAL_FIELDS = {
        "detection_logits": ((50, 7), np.float32),
        "map_logits": ((10, 3), np.float32),
        "detection_scores": ((50,), np.float32),
        "map_scores": ((10,), np.float32),
    }

    @classmethod
    def _validate(cls, token, values):
        if not isinstance(values, Mapping):
            raise TypeError(f"geometry for {token} must be a mapping")
        required = set(cls.PLANNER_FIELDS)
        allowed = required | set(cls.OPTIONAL_FIELDS)
        if not required <= set(values) or not set(values) <= allowed:
            raise KeyError(f"geometry for {token} has incorrect fields")
        expected = {**cls.PLANNER_FIELDS, **cls.OPTIONAL_FIELDS}
        for name in values:
            shape, dtype = expected[name]
            value = np.asarray(values[name])
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(
                    f"geometry {name} for {token} must be {shape} {dtype}"
                )
            if dtype == np.bool_:
                if not value.all():
                    raise ValueError(f"geometry {name} for {token} contains padding")
            elif not np.isfinite(value).all():
                raise ValueError(f"geometry {name} for {token} is non-finite")

    def __call__(self, frame):
        token = frame.get("_frame_token")
        if self.schema_version == 1:
            if token not in self.geometry:
                raise KeyError(f"no EMPERROR geometry for frame {token}")
            values = self.geometry[token]
        else:
            scenario_token = frame.get("_geometry_scenario_token")
            frame_index = frame.get("_geometry_frame_index")
            if scenario_token not in self.occurrences:
                raise KeyError(
                    f"no EMPERROR geometry occurrence for {scenario_token}"
                )
            if (
                not isinstance(frame_index, int)
                or isinstance(frame_index, bool)
                or not 0 <= frame_index < 4
            ):
                raise ValueError("geometry frame index must be an integer in [0, 4)")
            record = self.occurrences[scenario_token][frame_index]
            if record["token"] != token:
                raise ValueError("geometry occurrence frame token differs from replay")
            values = record["geometry"]
            self.used_occurrences.add((scenario_token, frame_index))
        self.calls += 1
        self.used_tokens.add(token)
        return {
            name: torch.as_tensor(values[name], device=self.device).unsqueeze(0)
            for name in self.PLANNER_FIELDS
        }

    def metadata(self):
        return {
            "path": self.path,
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "calls": self.calls,
            "unique_tokens_used": len(self.used_tokens),
            "tokens_used": sorted(self.used_tokens),
            "unique_occurrences_used": len(self.used_occurrences),
            "source": self.source_metadata,
        }
