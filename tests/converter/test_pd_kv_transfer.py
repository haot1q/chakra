"""Strict actual-endpoint P/D KV converter tests."""

import hashlib
import json
from pathlib import Path

import pytest

from chakra.src.converter.pd_kv_events import convert_pd_kv_sidecar


def _sidecar(tmp_path: Path) -> Path:
    descriptor = {
        "run_id": "run-1", "attempt_id": "attempt-1", "sequence": 0,
        "transfer_id": "pd-kv:abc", "request_id": "7",
        "source_instance_id": 0, "destination_instance_id": 2,
        "compatibility_profile": "homogeneous_pd_kv_v1",
        "layout_digest": "sha256:" + "a" * 64,
        "initialized_tokens": 17, "initialized_bytes_per_rank": 100,
        "allocated_bytes_per_rank": 128, "charged_bytes_per_rank": 128,
        "charged_bytes_aggregate": 256,
        "blocks": [{
            "request_id": "7", "logical_block_index": 0, "token_start": 0,
            "token_capacity": 16, "initialized_tokens": 16,
            "allocated_bytes": 128, "geometry_digest": "sha256:" + "a" * 64,
            "layout_provenance": "analytical_estimate",
        }],
        "rank_pairs": [
            {"source_rank": 3, "destination_rank": 11, "tp_rank": 0,
             "pp_stage": 0, "layer_start": 0, "layer_end": 4,
             "charged_bytes": 128, "tag": 1_000_000_000},
            {"source_rank": 4, "destination_rank": 12, "tp_rank": 1,
             "pp_stage": 0, "layer_start": 0, "layer_end": 4,
             "charged_bytes": 128, "tag": 1_000_000_001},
        ],
        "artifact_identity": {"outer_commit": "a" * 40},
        "schema_version": "pd-kv-transfer-v1",
        "transfer_basis": "full_framework_block_v1",
        "transport": "astra_analytical_p2p_v1",
        "evidence_tier": "mechanics_uncalibrated",
    }
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        "descriptor": descriptor,
        "descriptor_digest": "sha256:" + hashlib.sha256(encoded).hexdigest(),
    }
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_converter_emits_only_actual_source_and_destination_ranks(tmp_path) -> None:
    paths = convert_pd_kv_sidecar(str(_sidecar(tmp_path)), str(tmp_path / "pd"))
    assert {path.name for path in paths} == {
        "pd.3.et", "pd.4.et", "pd.11.et", "pd.12.et"
    }


def test_converter_rejects_digest_tampering(tmp_path) -> None:
    path = _sidecar(tmp_path)
    payload = json.loads(path.read_text())
    payload["descriptor"]["charged_bytes_aggregate"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        convert_pd_kv_sidecar(str(path), str(tmp_path / "pd"))


def test_converter_rejects_unknown_block_fields(tmp_path) -> None:
    path = _sidecar(tmp_path)
    payload = json.loads(path.read_text())
    payload["descriptor"]["blocks"][0]["unknown"] = True
    encoded = json.dumps(
        payload["descriptor"], sort_keys=True, separators=(",", ":")
    ).encode()
    payload["descriptor_digest"] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="block fields"):
        convert_pd_kv_sidecar(str(path), str(tmp_path / "pd"))
