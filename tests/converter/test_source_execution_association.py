"""Source references annotate existing traffic without creating a second bill."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from chakra.schema.protobuf.et_def_pb2 import AttributeProto, GlobalMetadata, Node, MEM_LOAD_NODE, MEM_STORE_NODE
from chakra.src.converter.llm_converter import LLMConverter
from chakra.src.converter.operator_io import OperatorIO
from chakra.src.converter.source_execution_association import SourceExecutionAssociation, parse_limits_argument
from chakra.src.converter.tier_manifest import TierManifest, canonical_manifest_digest
from chakra.src.third_party.utils.protolib import decodeMessage


def _case(tmp_path: Path):
    manifest = {
        "schema_version": "memory-tier-runtime-v1", "id_mode": "native",
        "tiers": [{"tier_name": name, "tier_id": 16 + i,
                   "backend_kind": "analytical", "scope": "instance", "pool_key": name,
                   "devices": [{"device_id": 0, "capacity_bytes": 1024}],
                   "num_devices": 1, "mem_bw_gbps": 1000, "mem_latency_ns": 100}
                  for i, name in enumerate(("hbm", "lpddr"))],
    }
    manifest["manifest_digest"] = canonical_manifest_digest(manifest)
    manifest_path = tmp_path / "tiers.json"
    manifest_path.write_text(json.dumps(manifest))
    trace = tmp_path / "input.txt"
    trace.write_text(
        "COLOCATED model_parallel_NPU_group: 1 trace_schema: llm-tier-v1 "
        f"tier_manifest_digest: {manifest['manifest_digest']} operator_io: ucie_transport_v1\n"
        "3\nignored\n"
        "entry 1 hbm:0 16 hbm:0 0 hbm:0 16 NONE 0 NONE -\n"
        "middle 2 hbm:0 16 hbm:0 0 hbm:0 16 NONE 0 NONE -\n"
        "exit 1 hbm:0 16 hbm:0 0 hbm:0 16 NONE 0 NONE -\n"
    )
    trace_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
    operator = {
        "schema_version": "operator-io-v2", "accounting": "ucie_transport_v1",
        "tier_manifest_digest": manifest["manifest_digest"], "trace_sha256": trace_sha,
        "rows": [{"rank": 0, "row": 1,
                  "reads": [{"location": "hbm:0", "bytes": 16},
                            {"location": "lpddr:0", "bytes": 16}],
                  "writes": [{"location": "hbm:0", "bytes": 8}]}],
    }
    operator_path = tmp_path / "operator.json"
    operator_path.write_text(json.dumps(operator))
    rank = {
        "execution": {"run_id": "run", "source_provenance_digest": "a" * 64,
                      "execution_iteration": "1", "rank": "0"},
        "invocations": [{"ref": "0", "batch_id": "batch-7", "invocation_id": "call"}],
        "backings": [{"ref": "0", "backing_identity": {
            "profile_id": "native", "owner_scope": "rank0", "allocation_id": "buffer"},
            "extent_bytes": "64", "catalog_generation": None}],
        "operands": [], "pieces": [], "groups": [], "source_unassociated": [],
    }
    # Two repeated reads are distinct occurrences; each crosses two endpoints.
    for i, (operation, count, length) in enumerate((("read", 2, 16), ("write", 1, 8))):
        rank["operands"].append({
            "ref": str(i), "invocation_ref": "0", "source_boundary": "original.middle",
            "execution_layer": "0", "role": "activation", "origin": operation + "s",
            "operation": operation, "ordinal_start": "0", "backing_ref": "0",
            "offset_bytes": "0", "length_bytes": str(length), "stride_bytes": "0", "count": str(count),
        })
        for occurrence in range(count):
            spans = ((0, 8, "hbm:0"), (8, 8, "lpddr:0")) if operation == "read" else ((0, 8, "hbm:0"),)
            for offset, size, location in spans:
                rank["pieces"].append({
                    "ref": str(len(rank["pieces"])), "operand_ref": str(i),
                    "run_index": str(occurrence), "offset_bytes": str(offset),
                    "length_bytes": str(size), "location": location,
                    "authorization": {"kind": "workspace", "owner": "original-owner",
                                      "page_id": None, "residency_version": None},
                    "volume_ordinal": None,
                })
    for operation, segment, location, size, pieces in (
            ("read", 0, "hbm:0", 16, ["0", "2"]),
            ("read", 1, "lpddr:0", 16, ["1", "3"]),
            ("write", 0, "hbm:0", 8, ["4"])):
        rank["groups"].append({"ref": f"r0/row1/{operation}/s{segment}", "row": "1",
                               "operation": operation, "segment_index": str(segment),
                               "location": location, "bytes": str(size), "piece_refs": pieces})
    document = {
        "schema": "source-execution-association-v1", "run_id": "run",
        "source_provenance_digest": "a" * 64, "source_profile": "native-source-prefix-v1",
        "source_selection_sha256": "b" * 64, "trace_sha256": trace_sha,
        "operator_io_sha256": hashlib.sha256(operator_path.read_bytes()).hexdigest(),
        "tier_manifest_digest": manifest["manifest_digest"], "ranks": [rank],
    }
    association = tmp_path / "association.json"
    association.write_text(json.dumps(document))
    return trace, manifest_path, operator_path, association, document


def _reader(case, **kwargs):
    trace, manifest_path, operator_path, association, _ = case
    manifest = TierManifest(manifest_path)
    return SourceExecutionAssociation(
        association, operator_io=OperatorIO(str(operator_path), manifest),
        manifest=manifest, trace_path=trace, **kwargs)


def _convert(case, output, *, associate):
    trace, manifest, operator, association, _ = case
    converter = LLMConverter(str(trace), str(output), 1, tier_manifest=str(manifest))
    converter.configure_operator_io(str(operator))
    if associate:
        converter.configure_source_association(str(association))
    result = []
    for path in converter.convert():
        with path.open("rb") as stream:
            assert decodeMessage(stream, GlobalMetadata())
            while True:
                node = Node()
                if not decodeMessage(stream, node):
                    break
                result.append(node)
    return converter, result


def test_real_converter_preserves_every_original_node_and_byte_bill(tmp_path):
    case = _case(tmp_path)
    _, original = _convert(case, tmp_path / "legacy", associate=False)
    converter, associated = _convert(case, tmp_path / "associated", associate=True)
    refs = []
    for before, after in zip(original, associated, strict=True):
        refs.extend(a.string_val for a in after.attr if a.name == "joint_access_ref")
        clean = Node()
        clean.CopyFrom(after)
        attributes = [a for a in clean.attr if a.name != "joint_access_ref"]
        del clean.attr[:]
        clean.attr.extend(attributes)
        assert clean == before
    assert sorted(refs) == ["r0/row1/read/s0", "r0/row1/read/s1", "r0/row1/write/s0"]
    assert converter.source_association.input_sha256 == hashlib.sha256(case[3].read_bytes()).hexdigest()


@pytest.mark.parametrize("fault", [
    "version", "trace_hash", "operator_hash", "manifest", "run", "rank", "iteration_type",
    "unknown_field", "duplicate_rank", "duplicate_piece", "duplicate_group", "wrong_bill",
    "wrong_location", "wrong_operation", "gap", "overlap", "run_overflow", "missing_occurrence",
    "missing_reference", "partial_occurrence", "bad_owner", "bad_page", "noncanonical", "bad_stride",
])
def test_semantic_corruption_fails_before_emission(tmp_path, fault):
    case = _case(tmp_path)
    doc = case[4]
    rank = doc["ranks"][0]
    mutations = {
        "version": (doc, "schema", "unknown"),
        "trace_hash": (doc, "trace_sha256", "c" * 64),
        "operator_hash": (doc, "operator_io_sha256", "c" * 64),
        "manifest": (doc, "tier_manifest_digest", "c" * 64),
        "run": (rank["execution"], "run_id", "another"),
        "rank": (rank["execution"], "rank", "1"),
        "iteration_type": (rank["execution"], "execution_iteration", True),
        "unknown_field": (rank, "extra", "value"),
        "duplicate_rank": (doc, "ranks", [rank, deepcopy(rank)]),
        "duplicate_piece": (rank["groups"][0], "piece_refs", ["0", "2", "0"]),
        "duplicate_group": (rank, "groups", [*rank["groups"], deepcopy(rank["groups"][0])]),
        "wrong_bill": (rank["groups"][0], "bytes", "15"),
        "wrong_location": (rank["pieces"][0], "location", "lpddr:0"),
        "wrong_operation": (rank["groups"][0], "operation", "write"),
        "gap": (rank["pieces"][0], "length_bytes", "7"),
        "overlap": (rank["pieces"][1], "offset_bytes", "7"),
        "run_overflow": (rank["operands"][0], "count", str(2**64 - 1)),
        "missing_occurrence": (rank["operands"][0], "count", "3"),
        "missing_reference": (rank["pieces"][0], "operand_ref", "99"),
        "bad_owner": (rank["pieces"][0]["authorization"], "owner", ["one", "two"]),
        "bad_page": (rank["pieces"][0]["authorization"], "kind", "page"),
        "noncanonical": (rank["operands"][0], "count", "02"),
        "bad_stride": (rank["operands"][0], "stride_bytes", "-0"),
    }
    if fault == "partial_occurrence":
        rank["groups"].pop(1)
        rank["source_unassociated"] = [{"piece_ref": p, "reason": "outside_selected_segment"} for p in ("1", "3")]
    else:
        owner, key, value = mutations[fault]
        owner[key] = value
    case[3].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="association"):
        _reader(case)


@pytest.mark.parametrize("raw", [b'{"schema":"x","schema":"x"}', b"[" * 33 + b"]" * 33,
                                  b'{"bad":NaN}', b'{"bad":"\\u0000"}', b'{"bad":"\\ud800"}', b'\xff'])
def test_malformed_json_is_rejected(tmp_path, raw):
    case = _case(tmp_path)
    case[3].write_bytes(raw)
    with pytest.raises(ValueError):
        _reader(case)


def test_host_byte_limit_and_symlink_are_rejected(tmp_path):
    case = _case(tmp_path)
    with pytest.raises(ValueError, match="association"):
        _reader(case, limits={"association_bytes": 100})
    target = case[3].with_suffix(".target")
    case[3].rename(target)
    case[3].symlink_to(target)
    with pytest.raises(ValueError, match="association"):
        _reader(case)


def test_an_omitted_whole_occurrence_remains_explicit(tmp_path):
    case = _case(tmp_path)
    rank = case[4]["ranks"][0]
    rank["groups"] = [g for g in rank["groups"] if g["operation"] == "write"]
    rank["source_unassociated"] = [{"piece_ref": str(i), "reason": "no_existing_mem_lowering"} for i in range(4)]
    case[3].write_text(json.dumps(case[4]))
    reader = _reader(case)
    assert len(reader.ranks[0]["source_unassociated"]) == 4


def test_attach_rejects_compute_or_duplicate_emission(tmp_path):
    reader = _reader(_case(tmp_path))
    node = Node(id=1, type=MEM_LOAD_NODE)
    node.attr.extend([AttributeProto(name="tensor_size", uint64_val=16),
                      AttributeProto(name="tensor_loc", uint32_val=16),
                      AttributeProto(name="tensor_device", uint32_val=0)])
    reader.attach(node, 0, 1, "read", 0)
    with pytest.raises(ValueError, match="association"):
        reader.attach(node, 0, 1, "read", 0)
    with pytest.raises(ValueError, match="association"):
        reader.validate_emission()
    with pytest.raises(ValueError, match="association"):
        reader.attach(Node(id=2, type=MEM_STORE_NODE), 0, 1, "read", 1)


@pytest.mark.parametrize("argument", ["null", "[]", '{"pieces":true}', '{"pieces":1.5}',
                                      '{"pieces":0}', '{"pieces":"01"}', '{"pieces":1048577}',
                                      '{"pieces":1,"pieces":2}', '{"wrong":1}'])
def test_cli_limits_are_a_strict_bounded_host_object(argument):
    with pytest.raises(ValueError, match="association"):
        parse_limits_argument(argument)


@pytest.mark.parametrize("stride", ["16", "-16"])
def test_original_positive_and_negative_stride_are_retained(tmp_path, stride):
    case = _case(tmp_path)
    rank = case[4]["ranks"][0]
    operand = rank["operands"][0]
    operand["stride_bytes"] = stride
    operand["offset_bytes"] = "16" if stride == "-16" else "0"
    for piece in rank["pieces"][:4]:
        base = int(operand["offset_bytes"]) + int(piece["run_index"]) * int(stride)
        piece["offset_bytes"] = str(base + (0 if piece["location"] == "hbm:0" else 8))
    case[3].write_text(json.dumps(case[4]))
    assert _reader(case).ranks[0]["operands"][0]["stride_bytes"] == stride


def test_declared_smaller_record_budget_is_enforced(tmp_path):
    case = _case(tmp_path)
    with pytest.raises(ValueError, match="association"):
        _reader(case, limits={"table_records": "3"})
    with pytest.raises(ValueError, match="association"):
        _reader(case, limits={"pieces": "4"})


def test_original_page_authorization_and_volume_ordinal_are_retained(tmp_path):
    case = _case(tmp_path)
    rank = case[4]["ranks"][0]
    rank["operands"][0]["origin"] = "captured_page_reads"
    rank["backings"][0]["catalog_generation"] = "process"
    for piece in rank["pieces"][:4]:
        piece["authorization"] = {"kind": "page", "owner": ["pool", "layer", "K"],
                                  "page_id": "original-page", "residency_version": "0"}
        piece["volume_ordinal"] = "17"
    case[3].write_text(json.dumps(case[4]))
    assert _reader(case).ranks[0]["pieces"][0]["volume_ordinal"] == "17"


def test_real_cli_applies_association_and_rejects_a_lower_budget(tmp_path):
    case = _case(tmp_path)
    trace, manifest, operator, association, _ = case
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, PYTHONPATH=str(root.parent))
    command = [sys.executable, "-m", "chakra.src.converter.converter", "--log-filename",
               str(tmp_path / "convert.log"), "LLM", "--input", str(trace),
               "--output", str(tmp_path / "cli"), "--num-npus", "1", "--tier-manifest",
               str(manifest), "--operator-io", str(operator), "--source-execution-association",
               str(association), "--source-association-limits", '{"pieces":"5"}']
    result = subprocess.run(command, env=environment, cwd=root,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "cli.0.et").is_file()
    command[-1] = '{"pieces":"4"}'
    result = subprocess.run(command, env=environment, cwd=root,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "association" in result.stderr


@pytest.mark.parametrize("fault", ["missing_volume_ordinal", "non_page_authorization"])
def test_captured_page_read_requires_its_original_volume_authorization(tmp_path, fault):
    case = _case(tmp_path)
    rank = case[4]["ranks"][0]
    rank["operands"][0]["origin"] = "captured_page_reads"
    rank["backings"][0]["catalog_generation"] = "process"
    for piece in rank["pieces"][:4]:
        piece["authorization"] = {"kind": "page", "owner": ["pool", "layer", "K"],
                                  "page_id": "original-page", "residency_version": "0"}
        piece["volume_ordinal"] = "17"
    if fault == "missing_volume_ordinal":
        rank["pieces"][0]["volume_ordinal"] = None
    else:
        rank["pieces"][0]["authorization"] = {
            "kind": "workspace", "owner": "unrelated", "page_id": None, "residency_version": None}
    case[3].write_text(json.dumps(case[4]))
    with pytest.raises(ValueError, match="Page|volume"):
        _reader(case)


def test_actual_rank_mismatch_rejects_before_expanding_an_unbounded_range(tmp_path, monkeypatch):
    from chakra.src.converter import source_execution_association as module

    reader = _reader(_case(tmp_path))
    def forbid_expansion(values):
        if isinstance(values, range):
            raise AssertionError("rank mismatch expanded a collection before validation")
        return set(values)
    monkeypatch.setattr(module, "set", forbid_expansion, raising=False)
    with pytest.raises(ValueError, match="rank"):
        reader.reset_emission(range(1 << 40))
