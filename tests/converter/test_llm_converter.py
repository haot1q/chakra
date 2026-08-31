from __future__ import annotations

from pathlib import Path
import json

import pytest

from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
from chakra.src.converter.et_validator import ETValidationError, validate_et_group
from chakra.src.converter.llm_converter import LLMConverter
from chakra.src.converter.tier_manifest import canonical_manifest_digest
from chakra.src.third_party.utils.protolib import decodeMessage, encodeMessage


def _layer(
    name: str,
    input_size: int = 16,
    output_size: int = 16,
    comm_type: str = "NONE",
    comm_size: int = 0,
) -> str:
    return (
        f"{name} 1 LOCAL {input_size} LOCAL 16 LOCAL {output_size} "
        f"{comm_type} {comm_size} NONE\n"
    )


def _expert(expert: str) -> str:
    return f"EXPERT {expert} NONE 0\n"


def _dense_rows() -> list[str]:
    return [
        _layer("embedding"),
        _layer("block0_layernorm"),
        _layer("block0_qkv_proj", output_size=24),
        _layer("block0_down_proj"),
        _layer("block1_layernorm"),
        _layer("block1_qkv_proj", output_size=24),
        _layer("block1_down_proj"),
        _layer("lm_head"),
    ]


def _moe_rows() -> list[str]:
    return [
        _layer("embedding"),
        _layer("block0_layernorm"),
        _expert("0"),
        _layer("block0_moe"),
        _expert("END"),
        _layer("block0_down_proj"),
        _layer("block1_layernorm"),
        _layer("block1_qkv_proj", output_size=24),
        _expert("0"),
        _layer("block1_moe"),
        _expert("END"),
        _layer("block1_down_proj"),
        _layer("lm_head"),
    ]


def _pim_rows() -> list[str]:
    return [
        _layer("embedding"),
        _layer("block0_layernorm"),
        "PIM 0\n",
        _layer("block0_pim_attention"),
        "PIM END\n",
        _layer("block0_down_proj"),
        _layer("block1_layernorm"),
        "PIM 0\n",
        _layer("block1_pim_attention"),
        "PIM END\n",
        _layer("block1_down_proj"),
        _layer("lm_head"),
    ]


def _prefill_rows() -> list[str]:
    return [
        _layer("embedding"),
        _layer("block0_layernorm"),
        _layer("block0_v_proj", comm_size=16),
        _layer("block0_down_proj"),
        _layer("block1_layernorm"),
        _layer("block1_v_proj", comm_size=16),
        _layer("block1_down_proj"),
        _layer("lm_head"),
    ]


def _write_trace(
    path: Path,
    rows: list[str],
    *,
    pp_size: int = 2,
    boundaries: str | None = None,
    execution_type: str = "COLOCATED",
    header_suffix: str = "",
) -> None:
    header = f"{execution_type}  model_parallel_NPU_group: {pp_size}"
    if boundaries is not None:
        header += f"  pp_stage_boundaries: {boundaries}"
    path.write_text(
        header + header_suffix + f"\n{len(rows)}\nignored table header\n" + "".join(rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("rows", "boundary", "kind"),
    [(_moe_rows(), "8", "EXPERT"), (_pim_rows(), "8", "PIM")],
)
def test_boundary_inside_marker_region_fails_closed(
    tmp_path: Path, rows: list[str], boundary: str, kind: str
) -> None:
    trace = tmp_path / "bad-boundary.txt"
    _write_trace(trace, rows, boundaries=boundary)

    with pytest.raises(ValueError, match=rf"inside {kind} region"):
        LLMConverter(str(trace), str(tmp_path / "llm"), num_npus=2).convert()


def test_unclosed_expert_region_fails_closed(tmp_path: Path) -> None:
    rows = _moe_rows()[:4]
    trace = tmp_path / "unclosed.txt"
    _write_trace(trace, rows, boundaries="1")

    with pytest.raises(ValueError, match="unclosed EXPERT region"):
        LLMConverter(str(trace), str(tmp_path / "llm"), num_npus=2).convert()


def test_nested_marker_region_fails_closed(tmp_path: Path) -> None:
    rows = [
        _layer("embedding"),
        _expert("0"),
        "PIM 0\n",
        _layer("bad_nested_region"),
        "PIM END\n",
        _expert("END"),
        _layer("lm_head"),
    ]
    trace = tmp_path / "nested.txt"
    _write_trace(trace, rows, boundaries="1")

    with pytest.raises(ValueError, match="nested inside EXPERT region"):
        LLMConverter(str(trace), str(tmp_path / "llm"), num_npus=2).convert()


def test_pp1_rejects_unexpected_boundaries(tmp_path: Path) -> None:
    trace = tmp_path / "pp1.txt"
    _write_trace(trace, _dense_rows(), pp_size=1, boundaries="4")

    with pytest.raises(ValueError, match="PP1 Trace must not declare"):
        LLMConverter(str(trace), str(tmp_path / "llm"), num_npus=2).convert()


def test_duplicate_header_key_fails_closed(tmp_path: Path) -> None:
    trace = tmp_path / "duplicate-header.txt"
    _write_trace(
        trace,
        _dense_rows(),
        boundaries="4",
        header_suffix="  pp_stage_boundaries: 5",
    )

    with pytest.raises(ValueError, match="Duplicate Trace header key"):
        LLMConverter(str(trace), str(tmp_path / "llm"), num_npus=2).convert()


def test_layer_state_reset_discards_previous_et_nodes() -> None:
    layers = LLMConverter.__new__(LLMConverter).get_layers(iter(_dense_rows()))
    layers[0].comp_node = object()
    layers[0].comm_node = object()
    layers[0].input_memory_node = object()

    LLMConverter._reset_layer_state(layers)

    assert layers[0].comp_node is None
    assert layers[0].comm_node is None
    assert layers[0].input_memory_node is None


@pytest.mark.parametrize(
    ("rows", "boundary"),
    [(_dense_rows(), "4"), (_moe_rows(), "6"), (_pim_rows(), "6")],
)
def test_pp2_valid_traces_emit_valid_et_groups(
    tmp_path: Path, rows: list[str], boundary: str
) -> None:
    trace = tmp_path / "model.txt"
    _write_trace(trace, rows, boundaries=boundary)

    outputs = LLMConverter(
        str(trace), str(tmp_path / "llm"), num_npus=2
    ).convert()

    report = validate_et_group(outputs)
    assert report.file_count == 2
    assert report.node_count > 0


def test_pp1_remains_backward_compatible(tmp_path: Path) -> None:
    trace = tmp_path / "pp1.txt"
    _write_trace(trace, _dense_rows(), pp_size=1)

    outputs = LLMConverter(
        str(trace), str(tmp_path / "llm"), num_npus=2
    ).convert()

    assert validate_et_group(outputs).file_count == 2


def test_prefill_validates_decode_pair_outputs(tmp_path: Path) -> None:
    trace = tmp_path / "prefill.txt"
    _write_trace(trace, _prefill_rows(), boundaries="4", execution_type="PREFILL")

    outputs = LLMConverter(
        str(trace), str(tmp_path / "llm"), num_npus=2
    ).convert()

    assert validate_et_group(outputs).file_count == 4


def _write_native_manifest(path: Path) -> str:
    payload = {
        "schema_version": "memory-tier-runtime-v1",
        "id_mode": "native",
        "tiers": [
            {
                "tier_name": name,
                "tier_id": 16 + index,
                "backend_kind": "analytical",
                "scope": "instance",
                "pool_key": name,
                "devices": [{"device_id": 0, "capacity_bytes": 1024}],
                "num_devices": 1,
                "mem_bw_gbps": 1000,
                "mem_latency_ns": 100,
            }
            for index, name in enumerate(("hbm", "lpddr", "remote"))
        ],
    }
    payload["manifest_digest"] = canonical_manifest_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload["manifest_digest"]


def _native_layer(name: str, segments: str = "-") -> str:
    weight_size = 0 if segments != "-" else 16
    return (
        f"{name} 1 hbm:0 16 hbm:0 {weight_size} hbm:0 16 "
        f"NONE 0 NONE {segments}\n"
    )


def _read_et(path: Path) -> tuple[GlobalMetadata, list[Node]]:
    metadata = GlobalMetadata()
    nodes = []
    with path.open("rb") as stream:
        assert decodeMessage(stream, metadata)
        while True:
            node = Node()
            if not decodeMessage(stream, node):
                break
            nodes.append(node)
    return metadata, nodes


def _uint_attr(node: Node, name: str) -> int:
    return next(attr.uint32_val for attr in node.attr if attr.name == name)


def _string_attr(node: Node, name: str) -> str:
    return next(attr.string_val for attr in node.attr if attr.name == name)


def _write_movement_events(
    path: Path,
    digest: str,
    *,
    selected_path: str = "base_die_local",
) -> None:
    resources = (
        ["lpddr_read", "base_die_dma", "local_stack_fabric", "hbm_write"]
        if selected_path == "base_die_local"
        else [
            "lpddr_read",
            "base_to_gpu_link",
            "gpu_dma",
            "gpu_to_base_link",
            "hbm_write",
        ]
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": "memory-events-v1",
                "run_id": "test-run",
                "instance_id": "instance-0",
                "manifest_digest": digest,
                "selected_path": {
                    "id": selected_path,
                    "engine_count": 1,
                    "max_priority_burst": 4,
                    "max_in_flight_page_movements": 1,
                    "resource_ids": resources,
                },
                "events": [
                    {
                        "event_id": "critical-0",
                        "page_id": "page-v1:" + "1" * 64,
                        "transaction_id": "promote-00000001-page",
                        "expected_residency_version": 0,
                        "source_iteration_id": 0,
                        "npu_id": 0,
                        "kind": "page_promote",
                        "phase": "critical_line",
                        "source": {"tier_id": 17, "device_id": 0},
                        "destination": {"tier_id": 16, "device_id": 0},
                        "bytes": 4096,
                        "priority_class": "decode_critical",
                        "depends_on": [],
                        "releases": ["block0_layernorm"],
                    },
                    {
                        "event_id": "background-0",
                        "page_id": "page-v1:" + "1" * 64,
                        "transaction_id": "promote-00000001-page",
                        "expected_residency_version": 0,
                        "source_iteration_id": 0,
                        "npu_id": 0,
                        "kind": "page_promote",
                        "phase": "background_fill",
                        "source": {"tier_id": 17, "device_id": 0},
                        "destination": {"tier_id": 16, "device_id": 0},
                        "bytes": 2 * 1024 * 1024,
                        "priority_class": "background_fill",
                        "depends_on": ["critical-0"],
                        "releases": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_memory_events_add_logical_nodes_and_only_true_consumer_waits(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "tiers.json"
    digest = _write_native_manifest(manifest_path)
    events_path = tmp_path / "memory-events.json"
    _write_movement_events(events_path, digest)
    trace = tmp_path / "native.txt"
    rows = [_native_layer("block0_layernorm"), _native_layer("unrelated")]
    _write_trace(
        trace,
        rows,
        pp_size=1,
        header_suffix=(
            f"  trace_schema: llm-tier-v1  tier_manifest_digest: {digest}"
        ),
    )

    outputs = LLMConverter(
        str(trace),
        str(tmp_path / "llm"),
        num_npus=1,
        tier_manifest=str(manifest_path),
        memory_events=str(events_path),
    ).convert()

    _, nodes = _read_et(outputs[0])
    movement = {node.name: node for node in nodes if node.name.startswith("MEMORY_MOVEMENT")}
    critical = movement["MEMORY_MOVEMENT_critical-0"]
    background = movement["MEMORY_MOVEMENT_background-0"]
    consumer = next(node for node in nodes if node.name == "COMP_NODE_block0_layernorm")
    unrelated = next(node for node in nodes if node.name == "COMP_NODE_unrelated")
    assert critical.id in consumer.data_deps
    assert critical.id not in unrelated.data_deps
    assert background.id not in consumer.data_deps
    assert _uint_attr(critical, "movement_source_iteration_id") == 0
    assert (
        _uint_attr(critical, "movement_max_in_flight_page_movements") == 1
    )
    assert _string_attr(critical, "movement_page_id") == "page-v1:" + "1" * 64
    assert _string_attr(critical, "movement_transaction_id") == (
        "promote-00000001-page"
    )
    assert _uint_attr(critical, "movement_expected_residency_version") == 0


def test_memory_events_fail_closed_on_cross_pair_and_digest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tiers.json"
    digest = _write_native_manifest(manifest_path)
    events_path = tmp_path / "memory-events.json"
    _write_movement_events(events_path, digest)
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    payload["events"][0]["destination"]["device_id"] = 1
    events_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="paired HBM"):
        LLMConverter(
            "unused",
            "unused",
            num_npus=1,
            tier_manifest=str(manifest_path),
            memory_events=str(events_path),
        )

    _write_movement_events(events_path, f"sha256:{'0' * 64}")
    with pytest.raises(ValueError, match="manifest_digest"):
        LLMConverter(
            "unused",
            "unused",
            num_npus=1,
            tier_manifest=str(manifest_path),
            memory_events=str(events_path),
        )


def test_page_memory_event_requires_transaction_identity(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tiers.json"
    digest = _write_native_manifest(manifest_path)
    events_path = tmp_path / "memory-events.json"
    _write_movement_events(events_path, digest)
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    payload["events"][0].pop("transaction_id")
    events_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="transaction_id must be non-empty"):
        LLMConverter(
            "unused",
            "unused",
            num_npus=1,
            tier_manifest=str(manifest_path),
            memory_events=str(events_path),
        )


def test_memory_events_reject_page_limit_above_engine_count(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tiers.json"
    digest = _write_native_manifest(manifest_path)
    events_path = tmp_path / "memory-events.json"
    _write_movement_events(events_path, digest)
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    payload["selected_path"]["max_in_flight_page_movements"] = 2
    events_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must not exceed engine_count"):
        LLMConverter(
            "unused",
            "unused",
            num_npus=1,
            tier_manifest=str(manifest_path),
            memory_events=str(events_path),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("npu_id", 99, "no output ET"),
        ("releases", ["missing_layer"], "do not name emitted compute"),
    ],
)
def test_memory_events_reject_unconsumed_targets(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    manifest_path = tmp_path / "tiers.json"
    digest = _write_native_manifest(manifest_path)
    events_path = tmp_path / "memory-events.json"
    _write_movement_events(events_path, digest)
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    payload["events"][0][field] = value
    events_path.write_text(json.dumps(payload), encoding="utf-8")
    trace = tmp_path / "native.txt"
    _write_trace(
        trace,
        [_native_layer("block0_layernorm"), _native_layer("unrelated")],
        pp_size=1,
        header_suffix=(
            f"  trace_schema: llm-tier-v1  tier_manifest_digest: {digest}"
        ),
    )

    with pytest.raises(ValueError, match=message):
        LLMConverter(
            str(trace),
            str(tmp_path / "llm"),
            num_npus=1,
            tier_manifest=str(manifest_path),
            memory_events=str(events_path),
        ).convert()

def test_native_segments_emit_one_compute_with_multiple_load_parents(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "tiers.json"
    digest = _write_native_manifest(manifest_path)
    trace = tmp_path / "native.txt"
    rows = [
        _native_layer("moe", "hbm:0@64,lpddr:0@128"),
        _native_layer("output"),
    ]
    _write_trace(
        trace,
        rows,
        pp_size=1,
        header_suffix=(
            f"  trace_schema: llm-tier-v1  tier_manifest_digest: {digest}"
        ),
    )

    outputs = LLMConverter(
        str(trace),
        str(tmp_path / "llm"),
        num_npus=1,
        tier_manifest=str(manifest_path),
    ).convert()

    metadata, nodes = _read_et(outputs[0])
    metadata_attrs = {attr.name: attr.string_val for attr in metadata.attr}
    assert metadata_attrs["tier_manifest_digest"] == digest
    segment_loads = [node for node in nodes if "WEIGHT_SEGMENT" in node.name]
    moe_compute = [node for node in nodes if node.name == "COMP_NODE_moe"]
    assert len(segment_loads) == 2
    assert len(moe_compute) == 1
    assert [_uint_attr(node, "tensor_loc") for node in segment_loads] == [16, 17]
    assert [_uint_attr(node, "tensor_device") for node in segment_loads] == [0, 0]
    assert set(moe_compute[0].data_deps).issuperset(
        {node.id for node in segment_loads}
    )


def test_native_trace_digest_mismatch_fails_before_conversion(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tiers.json"
    _write_native_manifest(manifest_path)
    trace = tmp_path / "stale.txt"
    _write_trace(
        trace,
        [_native_layer("a"), _native_layer("b")],
        pp_size=1,
        header_suffix=(
            "  trace_schema: llm-tier-v1  "
            f"tier_manifest_digest: sha256:{'0' * 64}"
        ),
    )

    with pytest.raises(ValueError, match="tier_manifest_digest mismatch"):
        LLMConverter(
            str(trace),
            str(tmp_path / "llm"),
            num_npus=1,
            tier_manifest=str(manifest_path),
        ).convert()


def test_native_unknown_segment_tier_fails_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tiers.json"
    digest = _write_native_manifest(manifest_path)
    trace = tmp_path / "unknown.txt"
    _write_trace(
        trace,
        [_native_layer("a", "missing:0@64"), _native_layer("b")],
        pp_size=1,
        header_suffix=(
            f"  trace_schema: llm-tier-v1  tier_manifest_digest: {digest}"
        ),
    )

    with pytest.raises(ValueError, match="Cannot parse"):
        LLMConverter(
            str(trace),
            str(tmp_path / "llm"),
            num_npus=1,
            tier_manifest=str(manifest_path),
        ).convert()


@pytest.mark.parametrize(
    ("boundaries", "message"),
    [
        (None, "carries 0 pp_stage_boundaries"),
        ("0", "strictly increasing"),
        ("4,6", "expected 1"),
        ("not-an-int", "invalid pp_stage_boundaries"),
    ],
)
def test_invalid_pp_boundaries_fail_closed(
    tmp_path: Path, boundaries: str | None, message: str
) -> None:
    trace = tmp_path / "bad.txt"
    _write_trace(trace, _dense_rows(), boundaries=boundaries)

    with pytest.raises(ValueError, match=message):
        LLMConverter(str(trace), str(tmp_path / "llm"), num_npus=2).convert()


def test_unmatched_marker_end_fails_closed(tmp_path: Path) -> None:
    rows = [_layer("embedding"), _expert("END"), _layer("lm_head")]
    trace = tmp_path / "unmatched.txt"
    _write_trace(trace, rows, boundaries="1")

    with pytest.raises(ValueError, match="unmatched EXPERT END marker"):
        LLMConverter(str(trace), str(tmp_path / "llm"), num_npus=2).convert()


def _write_nodes(path: Path, nodes: list[Node]) -> None:
    with path.open("wb") as stream:
        encodeMessage(stream, GlobalMetadata(version="test"))
        for node in nodes:
            encodeMessage(stream, node)


def test_et_validator_rejects_node_disguised_as_metadata(tmp_path: Path) -> None:
    path = tmp_path / "missing-metadata.et"
    with path.open("wb") as stream:
        encodeMessage(stream, Node(id=1))
        encodeMessage(stream, Node(id=2, name="payload"))

    with pytest.raises(ETValidationError, match="invalid GlobalMetadata"):
        validate_et_group([path])


def test_et_validator_rejects_missing_local_dependency(tmp_path: Path) -> None:
    node = Node(id=1, name="orphan")
    node.data_deps.append(99)
    path = tmp_path / "orphan.et"
    _write_nodes(path, [node])

    with pytest.raises(ETValidationError, match="missing local dependency 99"):
        validate_et_group([path])


def test_et_validator_rejects_duplicate_node_id(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.et"
    _write_nodes(path, [Node(id=1, name="first"), Node(id=1, name="second")])

    with pytest.raises(ETValidationError, match="duplicate node id"):
        validate_et_group([path])


def test_et_validator_rejects_dependency_cycle(tmp_path: Path) -> None:
    first = Node(id=1, name="first")
    second = Node(id=2, name="second")
    first.data_deps.append(2)
    second.data_deps.append(1)
    path = tmp_path / "cycle.et"
    _write_nodes(path, [first, second])

    with pytest.raises(ETValidationError, match="dependency cycle"):
        validate_et_group([path])


def _has_attr(node: Node, name: str) -> bool:
    return any(attr.name == name for attr in node.attr)


def _write_ucie_manifest(path: Path) -> str:
    payload = {
        "schema_version": "memory-tier-runtime-v1",
        "id_mode": "native",
        "tiers": [
            {
                "tier_name": name,
                "tier_id": 16 + index,
                "backend_kind": "analytical",
                "scope": "instance",
                "pool_key": name,
                "devices": [{"device_id": 0, "capacity_bytes": 1024}],
                "num_devices": 1,
                "mem_bw_gbps": 1000,
                "mem_latency_ns": 100,
            }
            for index, name in enumerate(("hbm", "lpddr", "remote"))
        ],
        "ucie_links": [
            {
                "id": "ucie-frontside",
                "endpoints": ["compute", "hbm"],
                "stack_count": 1,
                "header_bytes": 64,
                "latency_ns": 0,
                "bandwidth_resource": {
                    "schema_version": "bandwidth-resource-v1",
                    "read_bytes_per_second": 500_000_000,
                    "write_bytes_per_second": 500_000_000,
                    "shared_bytes_per_second": 1_000_000_000,
                    "concurrency": "simultaneous",
                    "turnaround_ns": 0,
                },
            }
        ],
    }
    payload["manifest_digest"] = canonical_manifest_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload["manifest_digest"]


def test_native_ucie_links_annotate_hot_mem_nodes_only(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tiers.json"
    digest = _write_ucie_manifest(manifest_path)
    trace = tmp_path / "native.txt"
    _write_trace(
        trace,
        [_native_layer("block0_layernorm"), _native_layer("unrelated")],
        pp_size=1,
        header_suffix=(
            f"  trace_schema: llm-tier-v1  tier_manifest_digest: {digest}"
        ),
    )

    outputs = LLMConverter(
        str(trace),
        str(tmp_path / "llm"),
        num_npus=1,
        tier_manifest=str(manifest_path),
    ).convert()
    _, nodes = _read_et(outputs[0])
    mem_nodes = [node for node in nodes if node.name.startswith("MEM_")]
    assert mem_nodes
    for node in mem_nodes:
        loc = next(attr.uint32_val for attr in node.attr if attr.name == "tensor_loc")
        if loc == 16:
            assert _string_attr(node, "ucie_transport_schema_version") == (
                "ucie-transport-v1"
            )
            assert _string_attr(node, "ucie_link_id") == "ucie-frontside"
        else:
            assert not _has_attr(node, "ucie_link_id")


def test_folded_native_manifest_omits_ucie_attrs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tiers.json"
    digest = _write_native_manifest(manifest_path)
    trace = tmp_path / "native.txt"
    _write_trace(
        trace,
        [_native_layer("block0_layernorm"), _native_layer("unrelated")],
        pp_size=1,
        header_suffix=(
            f"  trace_schema: llm-tier-v1  tier_manifest_digest: {digest}"
        ),
    )
    outputs = LLMConverter(
        str(trace),
        str(tmp_path / "llm"),
        num_npus=1,
        tier_manifest=str(manifest_path),
    ).convert()
    _, nodes = _read_et(outputs[0])
    assert not any(_has_attr(node, "ucie_link_id") for node in nodes)


def test_et_validator_rejects_unpaired_pipeline_send(tmp_path: Path) -> None:
    trace = tmp_path / "moe.txt"
    _write_trace(trace, _moe_rows(), boundaries="6")
    outputs = LLMConverter(
        str(trace), str(tmp_path / "llm"), num_npus=2
    ).convert()

    with pytest.raises(ETValidationError, match="unpaired SEND/RECV"):
        validate_et_group([outputs[0]])
