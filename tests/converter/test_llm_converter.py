from __future__ import annotations

from pathlib import Path

import pytest

from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
from chakra.src.converter.et_validator import ETValidationError, validate_et_group
from chakra.src.converter.llm_converter import LLMConverter
from chakra.src.third_party.utils.protolib import encodeMessage


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


def test_et_validator_rejects_unpaired_pipeline_send(tmp_path: Path) -> None:
    trace = tmp_path / "moe.txt"
    _write_trace(trace, _moe_rows(), boundaries="6")
    outputs = LLMConverter(
        str(trace), str(tmp_path / "llm"), num_npus=2
    ).convert()

    with pytest.raises(ETValidationError, match="unpaired SEND/RECV"):
        validate_et_group([outputs[0]])
