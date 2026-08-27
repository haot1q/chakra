from __future__ import annotations

from pathlib import Path

import pytest
from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
from chakra.src.converter.et_validator import ETValidationError, validate_et_group
from chakra.src.converter.llm_converter import LLMConverter
from chakra.src.third_party.utils.protolib import encodeMessage


def _layer(name: str, input_size: int = 16, output_size: int = 16) -> str:
    return (
        f"{name} 1 LOCAL {input_size} LOCAL 16 LOCAL {output_size} "
        "NONE 0 NONE\n"
    )


def _expert(expert: str) -> str:
    return f"EXPERT {expert} NONE 0\n"


def _moe_rows() -> list[str]:
    """Two blocks whose line-count midpoint falls inside block 1's Expert."""
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


def _write_trace(path: Path, boundaries: str | None, rows: list[str]) -> None:
    header = "COLOCATED  model_parallel_NPU_group: 2"
    if boundaries is not None:
        header += f"  pp_stage_boundaries: {boundaries}"
    path.write_text(
        header + f"\n{len(rows)}\nignored table header\n" + "".join(rows),
        encoding="utf-8",
    )


def test_moe_pp_uses_declared_block_boundary_and_emits_valid_group(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "moe.txt"
    _write_trace(trace, "6", _moe_rows())

    outputs = LLMConverter(
        str(trace), str(tmp_path / "llm"), num_npus=2
    ).convert()

    assert outputs == [tmp_path / "llm.0.et", tmp_path / "llm.1.et"]
    report = validate_et_group(outputs)
    assert report.file_count == 2
    assert report.node_count > 0


@pytest.mark.parametrize(
    ("rows", "boundary"),
    [(_dense_rows(), "4"), (_pim_rows(), "6")],
)
def test_dense_and_pim_pp_emit_valid_groups(
    tmp_path: Path, rows: list[str], boundary: str
) -> None:
    trace = tmp_path / "model.txt"
    _write_trace(trace, boundary, rows)

    outputs = LLMConverter(
        str(trace), str(tmp_path / "llm"), num_npus=2
    ).convert()

    assert validate_et_group(outputs).file_count == 2


def test_pp1_remains_backward_compatible(tmp_path: Path) -> None:
    rows = _dense_rows()
    trace = tmp_path / "dense.txt"
    trace.write_text(
        "COLOCATED  model_parallel_NPU_group: 1"
        f"\n{len(rows)}\nignored table header\n" + "".join(rows),
        encoding="utf-8",
    )

    outputs = LLMConverter(
        str(trace), str(tmp_path / "llm"), num_npus=2
    ).convert()

    assert validate_et_group(outputs).file_count == 2


def test_pp_trace_without_boundaries_fails_closed(tmp_path: Path) -> None:
    trace = tmp_path / "legacy.txt"
    _write_trace(trace, None, _moe_rows())

    with pytest.raises(ValueError, match="carries 0 pp_stage_boundaries"):
        LLMConverter(str(trace), str(tmp_path / "llm"), num_npus=2).convert()


@pytest.mark.parametrize(
    ("boundaries", "message"),
    [
        ("8", "inside EXPERT region"),
        ("0", "strictly increasing"),
        ("6,8", "expected 1"),
        ("not-an-int", "invalid pp_stage_boundaries"),
    ],
)
def test_invalid_pp_boundaries_fail_closed(
    tmp_path: Path, boundaries: str, message: str
) -> None:
    trace = tmp_path / "bad.txt"
    _write_trace(trace, boundaries, _moe_rows())

    with pytest.raises(ValueError, match=message):
        LLMConverter(str(trace), str(tmp_path / "llm"), num_npus=2).convert()


def test_unclosed_expert_region_fails_closed(tmp_path: Path) -> None:
    rows = _moe_rows()[:-3]
    trace = tmp_path / "unclosed.txt"
    _write_trace(trace, "6", rows)

    with pytest.raises(ValueError, match="unclosed EXPERT region"):
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
    _write_trace(trace, "6", _moe_rows())
    outputs = LLMConverter(
        str(trace), str(tmp_path / "llm"), num_npus=2
    ).convert()

    with pytest.raises(ETValidationError, match="unpaired SEND/RECV"):
        validate_et_group([outputs[0]])
