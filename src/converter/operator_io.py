"""Opt-in operand IO for existing memory nodes, not a memory state engine."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

from ...schema.protobuf.et_def_pb2 import Node
from .tier_manifest import TierManifest

# Complete multi-rank Native sidecars exceed the old 4 MiB component budget.
# Keep a finite bound; row/segment limits and strict field validation remain.
MAX_OPERATOR_IO_BYTES = 256 * 1024 * 1024
MAX_OPERATOR_IO_ROWS = 1_000_000

if TYPE_CHECKING:
    from .llm_converter import LLMConverter, Layer


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("operator IO: duplicate JSON key")
        result[key] = value
    return result


def _fields(value: object, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("operator IO: missing or unknown fields")
    return value


def _uint(value: object, *, positive: bool = False) -> int:
    if type(value) is not int or not (int(positive) <= value < 2**64):
        raise ValueError("operator IO: expected bounded unsigned integer")
    return value


def _segments(values: object, manifest: TierManifest) -> tuple[tuple[str, int], ...]:
    if not isinstance(values, list) or len(values) > 4096:
        raise ValueError("operator IO: expected bounded memory segments")
    result = []
    for value in values:
        item = _fields(value, {"location", "bytes"})
        location = item["location"]
        if not isinstance(location, str):
            raise ValueError("operator IO: location must be a tier/device string")
        try:
            manifest.resolve(location)
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError("operator IO: location is not a manifest endpoint") from error
        result.append((location, _uint(item["bytes"], positive=True)))
    return tuple(result)


@dataclass(frozen=True)
class OperandIO:
    name: str
    reads: tuple[tuple[str, int], ...]
    writes: tuple[tuple[str, int], ...]


class OperatorIO:
    """Bind explicit internal operand traffic to exact Trace rows and ranks.

    Global row indexes include prefix rows. Legacy ingress/final output and
    collectives keep their old owners; this schema describes neither. Each
    annotated row declares every TP member, including an empty IO declaration
    when a member has no traffic. Unannotated rows retain legacy semantics.
    """

    def __init__(self, path: str, manifest: TierManifest) -> None:
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("operator IO: expected regular sidecar file")
        with source.open("rb") as stream:
            encoded = stream.read(MAX_OPERATOR_IO_BYTES + 1)
        if len(encoded) > MAX_OPERATOR_IO_BYTES:
            raise ValueError("operator IO: sidecar exceeds byte limit")
        body = _fields(json.loads(encoded, object_pairs_hook=_object), {
            "schema_version", "accounting", "tier_manifest_digest", "trace_sha256", "rows",
        })
        if (body["schema_version"] not in ("operator-io-v1", "operator-io-v2")
                or body["accounting"] != "ucie_transport_v1"
                or body["tier_manifest_digest"] != manifest.digest):
            raise ValueError("operator IO: schema/accounting/manifest mismatch")
        digest = body["trace_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("operator IO: malformed Trace digest")
        self.trace_digest = digest
        self.schema_version = body["schema_version"]
        self._trace_validated = False
        rows = body["rows"]
        if not isinstance(rows, list) or not rows or len(rows) > MAX_OPERATOR_IO_ROWS:
            raise ValueError("operator IO: expected nonempty bounded rows")
        self.rows: dict[tuple[int, int], OperandIO] = {}
        self._emitted: set[tuple[int, int]] = set()
        fields = {"rank", "row", "reads", "writes"}
        if self.schema_version == "operator-io-v1":
            fields.add("name")
        for raw in rows:
            row = _fields(raw, fields)
            key = (_uint(row["rank"]), _uint(row["row"]))
            name = row["name"] if self.schema_version == "operator-io-v1" else ""
            if key in self.rows or not isinstance(name, str) or (
                    self.schema_version == "operator-io-v1" and not name):
                raise ValueError("operator IO: duplicate coordinate or empty row name")
            self.rows[key] = OperandIO(name, _segments(row["reads"], manifest),
                                       _segments(row["writes"], manifest))

    def validate_trace(self, path: str, header: dict[str, str], execution: str) -> None:
        self._emitted.clear()
        self._trace_validated = False
        if (header.get("operator_io") != "ucie_transport_v1"
                or execution not in {"COLOCATED", "DECODE", "PREFILL"}):
            raise ValueError("operator IO: incompatible Trace header or execution mode")
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.trace_digest:
            raise ValueError("operator IO: Trace content digest mismatch")
        self._trace_validated = True

    def validate_rows(
        self, layers: list[Layer], stages: list[tuple[int, int]], *,
        prefix_count: int, tp: int, rank_offset: int,
    ) -> None:
        """Reject wrong ownership or double charging before any ET is created."""
        if not self._trace_validated:
            raise ValueError("operator IO: validate Trace identity before resolving rows")
        if any(layer.is_pim or (not layer.is_expert and layer.misc != "NONE") for layer in layers):
            raise ValueError("operator IO: PIM/interleaving is not supported")
        declared: dict[int, tuple[set[int], set[int]]] = {}
        expert_rows: set[int] = set()
        in_expert = False
        for index, layer in enumerate(layers):
            if layer.is_expert:
                in_expert = layer.expert_num != "END"
            elif in_expert:
                expert_rows.add(index)
        for (rank, row), io in self.rows.items():
            index = row - prefix_count
            if not 0 <= index < len(layers):
                raise ValueError("operator IO: row outside model Trace")
            if index in expert_rows:
                raise ValueError("operator IO: expert-owned rows are not supported")
            layer = layers[index]
            if self.schema_version == "operator-io-v2":
                io = OperandIO(layer.name, io.reads, io.writes)
                self.rows[rank, row] = io
            if (layer.is_expert or io.name != layer.name or layer.comm_type != "NONE"
                    or layer.weight_memory_size or layer.weight_segments):
                raise ValueError("operator IO: row identity or legacy IO/collective overlap")
            if row == prefix_count or index == len(layers) - 1:
                raise ValueError("operator IO: ingress/final output keep legacy ownership")
            stage = next(i for i, (start, stop) in enumerate(stages) if start <= index < stop)
            owners = set(range(rank_offset + stage * tp, rank_offset + (stage + 1) * tp))
            if rank not in owners:
                raise ValueError("operator IO: rank does not own this pipeline row")
            declared.setdefault(row, (owners, set()))[1].add(rank)
        if any(owners != actual for owners, actual in declared.values()):
            raise ValueError("operator IO: missing participating rank declaration")

    def validate_emission(self) -> None:
        """Prevent later converter branches from silently dropping a declaration."""
        if self._emitted != self.rows.keys():
            raise ValueError("operator IO: declared rank/row was not emitted")

    def before_compute(
        self, converter: LLMConverter, node: Node, rank: int, row: int,
    ) -> tuple[Node, ...]:
        """Reads inherit the original producer, ingress and Page wait frontier."""
        io = self.rows.get((rank, row))
        if io is None:
            return ()
        parents = tuple(node.data_deps)
        reads = []
        for index, (location, size) in enumerate(io.reads):
            read = converter.get_memory_load_node(io.name, f"OPERAND_READ_{index}", location, size)
            read.data_deps.extend(parents)
            converter.add_parent(node, read)
            reads.append(read)
        return tuple(reads)

    def after_compute(
        self, converter: LLMConverter, node: Node, rank: int, row: int,
    ) -> tuple[Node, ...]:
        """Return all stores; the converter must join them before continuation."""
        io = self.rows.get((rank, row))
        if io is None:
            return ()
        if (rank, row) in self._emitted:
            raise ValueError("operator IO: rank/row emitted twice")
        self._emitted.add((rank, row))
        writes = []
        for index, (location, size) in enumerate(io.writes):
            write = converter.get_memory_store_node(io.name, f"OPERAND_WRITE_{index}", location, size)
            converter.add_parent(write, node)
            writes.append(write)
        return tuple(writes)
