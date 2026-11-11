"""Bounded original-access associations for already existing operand nodes."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Mapping

from ...schema.protobuf.et_def_pb2 import (
    AttributeProto, MEM_LOAD_NODE, MEM_STORE_NODE, Node,
)
from .operator_io import OperatorIO
from .tier_manifest import TierManifest


DEFAULT_LIMITS = MappingProxyType({
    "context_bytes": 65_536, "association_bytes": 67_108_864,
    "json_depth": 32, "string_bytes": 512, "table_records": 262_144,
    "pieces": 1_048_576, "et_bytes": 268_435_456, "frame_bytes": 4_194_304,
    "et_nodes": 1_000_000, "pending_graphs_per_rank": 2,
    "live_index_charge_bytes": 268_435_456, "live_pieces": 1_048_576,
})
_UINT = re.compile(r"0|[1-9][0-9]*")
_INT = re.compile(r"0|-?[1-9][0-9]*")
_HASH = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("source association: " + message)


def _fields(value: object, names: str) -> dict:
    _require(isinstance(value, dict) and set(value) == set(names.split()),
             "missing or unknown fields")
    return value


def _uint(value: object, *, positive: bool = False) -> int:
    _require(isinstance(value, str) and len(value) <= 20 and _UINT.fullmatch(value),
             "expected canonical unsigned decimal string")
    result = int(value)
    _require(int(positive) <= result < 2**64, "unsigned value outside uint64")
    return result


def _signed(value: object) -> int:
    _require(isinstance(value, str) and len(value) <= 20 and _INT.fullmatch(value),
             "expected canonical signed decimal string")
    result = int(value)
    _require(-(2**63) <= result < 2**63, "stride outside int64")
    return result


def _string(value: object) -> str:
    _require(isinstance(value, str) and bool(value), "expected nonempty string")
    return value


def _hash(value: object) -> str:
    _require(isinstance(value, str) and _HASH.fullmatch(value), "invalid SHA256")
    return value


def _limits(values: Mapping[str, int | str] | None) -> dict[str, int]:
    result = dict(DEFAULT_LIMITS)
    if values is not None:
        _require(isinstance(values, Mapping) and set(values) <= result.keys(), "unknown limit")
        for name, raw in values.items():
            value = _uint(raw, positive=True) if isinstance(raw, str) else raw
            _require(type(value) is int and 0 < value <= result[name], "invalid host limit")
            result[name] = value
    return result


def parse_limits_argument(value: str) -> dict[str, int]:
    """Parse bounded host CLI limits; public association scalars stay strings."""
    _require(len(value.encode("utf-8")) <= DEFAULT_LIMITS["context_bytes"], "limits argument too large")
    _depth(value, DEFAULT_LIMITS["json_depth"])
    values = json.loads(value, object_pairs_hook=_object)
    _require(isinstance(values, dict), "limits argument must be an object")
    return _limits(values)


def _regular_bytes(path: str | Path, limit: int) -> bytes:
    candidate = Path(path).absolute()
    _require(".." not in candidate.parts, "path must not traverse upward")
    _require(not any(part.is_symlink() for part in (candidate, *candidate.parents)),
             "symlink input is not allowed")
    before = candidate.stat()
    _require(stat.S_ISREG(before.st_mode) and before.st_size <= limit,
             "expected regular input within byte limit")
    with candidate.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        _require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                 "input changed before read")
        encoded = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    def identity(info: os.stat_result) -> tuple[int, int, int, int]:
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
    _require(len(encoded) <= limit and identity(before) == identity(opened) == identity(after)
             == identity(candidate.stat()),
             "input exceeded byte limit or changed during read")
    return encoded


def _depth(text: str, maximum: int) -> None:
    # Bound nesting before the JSON parser allocates recursive containers.
    depth = 0
    quoted = escaped = False
    for char in text:
        if escaped:
            escaped = False
        elif quoted and char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted:
            depth += int(char in "[{") - int(char in "]}")
            _require(0 <= depth <= maximum, "JSON nesting exceeds limit")


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _strings(value: object, maximum: int) -> None:
    if isinstance(value, str):
        _require("\0" not in value and len(value.encode("utf-8")) <= maximum,
                 "invalid or oversized string")
    elif isinstance(value, dict):
        for key, child in value.items():
            _strings(key, maximum)
            _strings(child, maximum)
    elif isinstance(value, list):
        for child in value:
            _strings(child, maximum)
    else:
        _require(value is None, "wire scalars must be strings or null")


def _document(encoded: bytes, limits: Mapping[str, int]) -> dict:
    try:
        text = encoded.decode("utf-8")
        _depth(text, limits["json_depth"])
        value = json.loads(text, object_pairs_hook=_object)
        _strings(value, limits["string_bytes"])
    except (UnicodeError, RecursionError) as error:
        raise ValueError("source association: malformed JSON encoding") from error
    return _fields(value, "schema run_id source_provenance_digest source_profile "
                   "source_selection_sha256 trace_sha256 operator_io_sha256 tier_manifest_digest ranks")


def _table(value: object, fields: str, limit: int) -> list[dict]:
    _require(isinstance(value, list) and len(value) <= limit, "table exceeds record limit")
    for index, item in enumerate(value):
        _fields(item, fields)
        _require(item["ref"] == str(index), "table refs must be canonical indices")
    return value


def _reference(value: object, table: list[dict]) -> dict:
    index = _uint(value)
    _require(index < len(table), "table reference does not exist")
    return table[index]


def _invocations(value: object, limit: int) -> list[dict]:
    table = _table(value, "ref batch_id invocation_id", limit)
    identities = set()
    for item in table:
        identity = (_string(item["batch_id"]), _string(item["invocation_id"]))
        _require(identity not in identities, "duplicate invocation identity")
        identities.add(identity)
    return table


def _backings(value: object, limit: int) -> list[dict]:
    table = _table(value, "ref backing_identity extent_bytes catalog_generation", limit)
    identities = set()
    for item in table:
        backing = _fields(item["backing_identity"], "profile_id owner_scope allocation_id")
        identity = tuple(_string(backing[key]) for key in ("profile_id", "owner_scope", "allocation_id"))
        _require(identity not in identities, "duplicate backing identity")
        identities.add(identity)
        _uint(item["extent_bytes"], positive=True)
        if item["catalog_generation"] is not None:
            _string(item["catalog_generation"])
    return table


def _operands(value: object, invocations: list, backings: list, limit: int) -> list[dict]:
    table = _table(value, "ref invocation_ref source_boundary execution_layer role origin operation "
                   "ordinal_start backing_ref offset_bytes length_bytes stride_bytes count", limit)
    occurrences = 0
    ordinal_ranges = defaultdict(list)
    for item in table:
        _reference(item["invocation_ref"], invocations)
        backing = _reference(item["backing_ref"], backings)
        _string(item["source_boundary"])
        _string(item["role"])
        _uint(item["execution_layer"])
        _require(item["operation"] in ("read", "write") and item["origin"] in (
            ("reads", "captured_page_reads") if item["operation"] == "read" else ("writes",)),
            "operand operation/origin mismatch")
        offset, size = _uint(item["offset_bytes"]), _uint(item["length_bytes"], positive=True)
        count, stride = _uint(item["count"], positive=True), _signed(item["stride_bytes"])
        last = offset + (count - 1) * stride
        _require(min(offset, last) >= 0 and max(offset, last) + size <= int(backing["extent_bytes"]),
                 "operand run exceeds backing extent")
        ordinal = _uint(item["ordinal_start"])
        _require(ordinal + count <= 2**64, "operand ordinal overflows")
        identity = tuple(item[k] for k in ("invocation_ref", "source_boundary", "origin"))
        ordinal_ranges[identity].append((ordinal, ordinal + count))
        occurrences += count
        _require(occurrences <= limit, "expanded occurrences exceed record limit")
    for ranges in ordinal_ranges.values():
        previous = -1
        for start, end in sorted(ranges):
            _require(start >= previous, "duplicate original occurrence identity")
            previous = end
    return table


def _authorization(value: object) -> None:
    item = _fields(value, "kind owner page_id residency_version")
    _require(item["kind"] in ("workspace", "fixed_weight", "page"), "unknown authorization")
    owner = item["owner"]
    if isinstance(owner, list):
        _require(len(owner) == 3, "authorization owner triplet required")
        for part in owner:
            _string(part)
    else:
        _string(owner)
    if item["kind"] == "page":
        _string(item["page_id"])
        _uint(item["residency_version"])
    else:
        _require(item["page_id"] is None and item["residency_version"] is None,
                 "non-Page authorization cannot carry Page identity")


def _pieces(value: object, operands: list, manifest: TierManifest, limit: int) -> list[dict]:
    table = _table(value, "ref operand_ref run_index offset_bytes length_bytes location "
                   "authorization volume_ordinal", limit)
    intervals = defaultdict(list)
    for item in table:
        operand = _reference(item["operand_ref"], operands)
        index = _uint(item["run_index"])
        _require(index < int(operand["count"]), "piece occurrence does not exist")
        offset, size = _uint(item["offset_bytes"]), _uint(item["length_bytes"], positive=True)
        _string(item["location"])
        try:
            manifest.resolve(item["location"])
        except (ValueError, KeyError, IndexError) as error:
            raise ValueError("source association: unknown piece endpoint") from error
        _authorization(item["authorization"])
        if operand["origin"] == "captured_page_reads":
            _require(item["authorization"]["kind"] == "page"
                     and item["volume_ordinal"] is not None,
                     "captured Page read requires original volume authorization")
        if item["volume_ordinal"] is not None:
            _uint(item["volume_ordinal"])
        intervals[(item["operand_ref"], index)].append((offset, offset + size))
    for operand in operands:
        for index in range(int(operand["count"])):
            cursor = int(operand["offset_bytes"]) + index * int(operand["stride_bytes"])
            end = cursor + int(operand["length_bytes"])
            for start, stop in sorted(intervals.get((operand["ref"], index), ())):
                _require(start == cursor and stop <= end, "piece coverage has overlap or gap")
                cursor = stop
            _require(cursor == end, "piece coverage is incomplete")
    return table


class SourceExecutionAssociation:
    """Validate one input and attach only its declared original MEM references.

    ``document`` and ``ranks`` are validated producer records, retained for the
    finalizer. Consumers must treat them as read-only. No final ET IDs are guessed
    here: graph adapters may still renumber nodes after conversion.
    """

    def __init__(self, path: str | Path, *, operator_io: OperatorIO,
                 manifest: TierManifest, trace_path: str | Path,
                 limits: Mapping[str, int | str] | None = None) -> None:
        self.limits = MappingProxyType(_limits(limits))
        encoded = _regular_bytes(path, self.limits["association_bytes"])
        self.input_sha256 = hashlib.sha256(encoded).hexdigest()
        self.document = _document(encoded, self.limits)
        self._validate_identity(operator_io, manifest, trace_path)
        raw_ranks = self.document["ranks"]
        _require(isinstance(raw_ranks, list) and 0 < len(raw_ranks) <= self.limits["table_records"],
                 "expected bounded nonempty ranks")
        self.ranks: dict[int, dict] = {}
        self.groups: dict[tuple[int, int, str, int], dict] = {}
        self._emitted: set[tuple[int, int, str, int]] = set()
        self._manifest = manifest
        for raw in raw_ranks:
            rank = self._rank(raw, operator_io, manifest)
            _require(rank not in self.ranks, "duplicate rank")
            self.ranks[rank] = raw

    def _validate_identity(self, operator_io: OperatorIO, manifest: TierManifest,
                           trace_path: str | Path) -> None:
        body = self.document
        _require(body["schema"] == "source-execution-association-v1", "unsupported schema")
        _string(body["run_id"])
        _string(body["source_profile"])
        _hash(body["source_provenance_digest"])
        _hash(body["source_selection_sha256"])
        _require(_hash(body["operator_io_sha256"]) == operator_io.input_sha256,
                 "operator IO digest mismatch")
        digest = hashlib.sha256()
        with Path(trace_path).open("rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                digest.update(block)
        _require(_hash(body["trace_sha256"]) == digest.hexdigest() == operator_io.trace_digest,
                 "Trace digest mismatch")
        _require(body["tier_manifest_digest"] == manifest.digest, "tier manifest mismatch")

    def _rank(self, raw: object, operator_io: OperatorIO, manifest: TierManifest) -> int:
        item = _fields(raw, "execution invocations backings operands pieces groups source_unassociated")
        context = _fields(item["execution"], "run_id source_provenance_digest execution_iteration rank")
        _require(all(context[key] == self.document[key] for key in ("run_id", "source_provenance_digest")),
                 "rank execution context differs from input")
        _require(_uint(context["execution_iteration"]) < 2**32, "backend iteration exceeds uint32")
        rank = _uint(context["rank"])
        limit = self.limits["table_records"]
        invocations = _invocations(item["invocations"], limit)
        backings = _backings(item["backings"], limit)
        operands = _operands(item["operands"], invocations, backings, limit)
        _require(len(invocations) + len(backings) + len(operands) <= limit, "combined table limit exceeded")
        pieces = _pieces(item["pieces"], operands, manifest, self.limits["pieces"])
        covered = self._groups(item["groups"], pieces, operands, rank, operator_io)
        self._omissions(item["source_unassociated"], pieces, covered)
        return rank

    def _groups(self, values: object, pieces: list, operands: list,
                rank: int, operator_io: OperatorIO) -> set[str]:
        _require(isinstance(values, list) and len(values) <= self.limits["pieces"], "group limit exceeded")
        covered = set()
        for value in values:
            group = _fields(value, "ref row operation segment_index location bytes piece_refs")
            row, segment = _uint(group["row"]), _uint(group["segment_index"])
            operation, location = group["operation"], _string(group["location"])
            _require(operation in ("read", "write"), "unknown group operation")
            key = (rank, row, operation, segment)
            _require(key not in self.groups and group["ref"] == f"r{rank}/row{row}/{operation}/s{segment}",
                     "duplicate or noncanonical group identity")
            io = operator_io.rows.get((rank, row))
            _require(io is not None, "group has no original operand row")
            segments = io.reads if operation == "read" else io.writes
            size = _uint(group["bytes"], positive=True)
            _require(segment < len(segments) and segments[segment] == (location, size),
                     "group differs from original segment bill")
            self._group_pieces(group, pieces, operands, covered)
            self.groups[key] = group
        return covered

    def _group_pieces(self, group: dict, pieces: list, operands: list, covered: set[str]) -> None:
        refs = group["piece_refs"]
        _require(isinstance(refs, list) and 0 < len(refs) <= self.limits["pieces"], "empty or oversized group")
        total = 0
        for ref in refs:
            piece = _reference(ref, pieces)
            operand = operands[int(piece["operand_ref"])]
            _require(ref not in covered, "piece is associated more than once")
            _require(piece["location"] == group["location"] and operand["operation"] == group["operation"],
                     "piece endpoint or operation differs from group")
            covered.add(ref)
            total += int(piece["length_bytes"])
        _require(total == int(group["bytes"]), "group piece bytes differ from original bill")

    def _omissions(self, values: object, pieces: list, covered: set[str]) -> None:
        _require(isinstance(values, list) and len(values) <= self.limits["pieces"], "omission limit exceeded")
        omitted = set()
        for value in values:
            item = _fields(value, "piece_ref reason")
            ref = item["piece_ref"]
            _reference(ref, pieces)
            _require(ref not in covered and ref not in omitted, "duplicate or associated omission")
            _require(item["reason"] in ("no_existing_mem_lowering", "outside_selected_segment"),
                     "unknown omission reason")
            omitted.add(ref)
        _require(len(covered) + len(omitted) == len(pieces), "piece missing association/omission")
        occurrence_modes = {}
        for piece in pieces:
            identity = (piece["operand_ref"], piece["run_index"])
            mode = piece["ref"] in covered
            _require(identity not in occurrence_modes or occurrence_modes[identity] == mode,
                     "partially qualified original occurrence")
            occurrence_modes[identity] = mode

    def reset_emission(self, ranks: range) -> None:
        """Validate actual converter rank coverage before opening outputs."""
        _require(isinstance(ranks, range) and ranks.step == 1
                 and ranks.stop - ranks.start == len(self.ranks)
                 and all(rank in ranks for rank in self.ranks),
                 "input ranks differ from actual converter ranks")
        self._emitted.clear()

    def attach(self, node: Node, rank: int, row: int, operation: str,
               segment_index: int) -> None:
        """Attach an exact reference to the node the original converter created."""
        key = (rank, row, operation, segment_index)
        group = self.groups.get(key)
        if group is None:
            return
        expected = MEM_LOAD_NODE if operation == "read" else MEM_STORE_NODE
        values = {}
        for attr in node.attr:
            _require(attr.name not in values, "duplicate memory node attribute")
            values[attr.name] = attr
        endpoint = self._manifest.resolve(group["location"])
        required = {"tensor_size": ("uint64_val", int(group["bytes"])),
                    "tensor_loc": ("uint32_val", endpoint[0]),
                    "tensor_device": ("uint32_val", endpoint[1])}
        _require(node.type == expected and all(
            name in values and values[name].WhichOneof("value") == kind
            and getattr(values[name], kind) == value for name, (kind, value) in required.items()),
                 "emitted memory node differs from declared group")
        _require(key not in self._emitted and not any(a.name == "joint_access_ref" for a in node.attr),
                 "duplicate emitted reference")
        node.attr.append(AttributeProto(name="joint_access_ref", string_val=group["ref"]))
        self._emitted.add(key)

    def validate_emission(self) -> None:
        """Every selected segment must be emitted exactly once."""
        _require(self._emitted == self.groups.keys(), "declared segment was not emitted")
