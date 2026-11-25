"""Bounded operator-IO v3 regions; original ET nodes remain execution owners."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import chain
from typing import TYPE_CHECKING

from ...schema.protobuf.et_def_pb2 import Node
from .memory_events import MemoryEvents, MovementEvent, PriorMovement

if TYPE_CHECKING:
    from .llm_converter import LLMConverter
    from .operator_io import OperandIO


def _fields(value: object, expected: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("operator region: missing or unknown fields")
    return value


def _uint(value: object) -> int:
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("operator region: expected bounded unsigned integer")
    return value


@dataclass(frozen=True)
class RegionRow:
    rank: int
    row: int
    present: bool
    parents: tuple[int, ...]
    entry: int


class OperatorRegions:
    """One indexed invocation; no scheduler, allocation or source hazard state."""

    def __init__(self, values: object, io_rows: dict[tuple[int, int], OperandIO],
                 max_rows: int | None, *, sparse: bool = False) -> None:
        if not isinstance(values, list) or not values:
            raise ValueError("operator region: expected nonempty region array")
        self.sparse = sparse
        self.region_by_index: dict[int, int] = {}
        self.next_after: dict[tuple[int, int], int] = {}
        self.rows: dict[tuple[int, int], RegionRow] = {}
        self.intervals: list[tuple[int, int, int, frozenset[int]]] = []
        self.frontiers: dict[tuple[int, int], tuple[Node, ...]] = {}
        self.waits: dict[tuple[int, str], list[Node]] = {}
        for value in values:
            region = _fields(value, {"stage", "entry", "exit", "rows"} | ({"source_owner_sha256"} if sparse else set()))
            stage, entry, exit = (_uint(region[key]) for key in ("stage", "entry", "exit"))
            records = region["rows"]
            if (exit <= entry or not isinstance(records, list) or not records
                    or max_rows is not None and len(records) + len(self.rows) > max_rows):
                raise ValueError("operator region: unordered interval or row budget exceeded")
            ranks = {}
            for record in records:
                row = self._read_row(record, entry, exit)
                key = row.rank, row.row
                io = io_rows.get(key)
                if key in self.rows or io is None:
                    raise ValueError(f"operator region: overlap or missing original IO row rank={row.rank} row={row.row}")
                if not row.present and (io.reads or io.writes):
                    raise ValueError("operator region: transparent row cannot own IO")
                self.rows[key] = row
                ranks.setdefault(row.rank, {})[row.row] = row
            if sparse:
                digest = hashlib.sha256()
                for rank in sorted(ranks):
                    indices = tuple(sorted(ranks[rank]))
                    for row in indices:
                        if not ranks[rank][row].present:
                            raise ValueError(f"operator region: sparse rank={rank} row={row} must be present")
                        ordered = ",".join(map(str, ranks[rank][row].parents))
                        digest.update(f"{rank}/{row}:{ordered}\n".encode("ascii"))
                    for previous, following in zip(indices, indices[1:]):
                        if following != previous + 1:
                            self.next_after[rank, previous] = following
                if region["source_owner_sha256"] != digest.hexdigest():
                    raise ValueError(f"operator region: Source owner digest mismatch at entry={entry}")
                # Populate the public row index from observed real rows. Reject
                # even a maliciously wide interval without traversing its range.
                union = {i for owned in ranks.values() for i in owned}
                if len(union) != exit - entry + 1 or any(i in self.region_by_index for i in union):
                    raise ValueError("operator region: incomplete or overlapping sparse interval")
                self.region_by_index.update(dict.fromkeys(union, entry))
            for rows in ranks.values():
                self._require_closed(rows, entry, exit, sparse)
            self.intervals.append((stage, entry, exit, frozenset(ranks)))

    @staticmethod
    def _read_row(value: object, entry: int, exit: int) -> RegionRow:
        item = _fields(value, {"rank", "row", "present", "parents"})
        rank, row = _uint(item["rank"]), _uint(item["row"])
        if not entry <= row <= exit or type(item["present"]) is not bool:
            raise ValueError("operator region: invalid row or presence")
        if not isinstance(item["parents"], list):
            raise ValueError("operator region: parents must be an array")
        parents = tuple(_uint(parent) for parent in item["parents"])
        if len(set(parents)) != len(parents) or any(not entry <= p < row for p in parents):
            raise ValueError("operator region: duplicate, outside or nonpreceding parent")
        if (row == entry) != (not parents):
            raise ValueError("operator region: expected a single entry")
        if not item["present"] and parents != (entry,):
            raise ValueError("operator region: transparent row must pass through entry")
        return RegionRow(rank, row, item["present"], parents, entry)

    @staticmethod
    def _require_closed(rows: dict[int, RegionRow], entry: int, exit: int, sparse: bool) -> None:
        if ((not sparse and len(rows) != exit - entry + 1) or entry not in rows or exit not in rows
                or not rows[entry].present or not rows[exit].present):
            raise ValueError("operator region: incomplete coverage or absent entry/exit")
        for row in rows.values():
            if any(parent not in rows or not rows[parent].present for parent in row.parents):
                raise ValueError("operator region: parent is not a present row")
        reached, pending = set(), [exit]
        while pending:
            index = pending.pop()
            if index not in reached:
                reached.add(index)
                pending.extend(rows[index].parents)
        if any(row.present and row.row not in reached for row in rows.values()):
            raise ValueError("operator region: exit does not join every present row")

    def validate_rows(self, stages: list[tuple[int, int]], prefix_count: int,
                      tp: int, rank_offset: int) -> None:
        for stage, entry, exit, ranks in self.intervals:
            if stage >= len(stages):
                raise ValueError("operator region: unknown pipeline stage")
            start, stop = stages[stage]
            owners = frozenset(range(rank_offset + stage * tp, rank_offset + (stage + 1) * tp))
            if (ranks != owners or not start <= entry - prefix_count <= exit - prefix_count < stop):
                raise ValueError("operator region: row/rank crosses its pipeline stage")
        self.frontiers.clear()
        self.waits.clear()

    def stage_indices(self, rank: int, start: int, stop: int, prefix: int):
        """Original outside rows plus only the rank's in-region work."""
        index = start
        while index < stop:
            yield index
            global_row = index + prefix
            index = self.next_after.get((rank, global_row), global_row + 1) - prefix

    def validate_movements(self, events: MemoryEvents | None,
                           io_rows: dict[tuple[int, int], OperandIO], *, layers=(), prefix: int = 0) -> None:
        """Validate every rank before any ET opens, without constructing ET nodes."""
        if events is None:
            return
        if self.sparse:
            names = {layer.name: i + prefix for i, layer in enumerate(layers)
                     if i + prefix in self.region_by_index}
            for event in chain(events.events, events.prior_events):
                waits = event.waits_for if isinstance(event, MovementEvent) else ()
                for name in chain(event.releases, waits):
                    if name in names and (event.npu_id, names[name]) not in self.rows:
                        raise ValueError(f"operator region: absent Page anchor rank={event.npu_id} row={names[name]}")
            return
        absent = {(row.rank, io_rows[key].name) for key, row in self.rows.items()
                  if not row.present}
        for event in chain(events.events, events.prior_events):
            waits = event.waits_for if isinstance(event, MovementEvent) else ()
            if any((event.npu_id, name) in absent for name in chain(event.releases, waits)):
                raise ValueError("operator region: transparent row cannot own Page events")

    def prepare_rank(self, rank: int,
                     movement_nodes: dict[str, tuple[MovementEvent | PriorMovement, Node]]) -> None:
        """Bind this rank's original nodes once; prior waits only release consumers."""
        for event, node in movement_nodes.values():
            if isinstance(event, MovementEvent):
                for name in event.waits_for:
                    self.waits.setdefault((rank, name), []).append(node)

    def internal(self, rank: int, index: int) -> bool:
        row = self.rows.get((rank, index))
        return row is not None and index != row.entry

    def add_parents(self, converter: LLMConverter, node: Node, rank: int, index: int) -> None:
        row = self.rows.get((rank, index))
        if row is not None:
            for parent in row.parents:
                for previous in self.frontiers[rank, parent]:
                    converter.add_parent(node, previous)

    def complete(self, converter: LLMConverter, node: Node, writes: tuple[Node, ...],
                 rank: int, index: int) -> None:
        row = self.rows.get((rank, index))
        if row is None:
            return
        if row.present:
            self.frontiers[rank, index] = writes or (node,)
        else:
            # Its retained zero-COMP anchor is not a source completion proof.
            self.frontiers[rank, index] = self.frontiers[rank, row.entry]
        for movement in self.waits.get((rank, node.name.removeprefix("COMP_NODE_")), ()):
            for write in writes:
                converter.add_parent(movement, write)
