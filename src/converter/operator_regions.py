"""Bounded operator-IO v3 regions; original ET nodes remain execution owners."""

from __future__ import annotations

from dataclasses import dataclass
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
                 max_rows: int) -> None:
        if not isinstance(values, list) or not values:
            raise ValueError("operator region: expected nonempty region array")
        self.rows: dict[tuple[int, int], RegionRow] = {}
        self.intervals: list[tuple[int, int, int, frozenset[int]]] = []
        self.frontiers: dict[tuple[int, int], tuple[Node, ...]] = {}
        self.waits: dict[tuple[int, str], list[Node]] = {}
        for value in values:
            region = _fields(value, {"stage", "entry", "exit", "rows"})
            stage, entry, exit = (_uint(region[key]) for key in ("stage", "entry", "exit"))
            records = region["rows"]
            if (exit <= entry or not isinstance(records, list) or not records
                    or len(records) + len(self.rows) > max_rows):
                raise ValueError("operator region: unordered interval or row budget exceeded")
            ranks = {}
            for record in records:
                row = self._read_row(record, entry, exit)
                key = row.rank, row.row
                io = io_rows.get(key)
                if key in self.rows or io is None:
                    raise ValueError("operator region: overlap or missing original IO row")
                if not row.present and (io.reads or io.writes):
                    raise ValueError("operator region: transparent row cannot own IO")
                self.rows[key] = row
                ranks.setdefault(row.rank, {})[row.row] = row
            for rows in ranks.values():
                self._require_closed(rows, entry, exit)
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
    def _require_closed(rows: dict[int, RegionRow], entry: int, exit: int) -> None:
        if (len(rows) != exit - entry + 1 or entry not in rows or exit not in rows
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

    def validate_movements(self, events: MemoryEvents | None,
                           io_rows: dict[tuple[int, int], OperandIO]) -> None:
        """Validate every rank before any ET opens, without constructing ET nodes."""
        if events is None:
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
