"""Correctness checks for Chakra execution-trace groups."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from google.protobuf.message import DecodeError

from ...schema.protobuf.et_def_pb2 import (
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    GlobalMetadata,
    Node,
)
from ..third_party.utils.protolib import decodeMessage as decode_message


class ETValidationError(ValueError):
    """Raised when an execution trace cannot be consumed safely."""


@dataclass(frozen=True)
class ETValidationReport:
    """Small success summary returned to callers and tests."""

    file_count: int
    node_count: int
    send_count: int
    recv_count: int


def _read_nodes(path: Path) -> list[Node]:
    try:
        with path.open("rb") as stream:
            metadata = GlobalMetadata()
            if not decode_message(stream, metadata):
                raise ETValidationError(f"{path}: missing GlobalMetadata")
            nodes = []
            while True:
                node = Node()
                if not decode_message(stream, node):
                    break
                nodes.append(node)
    except ETValidationError:
        raise
    except (OSError, DecodeError) as exc:
        raise ETValidationError(f"{path}: cannot decode execution trace: {exc}") from exc

    if not nodes:
        raise ETValidationError(f"{path}: execution trace has no nodes")
    return nodes


def _attribute(node: Node, name: str, path: Path) -> object:
    for attr in node.attr:
        if attr.name != name:
            continue
        value_field = attr.WhichOneof("value")
        if value_field is None:
            break
        value = getattr(attr, value_field)
        return tuple(value.values) if value_field.endswith("_list") else value
    raise ETValidationError(f"{path}: node {node.id} is missing attribute {name}")


def _validate_local_graph(path: Path, nodes: Sequence[Node]) -> None:
    node_ids = {node.id for node in nodes}
    if len(node_ids) != len(nodes):
        raise ETValidationError(f"{path}: duplicate node id")

    children: dict[int, set[int]] = defaultdict(set)
    indegree = {node_id: 0 for node_id in node_ids}
    for node in nodes:
        for parent_id in (*node.data_deps, *node.ctrl_deps):
            if parent_id not in node_ids:
                raise ETValidationError(
                    f"{path}: node {node.id} has missing local dependency {parent_id}"
                )
            if node.id not in children[parent_id]:
                children[parent_id].add(node.id)
                indegree[node.id] += 1

    ready = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node_id = ready.popleft()
        visited += 1
        for child_id in children[node_id]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                ready.append(child_id)
    if visited != len(nodes):
        raise ETValidationError(f"{path}: dependency cycle detected")


def _p2p_key(path: Path, node: Node) -> tuple[object, ...]:
    return tuple(
        _attribute(node, name, path)
        for name in ("comm_src", "comm_dst", "comm_tag", "comm_size")
    )


def validate_et_group(paths: Iterable[str | Path]) -> ETValidationReport:
    """Validate local graphs and SEND/RECV pairing across one workload group."""
    resolved_paths = [Path(path) for path in paths]
    if not resolved_paths:
        raise ETValidationError("execution-trace group is empty")

    sends: Counter[tuple[object, ...]] = Counter()
    recvs: Counter[tuple[object, ...]] = Counter()
    node_count = 0
    for path in resolved_paths:
        nodes = _read_nodes(path)
        _validate_local_graph(path, nodes)
        node_count += len(nodes)
        for node in nodes:
            if node.type == COMM_SEND_NODE:
                sends[_p2p_key(path, node)] += 1
            elif node.type == COMM_RECV_NODE:
                recvs[_p2p_key(path, node)] += 1

    unmatched_sends = sends - recvs
    unmatched_recvs = recvs - sends
    if unmatched_sends or unmatched_recvs:
        send_example = next(iter(unmatched_sends), None)
        recv_example = next(iter(unmatched_recvs), None)
        raise ETValidationError(
            "execution-trace group has unpaired SEND/RECV nodes: "
            f"unmatched_sends={sum(unmatched_sends.values())}, "
            f"unmatched_recvs={sum(unmatched_recvs.values())}, "
            f"send_example={send_example}, recv_example={recv_example}"
        )

    return ETValidationReport(
        file_count=len(resolved_paths),
        node_count=node_count,
        send_count=sum(sends.values()),
        recv_count=sum(recvs.values()),
    )
