"""Strict parser for the memory-events-v1 Chakra input sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


SCHEMA_VERSION = "memory-events-v1"
_PATH_RESOURCES = {
    "base_die_local": (
        "lpddr_read",
        "base_die_dma",
        "local_stack_fabric",
        "hbm_write",
    ),
    "gpu_routed": (
        "lpddr_read",
        "base_to_gpu_link",
        "gpu_dma",
        "gpu_to_base_link",
        "hbm_write",
    ),
}
_PRIORITIES = {
    "decode_critical",
    "prefill_critical",
    "demand",
    "background_fill",
    "demote",
}


def _closed_object(payload, allowed, context):
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be an object")
    unknown = set(payload) - set(allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {sorted(unknown)}")


def _positive_int(value, context):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _non_negative_int(value, context):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _unique_strings(value, context, allow_empty=False):
    if not isinstance(value, list) or (
        not allow_empty and not value
    ) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{context} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{context} must not contain duplicates")
    return tuple(value)


@dataclass(frozen=True)
class Endpoint:
    tier_id: int
    device_id: int


@dataclass(frozen=True)
class MovementEvent:
    event_id: str
    page_id: str | None
    transaction_id: str | None
    expected_residency_version: int | None
    source_iteration_id: int
    npu_id: int
    kind: str
    phase: str
    source: Endpoint
    destination: Endpoint
    bytes: int
    priority_class: str
    depends_on: tuple[str, ...]
    releases: tuple[str, ...]


class MemoryEvents:
    """Validated, immutable movement sidecar used by the LLM converter."""

    def __init__(self, filename: str, expected_manifest_digest: str):
        try:
            payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read memory events {filename!r}: {error}") from error
        _closed_object(
            payload,
            {
                "schema_version",
                "run_id",
                "instance_id",
                "manifest_digest",
                "selected_path",
                "events",
            },
            "memory events",
        )
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"memory events schema_version must be {SCHEMA_VERSION}")
        for identity in ("run_id", "instance_id", "manifest_digest"):
            if not isinstance(payload.get(identity), str) or not payload[identity]:
                raise ValueError(f"memory events {identity} must be a non-empty string")
        if payload["manifest_digest"] != expected_manifest_digest:
            raise ValueError("memory events manifest_digest does not match Tier Manifest")

        selected = payload.get("selected_path")
        _closed_object(
            selected,
            {
                "id",
                "engine_count",
                "max_priority_burst",
                "max_in_flight_page_movements",
                "resource_ids",
            },
            "selected_path",
        )
        self.path_id = selected.get("id")
        if self.path_id not in _PATH_RESOURCES:
            raise ValueError("selected_path.id must be base_die_local or gpu_routed")
        self.engine_count = _positive_int(
            selected.get("engine_count"), "selected_path.engine_count"
        )
        self.max_priority_burst = _positive_int(
            selected.get("max_priority_burst"),
            "selected_path.max_priority_burst",
        )
        self.max_in_flight_page_movements = _positive_int(
            selected.get("max_in_flight_page_movements"),
            "selected_path.max_in_flight_page_movements",
        )
        if self.max_in_flight_page_movements > self.engine_count:
            raise ValueError(
                "selected_path.max_in_flight_page_movements must not exceed "
                "engine_count"
            )
        self.resource_ids = _unique_strings(
            selected.get("resource_ids"), "selected_path.resource_ids"
        )
        for required in _PATH_RESOURCES[self.path_id]:
            if not any(required in resource for resource in self.resource_ids):
                raise ValueError(
                    f"selected_path.resource_ids is missing {required!r}"
                )

        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("memory events events must be an array")
        events = []
        seen = set()
        for index, raw in enumerate(raw_events):
            context = f"events[{index}]"
            _closed_object(
                raw,
                {
                    "event_id",
                    "page_id",
                    "transaction_id",
                    "expected_residency_version",
                    "source_iteration_id",
                    "npu_id",
                    "kind",
                    "phase",
                    "source",
                    "destination",
                    "bytes",
                    "priority_class",
                    "depends_on",
                    "releases",
                },
                context,
            )
            event_id = raw.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError(f"{context}.event_id must be a non-empty string")
            if event_id in seen:
                raise ValueError(f"duplicate movement event_id {event_id!r}")
            seen.add(event_id)
            source = self._endpoint(raw.get("source"), f"{context}.source")
            destination = self._endpoint(
                raw.get("destination"), f"{context}.destination"
            )
            if source.device_id != destination.device_id:
                raise ValueError(f"{context} violates paired HBM destination")
            kind = raw.get("kind")
            if kind not in {"load", "store", "page_promote", "page_demote"}:
                raise ValueError(f"{context}.kind is unsupported")
            page_id = raw.get("page_id")
            transaction_id = raw.get("transaction_id")
            expected_version = raw.get("expected_residency_version")
            if kind in {"page_promote", "page_demote"}:
                if not isinstance(page_id, str) or not page_id:
                    raise ValueError(f"{context}.page_id must be non-empty")
                if not isinstance(transaction_id, str) or not transaction_id:
                    raise ValueError(f"{context}.transaction_id must be non-empty")
                expected_version = _non_negative_int(
                    expected_version, f"{context}.expected_residency_version"
                )
            elif any(
                value is not None
                for value in (page_id, transaction_id, expected_version)
            ):
                raise ValueError(
                    f"{context} non-page movement must not carry page identity"
                )
            phase = raw.get("phase")
            if phase not in {"critical_line", "background_fill", "whole_object"}:
                raise ValueError(f"{context}.phase is unsupported")
            priority = raw.get("priority_class")
            if priority not in _PRIORITIES:
                raise ValueError(f"{context}.priority_class is unsupported")
            releases = _unique_strings(
                raw.get("releases"), f"{context}.releases", allow_empty=True
            )
            if phase == "background_fill" and releases:
                raise ValueError("background_fill must not release compute nodes")
            if phase != "background_fill" and not releases:
                raise ValueError("foreground movement must release a compute node")
            events.append(
                MovementEvent(
                    event_id=event_id,
                    page_id=page_id,
                    transaction_id=transaction_id,
                    expected_residency_version=expected_version,
                    source_iteration_id=_non_negative_int(
                        raw.get("source_iteration_id"),
                        f"{context}.source_iteration_id",
                    ),
                    npu_id=_non_negative_int(raw.get("npu_id"), f"{context}.npu_id"),
                    kind=kind,
                    phase=phase,
                    source=source,
                    destination=destination,
                    bytes=_positive_int(raw.get("bytes"), f"{context}.bytes"),
                    priority_class=priority,
                    depends_on=_unique_strings(
                        raw.get("depends_on"),
                        f"{context}.depends_on",
                        allow_empty=True,
                    ),
                    releases=releases,
                )
            )
        self._events = tuple(events)
        self._validate_dependencies()
        self.run_id = payload["run_id"]
        self.instance_id = payload["instance_id"]
        self.manifest_digest = payload["manifest_digest"]

    @staticmethod
    def _endpoint(payload, context):
        _closed_object(payload, {"tier_id", "device_id"}, context)
        return Endpoint(
            tier_id=_non_negative_int(payload.get("tier_id"), f"{context}.tier_id"),
            device_id=_non_negative_int(
                payload.get("device_id"), f"{context}.device_id"
            ),
        )

    def _validate_dependencies(self):
        event_ids = {event.event_id for event in self._events}
        graph = {event.event_id: event.depends_on for event in self._events}
        for event in self._events:
            unknown = set(event.depends_on) - event_ids
            if unknown:
                raise ValueError(
                    f"movement event {event.event_id!r} has dangling dependencies "
                    f"{sorted(unknown)}"
                )
        visiting = set()
        visited = set()

        def visit(event_id):
            if event_id in visiting:
                raise ValueError("memory movement dependency graph contains a cycle")
            if event_id in visited:
                return
            visiting.add(event_id)
            for dependency in graph[event_id]:
                visit(dependency)
            visiting.remove(event_id)
            visited.add(event_id)

        for event_id in graph:
            visit(event_id)

    def events_for_npu(self, npu_id: int):
        return tuple(event for event in self._events if event.npu_id == npu_id)

    @property
    def events(self):
        return self._events
