"""Strict parser for the memory-events-v1 Chakra input sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal


SCHEMA_VERSION = "memory-events-v1"
PATH_SCHEMA_VERSION = "movement-path-v1"
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
class PathSegment:
    id: str
    kind: str
    resource_ref: str
    operation: str
    byte_rule: str

    @property
    def bill_id(self):
        return f"{self.id}:{self.resource_ref}"


@dataclass(frozen=True)
class MovementEvent:
    event_id: str
    page_id: str | None
    transaction_id: str | None
    expected_residency_version: int | None
    home_domain_id: int | None
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


@dataclass(frozen=True)
class PriorMovement:
    event_id: str
    npu_id: int
    source_iteration_id: int
    releases: tuple[str, ...]


class MemoryEvents:
    """Validated, immutable movement sidecar used by the LLM converter."""

    def __init__(
        self, filename: str, expected_manifest_digest: str, *,
        completion_owner: Literal["workload", "external_preparation"] = "workload",
    ):
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
                "completion_owner",
                "prior_events",
                "source_iteration_id",
            },
            "memory events",
        )
        if (completion_owner not in {"workload", "external_preparation"}
                or payload.get("completion_owner", "workload") != completion_owner):
            raise ValueError("memory events completion owner does not match consumer")
        self.completion_owner = completion_owner
        self.schema_version = payload.get("schema_version")
        if self.schema_version not in {SCHEMA_VERSION, "memory-events-v2"}:
            raise ValueError("unsupported memory events schema_version")
        self.prior_events = self._read_prior_events(payload)
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
                "contract_status",
                "schema_version",
                "timing_provenance",
                "segments",
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
        self.path_contract_status = selected.get(
            "contract_status", "compatibility_checkpoint"
        )
        if self.path_contract_status == "implemented":
            if selected.get("schema_version") != PATH_SCHEMA_VERSION:
                raise ValueError(
                    f"selected_path.schema_version must be {PATH_SCHEMA_VERSION}"
                )
            self.path_schema_version = PATH_SCHEMA_VERSION
            self.path_timing_provenance = selected.get("timing_provenance")
            if self.path_timing_provenance not in {"estimated", "measured"}:
                raise ValueError(
                    "selected_path.timing_provenance must be estimated or measured"
                )
            raw_segments = selected.get("segments")
            if not isinstance(raw_segments, list) or not raw_segments:
                raise ValueError("selected_path.segments must be a non-empty array")
            segments = []
            for index, raw in enumerate(raw_segments):
                context = f"selected_path.segments[{index}]"
                _closed_object(
                    raw,
                    {"id", "kind", "resource_ref", "operation", "byte_rule"},
                    context,
                )
                for field in ("id", "resource_ref"):
                    if not isinstance(raw.get(field), str) or not raw[field]:
                        raise ValueError(f"{context}.{field} must be non-empty")
                if raw.get("kind") not in {
                    "bandwidth_resource",
                    "ucie_transaction",
                }:
                    raise ValueError(f"{context}.kind is unsupported")
                if raw.get("operation") not in {"read", "write"}:
                    raise ValueError(f"{context}.operation is unsupported")
                if raw.get("byte_rule") != "payload":
                    raise ValueError(f"{context}.byte_rule must be payload")
                segments.append(
                    PathSegment(
                        raw["id"],
                        raw["kind"],
                        raw["resource_ref"],
                        raw["operation"],
                        raw["byte_rule"],
                    )
                )
            if len({segment.id for segment in segments}) != len(segments):
                raise ValueError("selected_path segment IDs must be unique")
            shape = tuple((item.kind, item.operation) for item in segments)
            expected_shape = (
                (
                    ("bandwidth_resource", "read"),
                    ("bandwidth_resource", "write"),
                )
                if self.path_id == "base_die_local"
                else (
                    ("ucie_transaction", "read"),
                    ("bandwidth_resource", "write"),
                    ("ucie_transaction", "write"),
                )
            )
            if shape != expected_shape:
                raise ValueError(
                    f"selected_path {self.path_id} segments have invalid shape"
                )
            self.path_segments = tuple(segments)
            expected_resources = (
                "lpddr_read",
                *(segment.bill_id for segment in self.path_segments),
                "hbm_write",
            )
            if self.resource_ids != expected_resources:
                raise ValueError(
                    "selected_path.resource_ids must match ordered segments"
                )
        elif self.path_contract_status == "compatibility_checkpoint":
            if any(
                selected.get(field) is not None
                for field in ("schema_version", "timing_provenance", "segments")
            ):
                raise ValueError(
                    "compatibility checkpoint must not claim implemented path fields"
                )
            self.path_schema_version = "adr-0020-checkpoint"
            self.path_timing_provenance = "proxy_unimplemented"
            self.path_segments = ()
            for required in _PATH_RESOURCES[self.path_id]:
                if not any(required in resource for resource in self.resource_ids):
                    raise ValueError(
                        f"selected_path.resource_ids is missing {required!r}"
                    )
        else:
            raise ValueError(
                "selected_path.contract_status must be implemented or "
                "compatibility_checkpoint"
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
                    "home_domain_id",
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
            home_domain_id = raw.get("home_domain_id")
            if kind in {"page_promote", "page_demote"}:
                if not isinstance(page_id, str) or not page_id:
                    raise ValueError(f"{context}.page_id must be non-empty")
                if not isinstance(transaction_id, str) or not transaction_id:
                    raise ValueError(f"{context}.transaction_id must be non-empty")
                expected_version = _non_negative_int(
                    expected_version, f"{context}.expected_residency_version"
                )
                home_domain_id = _non_negative_int(
                    home_domain_id, f"{context}.home_domain_id"
                )
                if home_domain_id != source.device_id:
                    raise ValueError(
                        f"{context}.home_domain_id must match paired device_id"
                    )
            elif any(
                value is not None
                for value in (
                    page_id,
                    transaction_id,
                    expected_version,
                    home_domain_id,
                )
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
            if completion_owner == "external_preparation" and releases:
                raise ValueError("external preparation must not release compute nodes")
            if phase == "background_fill" and releases:
                raise ValueError("background_fill must not release compute nodes")
            if (completion_owner == "workload"
                    and phase != "background_fill" and not releases):
                raise ValueError("foreground movement must release a compute node")
            events.append(
                MovementEvent(
                    event_id=event_id,
                    page_id=page_id,
                    transaction_id=transaction_id,
                    expected_residency_version=expected_version,
                    home_domain_id=home_domain_id,
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
        prior = {event.event_id: event for event in self.prior_events}
        if event_ids & prior.keys():
            raise ValueError("prior movement must not be resubmitted")
        used_prior = {event.event_id for event in self.prior_events if event.releases}
        graph = {event.event_id: event.depends_on for event in self._events}
        for event in self._events:
            if self.prior_events and event.source_iteration_id != self.source_iteration_id:
                raise ValueError("v2 events must belong to the declared current iteration")
            for dependency in set(event.depends_on) & prior.keys():
                if prior[dependency].npu_id != event.npu_id:
                    raise ValueError("prior movement dependency belongs to another rank")
                used_prior.add(dependency)
            unknown = set(event.depends_on) - event_ids - prior.keys()
            if unknown:
                raise ValueError(
                    f"movement event {event.event_id!r} has dangling dependencies "
                    f"{sorted(unknown)}"
                )
        visiting = set()
        visited = set()

        def visit(event_id):
            if event_id in prior:
                return
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
        if used_prior != set(prior):
            raise ValueError("unused prior movement declaration")

    def _read_prior_events(self, payload):
        if self.schema_version == SCHEMA_VERSION:
            if "prior_events" in payload or "source_iteration_id" in payload:
                raise ValueError("prior movement declarations require memory-events-v2")
            self.source_iteration_id = None
            return ()
        if self.completion_owner != "workload":
            raise ValueError("prior movement references require an ordinary workload")
        self.source_iteration_id = _non_negative_int(payload.get("source_iteration_id"), "current iteration")
        raw_events = payload.get("prior_events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError("memory-events-v2 requires nonempty prior events")
        result, seen = [], set()
        for raw in raw_events:
            if not isinstance(raw, dict) or set(raw) != {
                "event_id", "npu_id", "source_iteration_id", "releases"
            }:
                raise ValueError("prior movement fields differ from v2")
            event_id = raw["event_id"]
            if not isinstance(event_id, str) or not event_id or event_id in seen:
                raise ValueError("prior movement ID is missing or duplicate")
            rank = _non_negative_int(raw["npu_id"], "prior rank")
            iteration = _non_negative_int(raw["source_iteration_id"], "prior iteration")
            if iteration >= self.source_iteration_id:
                raise ValueError("prior movement must belong to an earlier iteration")
            result.append(PriorMovement(event_id, rank, iteration,
                                        _unique_strings(raw["releases"], "prior releases", allow_empty=True)))
            seen.add(event_id)
        return tuple(result)

    def events_for_npu(self, npu_id: int):
        return tuple(event for event in self._events if event.npu_id == npu_id)

    @property
    def events(self):
        return self._events
