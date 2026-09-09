"""Validate ADR-0041 wire identity before attaching it to individual rank ETs.

Deployment reachability and complete physical queue validation remain owned by
the producer and ASTRA factory. This boundary checks exact document structure,
canonical digest, activation, Manifest identity and emitted rank membership.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from ...schema.protobuf.et_def_pb2 import AttributeProto


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate physical service JSON key {key}")
        result[key] = value
    return result


def _object(raw: object, expected: set[str]) -> dict[str, object]:
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"physical services: expected exactly {sorted(expected)}")
    return raw


def _name(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        raise ValueError("physical service identifier must be nonempty ASCII")
    return value


def _uint(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError("physical service numeric identifier must be uint32")
    return value


def _rows(
    raw: object, fields: set[str], integers: set[str], key: tuple[str, ...],
) -> list[dict[str, object]]:
    if not isinstance(raw, list):
        raise ValueError("physical service table must be array")
    seen = set()
    for item in raw:
        row = _object(item, fields)
        for name, value in row.items():
            if name == "memory_endpoints":
                if not isinstance(value, list):
                    raise ValueError("memory_endpoints must be array")
                for endpoint in value:
                    _name(endpoint)
                if len(set(value)) != len(value):
                    raise ValueError("duplicate memory endpoint")
                row[name] = sorted(value)
            elif name in integers:
                _uint(value)
            else:
                _name(value)
        identity = tuple(row[field] for field in key)
        if identity in seen:
            raise ValueError("duplicate physical service table entry")
        seen.add(identity)
    return sorted(raw, key=lambda row: tuple(row[field] for field in key))


class ServiceBinding:
    """Checked binding metadata; not a capacity owner or a physical service queue."""

    def __init__(self, path: str | Path, manifest_digest: str):
        with Path(path).open(encoding="utf-8") as stream:
            raw = json.load(stream, object_pairs_hook=_pairs)
        body = _object(raw, {"schema_version", "tier_manifest_digest", "ranks",
                             "resources", "bindings", "binding_digest", "activation_id"})
        if (body["schema_version"] != "physical-service-binding-v1"
                or body["tier_manifest_digest"] != manifest_digest):
            raise ValueError("physical service schema/Manifest mismatch")
        self.activation = _name(body.pop("activation_id"))
        self.digest = _name(body.pop("binding_digest"))
        body["ranks"] = _rows(body["ranks"],
            {"rank", "instance_id", "node_id", "accelerator_id"},
            {"rank", "instance_id", "node_id"}, ("rank",))
        body["resources"] = _rows(body["resources"],
            {"id", "kind", "owner_kind", "owner_id", "resource_key", "physical_device_id", "memory_endpoints"},
            {"physical_device_id"}, ("id",))
        body["bindings"] = _rows(body["bindings"],
            {"rank", "kind", "logical_ref", "logical_device_id", "physical_resource_ref"},
            {"rank", "logical_device_id"}, ("rank", "kind", "logical_ref", "logical_device_id"))
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        if self.digest != "sha256:" + hashlib.sha256(canonical).hexdigest():
            raise ValueError("physical service binding content digest mismatch")
        self.ranks = frozenset(row["rank"] for row in body["ranks"])
        self.rank_owners = {row["rank"]: (row["instance_id"], row["node_id"])
                            for row in body["ranks"]}
        if not self.ranks or self.ranks != frozenset(range(len(self.ranks))):
            raise ValueError("physical service ranks must be complete and contiguous")

    def validate_trace(self, header: dict[str, str]) -> None:
        if (header.get("service_binding_digest") != self.digest
                or header.get("service_activation_id") != self.activation):
            raise ValueError("Trace physical service binding/activation mismatch")

    def metadata(self, rank: int) -> list[AttributeProto]:
        if _uint(rank) not in self.ranks:
            raise ValueError(f"ET rank {rank} is not in physical service bindings")
        return [
            AttributeProto(name="service_binding_digest", string_val=self.digest),
            AttributeProto(name="service_activation_id", string_val=self.activation),
            AttributeProto(name="service_rank", uint64_val=rank),
        ]
