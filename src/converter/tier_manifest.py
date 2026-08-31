"""Validation for the native N-tier runtime manifest.

This module intentionally mirrors the small cross-language contract owned by
the outer simulator.  It recomputes the digest instead of trusting producer
metadata, so a stale manifest cannot silently route an execution trace.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping


SCHEMA_VERSION = "memory-tier-runtime-v1"
ID_MODE = "native"
NATIVE_TIER_ID_START = 16
_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _reject_float(value: object, path: str = "manifest") -> None:
    if isinstance(value, float):
        raise ValueError(f"{path} must not contain floating-point values")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} object keys must be strings")
            _reject_float(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_float(child, f"{path}[{index}]")


def canonical_manifest_digest(payload: Mapping[str, object]) -> str:
    body = dict(payload)
    body.pop("manifest_digest", None)
    _reject_float(body)
    canonical = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _parse_ucie_peers(
    raw_links: object, tiers: Mapping[str, tuple[int, int]]
) -> dict[str, str]:
    """Map hot-tier names to opt-in UCIe link ids. Absent links emit nothing."""

    if raw_links is None:
        return {}
    if not isinstance(raw_links, list) or not raw_links:
        raise ValueError("ucie_links must be a non-empty array when present")
    peers: dict[str, str] = {}
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_links):
        if not isinstance(raw, Mapping):
            raise ValueError(f"ucie_links[{index}] must be an object")
        link_id = raw.get("id")
        endpoints = raw.get("endpoints")
        if not isinstance(link_id, str) or not link_id:
            raise ValueError(f"ucie_links[{index}].id must be a non-empty string")
        if link_id in seen_ids or link_id in tiers:
            raise ValueError(f"ucie_links[{index}].id collides or is duplicate")
        if (
            not isinstance(endpoints, list)
            or len(endpoints) != 2
            or any(not isinstance(item, str) or not item for item in endpoints)
        ):
            raise ValueError(f"ucie_links[{index}].endpoints must be two strings")
        if "compute" not in endpoints:
            raise ValueError(f"ucie_links[{index}] must include the compute endpoint")
        peer = next(item for item in endpoints if item != "compute")
        if peer not in tiers:
            raise ValueError(f"ucie_links[{index}] peer {peer!r} is not a tier")
        if peer in peers:
            raise ValueError(f"ucie_links[{index}] peer {peer!r} already has a link")
        seen_ids.add(link_id)
        peers[peer] = link_id
    return peers


class TierManifest:
    """Validated tier/device lookup used by the LLM converter."""

    def __init__(self, path: str | Path):
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("tier manifest must be a JSON object")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        if payload.get("id_mode") != ID_MODE:
            raise ValueError(f"id_mode must be {ID_MODE!r}")
        tiers = payload.get("tiers")
        if not isinstance(tiers, list) or not tiers:
            raise ValueError("tiers must be a non-empty array")

        self._tiers: dict[str, tuple[int, int]] = {}
        seen_ids: set[int] = set()
        for index, tier in enumerate(tiers):
            if not isinstance(tier, dict):
                raise ValueError("each tiers entry must be an object")
            name = tier.get("tier_name")
            tier_id = tier.get("tier_id")
            devices = tier.get("devices")
            num_devices = tier.get("num_devices")
            if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
                raise ValueError(f"invalid tier_name {name!r}")
            if name in self._tiers:
                raise ValueError(f"duplicate tier_name {name!r}")
            if isinstance(tier_id, bool) or not isinstance(tier_id, int):
                raise ValueError(f"tier {name} tier_id must be an integer")
            if tier_id == 0:
                raise ValueError("tier_id 0 is reserved and invalid")
            expected_id = NATIVE_TIER_ID_START + index
            if tier_id != expected_id:
                raise ValueError(
                    f"native tier {name!r} must have tier_id {expected_id}, got {tier_id}"
                )
            if tier_id in seen_ids:
                raise ValueError(f"duplicate tier_id {tier_id}")
            if not isinstance(devices, list) or not devices:
                raise ValueError(f"tier {name} devices must be a non-empty array")
            if num_devices != len(devices):
                raise ValueError(f"tier {name} num_devices does not match devices")
            for device_id, device in enumerate(devices):
                if not isinstance(device, dict) or device.get("device_id") != device_id:
                    raise ValueError(f"tier {name} device IDs must be contiguous from zero")
            self._tiers[name] = (tier_id, num_devices)
            seen_ids.add(tier_id)
        if list(self._tiers) != sorted(self._tiers):
            raise ValueError("native tiers must be sorted by tier_name")

        digest = payload.get("manifest_digest")
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise ValueError("manifest_digest must be sha256:<64 lowercase hex>")
        expected_digest = canonical_manifest_digest(payload)
        if digest != expected_digest:
            raise ValueError(
                f"manifest_digest mismatch: declared={digest}, expected={expected_digest}"
            )
        self.digest = digest
        self._ucie_by_peer = _parse_ucie_peers(payload.get("ucie_links"), self._tiers)

    def ucie_link_id(self, location: str) -> str | None:
        name = location.split(":", 1)[0]
        return self._ucie_by_peer.get(name)

    def resolve(self, location: str) -> tuple[int, int]:
        if not isinstance(location, str) or not location:
            raise ValueError("tier location must be a non-empty tier[:device] string")
        name, separator, raw_device_id = location.partition(":")
        tier = self._tiers.get(name)
        if tier is None:
            raise KeyError(f"unknown tier {name!r}")
        if separator:
            if not raw_device_id.isdigit():
                raise ValueError(f"invalid device in tier location {location!r}")
            device_id = int(raw_device_id)
        else:
            device_id = 0
        tier_id, num_devices = tier
        if device_id >= num_devices:
            raise IndexError(
                f"tier {name!r} device_id {device_id} is out of range [0, {num_devices})"
            )
        return tier_id, device_id
