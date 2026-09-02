"""Strict pd-kv-transfer-v1 sidecar to paired Chakra P2P ETs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ...schema.protobuf.et_def_pb2 import (
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    AttributeProto as ChakraAttr,
    GlobalMetadata,
    Node,
)
from ..third_party.utils.protolib import encodeMessage as encode_message
from .et_validator import validate_et_group


DESCRIPTOR_FIELDS = {
    "run_id", "attempt_id", "sequence", "transfer_id", "request_id",
    "source_instance_id", "destination_instance_id", "compatibility_profile",
    "layout_digest", "initialized_tokens", "initialized_bytes_per_rank",
    "allocated_bytes_per_rank", "charged_bytes_per_rank",
    "charged_bytes_aggregate", "blocks", "rank_pairs", "artifact_identity",
    "schema_version", "transfer_basis", "transport", "evidence_tier",
}
RANK_PAIR_FIELDS = {
    "source_rank", "destination_rank", "tp_rank", "pp_stage", "layer_start",
    "layer_end", "charged_bytes", "tag",
}
BLOCK_FIELDS = {
    "request_id", "logical_block_index", "token_start", "token_capacity",
    "initialized_tokens", "allocated_bytes", "geometry_digest",
    "layout_provenance",
}


def _digest(descriptor: dict[str, object]) -> str:
    encoded = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _node(pair: dict[str, int], *, send: bool, transfer_id: str,
          descriptor_digest: str) -> Node:
    node = Node()
    node.id = 1
    node.name = (
        f"PD_KV_{'SEND' if send else 'RECV'}_"
        f"{pair['source_rank']}_{pair['destination_rank']}"
    )
    node.type = COMM_SEND_NODE if send else COMM_RECV_NODE
    node.attr.extend((
        ChakraAttr(name="comm_type", int64_val=0),
        ChakraAttr(name="comm_src", int32_val=pair["source_rank"]),
        ChakraAttr(name="comm_dst", int32_val=pair["destination_rank"]),
        ChakraAttr(name="comm_size", int64_val=pair["charged_bytes"]),
        ChakraAttr(name="comm_tag", int32_val=pair["tag"]),
        ChakraAttr(name="pd_kv_transfer_id", string_val=transfer_id),
        ChakraAttr(name="pd_kv_descriptor_digest", string_val=descriptor_digest),
    ))
    return node


def convert_pd_kv_sidecar(input_path: str, output_prefix: str) -> list[Path]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "descriptor", "descriptor_digest"
    }:
        raise ValueError("P/D sidecar fields do not match v1 contract")
    descriptor = payload["descriptor"]
    if not isinstance(descriptor, dict) or set(descriptor) != DESCRIPTOR_FIELDS:
        raise ValueError("P/D descriptor fields do not match v1 contract")
    if descriptor["schema_version"] != "pd-kv-transfer-v1":
        raise ValueError("unsupported P/D descriptor schema")
    if descriptor["transport"] != "astra_analytical_p2p_v1":
        raise ValueError("unsupported P/D transport")
    if payload["descriptor_digest"] != _digest(descriptor):
        raise ValueError("P/D descriptor digest mismatch")
    rank_pairs = descriptor["rank_pairs"]
    if not isinstance(rank_pairs, list) or not rank_pairs:
        raise ValueError("P/D rank_pairs must be a non-empty array")
    blocks = descriptor["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("P/D blocks must be a non-empty array")
    allocated_sum = 0
    for block in blocks:
        if not isinstance(block, dict) or set(block) != BLOCK_FIELDS:
            raise ValueError("P/D block fields do not match v1 contract")
        if block["request_id"] != descriptor["request_id"]:
            raise ValueError("P/D block request_id mismatch")
        for key in (
            "logical_block_index", "token_start", "token_capacity",
            "initialized_tokens", "allocated_bytes",
        ):
            value = block[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"P/D block {key} must be non-negative")
        if block["token_capacity"] == 0 or block["initialized_tokens"] == 0:
            raise ValueError("P/D block token counts must be positive")
        if block["initialized_tokens"] > block["token_capacity"]:
            raise ValueError("P/D initialized_tokens exceeds token_capacity")
        if block["allocated_bytes"] == 0:
            raise ValueError("P/D block allocated_bytes must be positive")
        if not isinstance(block["geometry_digest"], str) or not block[
            "geometry_digest"
        ].startswith("sha256:"):
            raise ValueError("P/D block geometry_digest must be a SHA256 identity")
        if not isinstance(block["layout_provenance"], str) or not block[
            "layout_provenance"
        ]:
            raise ValueError("P/D block layout_provenance must be non-empty")
        allocated_sum += block["allocated_bytes"]
    if allocated_sum != descriptor["allocated_bytes_per_rank"]:
        raise ValueError("P/D block bytes differ from per-rank allocation")

    paths = []
    seen_ranks = set()
    seen_tags = set()
    byte_sum = 0
    metadata = GlobalMetadata(attr=(
        ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
        ChakraAttr(name="pd_kv_schema", string_val="pd-kv-transfer-v1"),
    ))
    for raw_pair in rank_pairs:
        if not isinstance(raw_pair, dict) or set(raw_pair) != RANK_PAIR_FIELDS:
            raise ValueError("P/D rank-pair fields do not match v1 contract")
        pair = {key: raw_pair[key] for key in RANK_PAIR_FIELDS}
        for key, value in pair.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"P/D rank-pair {key} must be non-negative")
        if pair["charged_bytes"] == 0:
            raise ValueError("P/D charged_bytes must be positive")
        if pair["tag"] in seen_tags:
            raise ValueError("P/D rank-pair tags must be unique")
        seen_tags.add(pair["tag"])
        byte_sum += pair["charged_bytes"]

        for rank, send in (
            (pair["source_rank"], True),
            (pair["destination_rank"], False),
        ):
            if rank in seen_ranks:
                raise ValueError("v1 requires one P/D endpoint per global rank")
            seen_ranks.add(rank)
            path = Path(f"{output_prefix}.{rank}.et")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                encode_message(stream, metadata)
                encode_message(stream, _node(
                    pair,
                    send=send,
                    transfer_id=descriptor["transfer_id"],
                    descriptor_digest=payload["descriptor_digest"],
                ))
            paths.append(path)
    if byte_sum != descriptor["charged_bytes_aggregate"]:
        raise ValueError("P/D rank-pair bytes differ from aggregate")
    validate_et_group(paths)
    return paths


__all__ = ["convert_pd_kv_sidecar"]
