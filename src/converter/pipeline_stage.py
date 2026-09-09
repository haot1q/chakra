"""Closed local-preparation stage provenance, independent of the Serving package."""

import hashlib
import json

from ...schema.protobuf.et_def_pb2 import AttributeProto
from .service_binding import ServiceBinding


def _fields(raw, names):
    if not isinstance(raw, dict) or set(raw) != set(names.split()):
        raise ValueError("pipeline stage has missing or unknown fields")
    return raw


def _uint(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("pipeline stage numeric field must be uint32")
    return value


def pipeline_stage_metadata(stage: dict, endpoint: dict, rank: int,
                            services: ServiceBinding) -> AttributeProto:
    """Validate layout/rank ownership before attaching its deterministic digest."""
    stage = _fields(stage, "binding owner_identity")
    binding = _fields(stage["binding"], "stage_id layer_start layer_end ranks registry_instance_id page_instance_id")
    identity = _fields(stage["owner_identity"], "backend_instance_id backend_node_id registry_instance_id registry_node_id page_instance_id")
    endpoint = _fields(endpoint, "instance_id first_global_rank model_id kv_dtype block_size_tokens kv_dim num_layers kv_dtype_bytes num_npus tp_size pp_size attention_layout")
    for name in ("instance_id", "first_global_rank"):
        _uint(endpoint[name])
    for name in ("block_size_tokens", "kv_dim", "num_layers", "kv_dtype_bytes", "num_npus", "tp_size", "pp_size"):
        if _uint(endpoint[name]) == 0:
            raise ValueError("pipeline endpoint dimensions must be positive")
    for name in ("stage_id", "layer_start", "layer_end"):
        _uint(binding[name])
    for name in ("backend_instance_id", "backend_node_id"):
        _uint(identity[name])
    for row, names in ((identity, ("registry_instance_id", "registry_node_id", "page_instance_id")),
                       (endpoint, ("model_id", "kv_dtype", "attention_layout"))):
        if any(not isinstance(row[name], str) or not row[name].strip() for name in names):
            raise ValueError("pipeline stage requires explicit original names")
    tp, pp, layers = endpoint["tp_size"], endpoint["pp_size"], endpoint["num_layers"]
    if endpoint["first_global_rank"] + endpoint["num_npus"] > len(services.rank_owners):
        raise ValueError("pipeline endpoint exceeds actual service ranks")
    if (endpoint["num_npus"] != tp * pp or layers < pp or binding["stage_id"] >= pp
            or identity["backend_instance_id"] != endpoint["instance_id"]
            or any(binding[name] != identity[name] for name in ("registry_instance_id", "page_instance_id"))):
        raise ValueError("pipeline stage differs from endpoint/original owner")
    first = endpoint["first_global_rank"] + binding["stage_id"] * tp
    remainder = layers % pp
    first_extra = pp - remainder - 1
    stage_id = binding["stage_id"]
    layer_start = stage_id * (layers // pp) + max(0, min(stage_id - first_extra, remainder))
    layer_end = (stage_id + 1) * (layers // pp) + max(0, min(stage_id + 1 - first_extra, remainder))
    ranks = binding["ranks"]
    if (not isinstance(ranks, list) or any(type(item) is not int for item in ranks)
            or ranks != list(range(first, first + tp)) or _uint(rank) not in ranks
            or binding["layer_start"] != layer_start or binding["layer_end"] != layer_end):
        raise ValueError("pipeline stage layers/ranks do not match actual output")
    expected = set(range(endpoint["first_global_rank"], endpoint["first_global_rank"] + tp * pp))
    observed = {item for item, owner in services.rank_owners.items() if owner[0] == endpoint["instance_id"]}
    if expected != observed or any(services.rank_owners[item][1] != identity["backend_node_id"] for item in expected):
        raise ValueError("pipeline endpoint differs from actual service ownership")
    body = json.dumps({"pipeline_stage": stage, "endpoint": endpoint},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return AttributeProto(name="pipeline_stage_digest", string_val="sha256:" + hashlib.sha256(body).hexdigest())
