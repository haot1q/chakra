"""Write strict standalone native memory traces for external preparation owners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..third_party.utils.protolib import encodeMessage as encode_message
from .llm_converter import LLMConverter
from .memory_events import MemoryEvents
from .pipeline_stage import pipeline_stage_metadata

# Protocol record bound shared with MemoryPreparationTrace, not a page size.
MAX_PREPARATION_RECORD_BYTES = 64 * 1024


@dataclass(frozen=True)
class PreparationTraceInputs:
    """Existing manifest, actual physical services, and explicit preparation work."""

    tier_manifest: Path
    physical_services: Path
    memory_events: Path
    pipeline_stage: dict[str, object] | None = None
    endpoint: dict[str, object] | None = None


def write_preparation_trace(
    inputs: PreparationTraceInputs, output: Path, *, rank: int,
) -> tuple[str, ...]:
    """Emit one actual rank's uncompressed ET without synthetic compute nodes.

    All input checks and record bounds precede opening output. The caller owns
    a fresh external artifact directory; existing output is never overwritten.
    Returns exactly the physical event IDs for the preparation IPC mapping.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < 2**32:
        raise ValueError("preparation rank must be uint32")
    converter = LLMConverter(
        str(inputs.memory_events), str(output), 1, npu_offset=rank,
        tier_manifest=str(inputs.tier_manifest),
    )
    converter.configure_physical_services(str(inputs.physical_services))
    converter.native_trace = True
    events = MemoryEvents(
        str(inputs.memory_events), converter.manifest.digest,
        completion_owner="external_preparation",
    )
    selected = events.events_for_npu(rank)
    if not selected or len(selected) != len(events.events):
        raise ValueError("preparation sidecar must contain exactly one actual rank")
    converter.movement_events = events
    nodes = converter.get_memory_movement_nodes(rank)
    records = [converter.get_global_metadata(rank)]
    if (inputs.pipeline_stage is None) != (inputs.endpoint is None):
        raise ValueError("preparation stage and endpoint must be supplied together")
    if inputs.pipeline_stage is not None:
        records[0].attr.append(pipeline_stage_metadata(
            inputs.pipeline_stage, inputs.endpoint, rank, converter.services,
        ))
    records.extend(node for _, node in nodes.values())
    if any(not 0 < record.ByteSize() <= MAX_PREPARATION_RECORD_BYTES
           for record in records):
        raise ValueError("preparation record exceeds protocol size bound")
    with output.open("xb") as stream:
        for record in records:
            encode_message(stream, record)
    return tuple(nodes)
