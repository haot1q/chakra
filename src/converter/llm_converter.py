from __future__ import annotations

import logging
from io import TextIOWrapper
from pathlib import Path
from typing import Any, List
from enum import Enum

from ...schema.protobuf.et_def_pb2 import *
from ...schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr
from ..third_party.utils.protolib import encodeMessage as encode_message
from .et_validator import validate_et_group
from .memory_events import MemoryEvents
from .tier_manifest import TierManifest
from .service_binding import ServiceBinding


def _parse_trace_header(line: str) -> tuple[str, dict[str, str]]:
    fields = line.strip().split()
    if not fields:
        raise ValueError("Trace header is empty")
    if len(fields[1:]) % 2 != 0:
        raise ValueError("Trace header must contain key: value pairs")

    header = {}
    for key, value in zip(fields[1::2], fields[2::2]):
        if not key.endswith(":"):
            raise ValueError(f"Malformed Trace header key {key!r}")
        name = key[:-1]
        if name in header:
            raise ValueError(f"Duplicate Trace header key {name!r}")
        header[name] = value
    return fields[0], header


# Memory type is deprecated for latest version of chakra & astra-sim
# It only uses tensor_size for the remote memory issue in Workload.cc
# Added memory types using enum and add tensor_loc in node attribute in et_feeder_node.cpp
class MemoryType(Enum):
    INVALID_MEMORY = 0
    LOCAL_MEMORY = 1
    REMOTE_MEMORY = 2
    CXL_MEMORY = 3
    STORAGE_MEMORY = 4


class Layer:
    def __init__(
        self,
        line: str | None = None,
        manifest: TierManifest | None = None,
        cols: List[str] | None = None,
    ):
        try:
            # ``cols`` lets a caller that already holds the fields skip the
            # format-and-resplit round trip; ``line`` is the text path.
            col = cols if cols is not None else line.strip().split()
            if col[0] == 'EXPERT': # If Expert Flag
                self.name = col[0]
                self.expert_num = col[1]
                self.comm_type, self.involved_dim = self._parse_comm_type(str(col[2]) if len(col) > 2 else "NONE")
                self.comm_size = int(col[3]) if len(col) > 3 else 0
                self.is_expert = True
                self.is_pim = False
                self.comm_node = None
                self.comp_node = None
            elif col[0] == 'PIM': # If PIM Flag
                self.name = col[0]
                self.pim_num = col[1]
                self.is_expert = False
                self.is_pim = True
                self.comm_node = None
                self.comp_node = None
                self.comm_type = "NONE"
                self.involved_dim = None
            else:
                self.is_expert = False
                self.is_pim = False
                self.name = col[0]

                # compuation
                self.comp_time = int(col[1])
                self.comp_node = None

                # memory
                self.input_memory_loc = str(col[2])
                self.input_memory_size = int(col[3])
                self.input_memory_node = None
                self.weight_memory_loc = str(col[4])
                self.weight_memory_size = int(col[5])
                self.weight_memory_node = None
                self.weight_memory_nodes = []
                self.output_memory_loc = str(col[6])
                self.output_memory_size = int(col[7])
                self.output_memory_node = None

                # communication (supports ALLREDUCE:1,0 format for involved_dim)
                self.comm_type, self.involved_dim = self._parse_comm_type(str(col[8]))
                self.comm_size = int(col[9])
                self.comm_node = None

                self.misc = str(col[10])
                self.weight_segments = []
                if manifest is not None:
                    if len(col) != 12:
                        raise ValueError("native Trace rows must contain 12 columns")
                    segment_text = col[11]
                    if segment_text != "-":
                        if self.weight_memory_size != 0:
                            raise ValueError(
                                "native multi-segment rows must set weight_size to 0"
                            )
                        previous_key = None
                        seen_locations = set()
                        for segment in segment_text.split(","):
                            location, separator, raw_size = segment.partition("@")
                            if not separator or not raw_size.isdigit() or int(raw_size) <= 0:
                                raise ValueError(f"invalid weight segment {segment!r}")
                            tier_id, device_id = manifest.resolve(location)
                            key = (tier_id, device_id)
                            if key in seen_locations:
                                raise ValueError(f"duplicate weight segment {location!r}")
                            if previous_key is not None and key <= previous_key:
                                raise ValueError("weight segments must be sorted by tier_id/device_id")
                            self.weight_segments.append((location, int(raw_size)))
                            seen_locations.add(key)
                            previous_key = key
                elif len(col) != 11:
                    raise ValueError("legacy Trace rows must contain 11 columns")
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Cannot parse the following layer -- \"{line}\": {error}"
            ) from error

    @staticmethod
    def _parse_comm_type(s: str):
        """Parse comm_type string, extracting optional involved_dim.

        'ALLREDUCE:1,0' -> ('ALLREDUCE', [True, False])
        'ALLTOALL'      -> ('ALLTOALL', None)
        'NONE'          -> ('NONE', None)
        """
        if ':' in s:
            comm_type, dim_str = s.split(':', 1)
            involved_dim = [v == '1' for v in dim_str.split(',')]
            return comm_type, involved_dim
        return s, None

class LLMConverter:
    def __init__(
        self,
        input_filename: str,
        output_filename: str,
        num_npus: int,
        npu_offset: int = 0,
        local_offloading: bool = False,
        tier_manifest: str | None = None,
        memory_events: str | None = None,
        pd_kv_transfer_mode: str = "legacy",
    ):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.num_npus = num_npus
        self.npu_offset = npu_offset
        self.local_offloading = local_offloading
        if pd_kv_transfer_mode not in {"faithful", "legacy"}:
            raise ValueError("pd_kv_transfer_mode must be faithful or legacy")
        self.pd_kv_transfer_mode = pd_kv_transfer_mode
        self.manifest = TierManifest(tier_manifest) if tier_manifest else None
        self.services: ServiceBinding | None = None
        self._services_frozen = False
        if memory_events and self.manifest is None:
            raise ValueError("--memory-events requires --tier-manifest")
        self.movement_events = (
            MemoryEvents(memory_events, self.manifest.digest)
            if memory_events
            else None
        )
        self._emitted_movement_event_ids: set[str] = set()
        self._emitted_movement_releases: set[tuple[str, str]] = set()
        self.native_trace = False
        self.next_node_id = 0

        # For send & recv nodes
        self.next_comm_tag = 0
        self.comm_tag_dict = dict()

        # Set by convert_rows so get_layers can return them instead of
        # parsing a file. See convert_rows.
        self._layers = None

    def configure_physical_services(self, path: str) -> "LLMConverter":
        """Bind once before emitting any ET; preserve the legacy constructor API."""
        if self._services_frozen or self.services is not None:
            raise ValueError("physical service configuration is already frozen")
        if self.manifest is None:
            raise ValueError("physical service bindings require a native tier manifest")
        self.services = ServiceBinding(path, self.manifest.digest)
        return self

    def get_global_metadata(self, rank: int | None = None):
        self._services_frozen = True
        # ``input_file`` carries the trace's *path*, not its contents.
        #
        # It used to embed the whole trace text. Nothing consumes it:
        # ETFeeder::readGlobalMetadata() reads the message into a local
        # shared_ptr and drops it on the floor, and no other reader touches
        # the attribute. So every byte of it was written by the converter,
        # re-parsed by the feeder's protobuf reader, and discarded.
        #
        # It was not a small overhead. On the swe-bench MoE DP+EP example a
        # trace is ~82 KB of a 117 KB .et -- 70% of the file -- and the
        # simulator generates 8,810 of them for one session, so it was
        # ~720 MB of encode/write/read/parse per run, plus a second full
        # read of the trace file here purely to obtain the text.
        #
        # The path keeps the provenance that made this attribute useful for
        # debugging; the trace itself is written to disk by --save-trace-text.
        attr = [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="input_file", string_val=self.input_filename),
        ]
        if self.native_trace:
            attr.append(
                ChakraAttr(
                    name="tier_manifest_digest",
                    string_val=self.manifest.digest,
                )
            )
        if self.services is not None:
            attr.extend(self.services.metadata(rank))
        metadata = GlobalMetadata(attr=attr)
        return metadata

    def get_layers(self, f: TextIOWrapper) -> List[Layer]:
        if getattr(self, "_layers", None) is not None:
            return self._layers
        layers: List[Layer] = []
        for line in f:
            manifest = getattr(self, "manifest", None)
            native_trace = getattr(self, "native_trace", False)
            layers.append(Layer(line, manifest if native_trace else None))
        return layers

    def convert_rows(self, header_line: str, rows: List[List[str]]) -> None:
        """Convert from pre-parsed field lists, with no text round trip.

        The simulator already holds every field. Formatting them into padded
        columns, writing the file, reading it back and splitting each line
        again is pure overhead now that the converter runs in the same
        process -- 0.65 ms of formatting and writing per batch plus 0.9 ms of
        reading and re-parsing, against 8,810 batches on the swe-bench MoE
        DP+EP example.

        Nothing downstream changes: convert_common, convert_prefill and
        convert_event each touch the file handle exactly once, to call
        get_layers, so seeding the layers is the whole of it. Header parsing
        mirrors convert() on a string instead of a readline, and num_layers
        is len(rows) -- which is precisely what convert() reads off line two,
        since that is what the writer puts there.
        """
        if self.services is not None or self.manifest is not None:
            raise ValueError("native in-memory conversion is unsupported; use validated text conversion")
        self._layers = [Layer(cols=cols) for cols in rows]

        first_line = header_line.strip().split()
        execution_type = first_line[0]

        header = {}
        fields = first_line[1:]
        for i in range(0, len(fields) - 1, 2):
            if fields[i].endswith(":"):
                header[fields[i][:-1]] = fields[i + 1]

        num_npu_group = int(header.get("model_parallel_NPU_group", 0))
        boundary_str = header.get("pp_stage_boundaries", "")
        stage_boundaries = (
            [int(b) for b in boundary_str.split(",")] if boundary_str else []
        )
        num_layers = len(rows)

        if execution_type in ("COLOCATED", "DECODE"):
            if num_npu_group <= 0:
                raise ValueError(f"model_parallel_NPU_group <= 0")
            self.convert_common(None, num_layers, num_npu_group, stage_boundaries)
        elif execution_type == "PREFILL":
            if num_npu_group <= 0:
                raise ValueError(f"model_parallel_NPU_group <= 0")
            if self.pd_kv_transfer_mode == "faithful":
                self.convert_common(None, num_layers, num_npu_group, stage_boundaries)
            else:
                self.convert_prefill(None, num_layers, num_npu_group, stage_boundaries)
        elif execution_type == "EVENT":
            self.convert_event(None, num_layers)
        else:
            raise ValueError(f"Unsupported execution type, {execution_type}")

    def get_next_node_id(self) -> int:
        ret = self.next_node_id
        self.next_node_id += 1
        return ret

    def get_next_comm_tag(self) -> int:
        ret = self.next_comm_tag
        self.next_comm_tag += 1
        return ret

    def get_node(self, name: str, node_type: NodeType) -> Any:
        node = Node()
        node.id = self.get_next_node_id()
        node.name = name
        node.type = node_type
        return node

    def get_comp_node(self, layer_name: str, comp_time: int) -> Any:
        node = self.get_node("COMP_NODE_" + layer_name, COMP_NODE)
        node.duration_micros = comp_time
        return node

    def get_comm_type(self, comm_type: str) -> int:
        if comm_type == "ALLREDUCE":
            return ALL_REDUCE
        elif comm_type == "ALLTOALL":
            return ALL_TO_ALL
        elif comm_type == "ALLGATHER":
            return ALL_GATHER
        elif comm_type == "REDUCESCATTER":
            return REDUCE_SCATTER
        return 0

    def get_comm_coll_node(self, layer_name: str, comm_type: str, comm_size: int,
                           involved_dim: list = None) -> Any:
        node = self.get_node(f"COMM_COLL_NODE_{layer_name}_{comm_type}", COMM_COLL_NODE)
        node.attr.append(ChakraAttr(name="comm_type", int64_val=self.get_comm_type(comm_type)))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
        if involved_dim is not None:
            node.attr.append(ChakraAttr(name="involved_dim",
                                        bool_list=BoolList(values=involved_dim)))
        return node

    def get_comm_node(self, is_send: bool, layer_name: str, comm_type: str, comm_size: int,
                           comm_src: int, comm_dst: int, id: int = 0) -> Any:
        if is_send:
            node = self.get_node(
                    f"COMM_SEND_NODE_{layer_name}_{comm_type}_{comm_src}_{comm_dst}",
                    COMM_SEND_NODE)
        else:
            node = self.get_node(
                    f"COMM_RECV_NODE_{layer_name}_{comm_type}_{comm_src}_{comm_dst}",
                    COMM_RECV_NODE)
        node.attr.append(ChakraAttr(name="comm_type", int64_val=self.get_comm_type(comm_type)))
        node.attr.append(ChakraAttr(name="comm_src", int32_val=comm_src))
        node.attr.append(ChakraAttr(name="comm_dst", int32_val=comm_dst))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=comm_size))
        comm_key = f"{comm_src}_{comm_dst}_{id}"
        if comm_key in self.comm_tag_dict:
            node.attr.append(ChakraAttr(name="comm_tag", int32_val=self.comm_tag_dict[comm_key]))
        else:
            new_tag = self.get_next_comm_tag()
            node.attr.append(ChakraAttr(name="comm_tag", int32_val=new_tag))
            self.comm_tag_dict[comm_key] = new_tag

        # check if SEND/RECV pair have same tags
        # print(f"name: {node.name}, src: {node.comm_src}, dst: {node.comm_dst}, size: {node.comm_size}, key: {comm_key}, tag: {node.comm_tag}")
        return node

    def get_mem_type(self, mem_type: str) -> int:
        if self.native_trace:
            return self.manifest.resolve(mem_type)[0]
        mem_type = mem_type.split(':')[0]  # Exclude the device number if present
        if mem_type == "LOCAL":
            return MemoryType.LOCAL_MEMORY.value
        elif mem_type == "REMOTE":
            return MemoryType.REMOTE_MEMORY.value
        elif mem_type == "CXL":
            return MemoryType.CXL_MEMORY.value
        elif mem_type == "STORAGE":
            return MemoryType.STORAGE_MEMORY.value
        return MemoryType.INVALID_MEMORY.value

    def get_mem_device(self, mem_type: str) -> int:
        """
        Extract the device index from mem_type.

        Supported formats:
        - "REMOTE"          -> returns 0
        - "REMOTE:1"        -> returns 1
        - "REMOTE:1.3"      -> returns 1  (device = 1, channel = 3)
        """
        if self.native_trace:
            return self.manifest.resolve(mem_type)[1]
        parts = mem_type.split(":", 1)
        if len(parts) == 2:
            # parts[1] may be "1" or "1.3"
            dev_part = parts[1].split(".", 1)[0]
            if dev_part.isdigit():
                return int(dev_part)
        return 0  # Default device number


    def get_mem_channel(self, mem_type: str) -> int:
        """
        Extract the channel index from mem_type.

        Supported formats:
        - "REMOTE"          -> returns 0
        - "REMOTE:1"        -> returns 0  (no channel specified)
        - "REMOTE:1.3"      -> returns 3  (channel = 3)
        """
        parts = mem_type.split(":", 1)
        if len(parts) == 2:
            # parts[1] may be "1" or "1.3"
            sub = parts[1].split(".", 1)
            if len(sub) == 2:
                chan_part = sub[1]
                if chan_part.isdigit():
                    return int(chan_part)
        return 0  # Default channel number


    def _attach_ucie_attrs(self, node: Any, mem_type: str) -> None:
        if not self.native_trace or self.manifest is None:
            return
        link_id = self.manifest.ucie_link_id(mem_type)
        if not link_id:
            return
        node.attr.append(
            ChakraAttr(
                name="ucie_transport_schema_version",
                string_val="ucie-transport-v1",
            )
        )
        node.attr.append(ChakraAttr(name="ucie_link_id", string_val=link_id))

    def get_memory_load_node(self, layer_name: str, tensor_type: str, mem_type: str, tensor_size: int) -> Any:
        node = self.get_node("MEM_LOAD_NODE_" + layer_name + "_" + tensor_type, MEM_LOAD_NODE)
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        node.attr.append(ChakraAttr(name="tensor_loc", uint32_val=self.get_mem_type(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_device", uint32_val=self.get_mem_device(mem_type)))
        self._attach_ucie_attrs(node, mem_type)
        return node

    def get_memory_store_node(self, layer_name: str, tensor_type: str, mem_type: str, tensor_size: int) -> Any:
        node = self.get_node("MEM_STORE_NODE_" + layer_name + "_" + tensor_type, MEM_STORE_NODE)
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        node.attr.append(ChakraAttr(name="tensor_loc", uint32_val=self.get_mem_type(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_device", uint32_val=self.get_mem_device(mem_type)))
        self._attach_ucie_attrs(node, mem_type)
        return node

    def get_pim_compute_node(self, layer_name: str, tensor_type: str, comp_time: int, mem_type: str, tensor_size: int) -> Any:
        node = self.get_node("PIM_COMP_NODE_" + layer_name + "_" + tensor_type, PIM_COMP_NODE)
        node.duration_micros = comp_time
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=tensor_size))
        node.attr.append(ChakraAttr(name="tensor_loc", uint32_val=self.get_mem_type(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_device", uint32_val=self.get_mem_device(mem_type)))
        node.attr.append(ChakraAttr(name="tensor_channel", uint32_val=self.get_mem_channel(mem_type)))
        return node

    def add_parent(self, child_node: Any, parent_node: Any) -> None:
        child_node.data_deps.append(parent_node.id)

    @staticmethod
    def _string_list_attr(name: str, values: tuple[str, ...]) -> ChakraAttr:
        return ChakraAttr(name=name, string_list=StringList(values=values))

    def get_memory_movement_nodes(self, npu_id: int) -> dict[str, tuple[Any, Any]]:
        if self.movement_events is None:
            return {}
        nodes = {}
        for event in self.movement_events.events_for_npu(npu_id):
            if event.event_id in self._emitted_movement_event_ids:
                raise ValueError(
                    f"movement event {event.event_id!r} was emitted more than once"
                )
            self._emitted_movement_event_ids.add(event.event_id)
            node = self.get_node(f"MEMORY_MOVEMENT_{event.event_id}", MEM_LOAD_NODE)
            node.attr.extend(
                [
                    ChakraAttr(name="tensor_size", uint64_val=event.bytes),
                    ChakraAttr(name="tensor_loc", uint32_val=event.source.tier_id),
                    ChakraAttr(name="tensor_device", uint32_val=event.source.device_id),
                    ChakraAttr(
                        name="memory_movement_schema_version",
                        string_val="memory-events-v1",
                    ),
                    ChakraAttr(
                        name="memory_movement_manifest_digest",
                        string_val=self.movement_events.manifest_digest,
                    ),
                    ChakraAttr(
                        name="movement_run_id",
                        string_val=self.movement_events.run_id,
                    ),
                    ChakraAttr(
                        name="movement_instance_id",
                        string_val=self.movement_events.instance_id,
                    ),
                    ChakraAttr(name="movement_event_id", string_val=event.event_id),
                    ChakraAttr(
                        name="movement_source_iteration_id",
                        uint32_val=event.source_iteration_id,
                    ),
                    ChakraAttr(name="movement_kind", string_val=event.kind),
                    ChakraAttr(name="movement_phase", string_val=event.phase),
                    ChakraAttr(
                        name="movement_priority_class",
                        string_val=event.priority_class,
                    ),
                    ChakraAttr(
                        name="movement_path_id",
                        string_val=self.movement_events.path_id,
                    ),
                    ChakraAttr(
                        name="movement_path_schema_version",
                        string_val=self.movement_events.path_schema_version,
                    ),
                    ChakraAttr(
                        name="movement_path_contract_status",
                        string_val=self.movement_events.path_contract_status,
                    ),
                    ChakraAttr(
                        name="movement_path_timing_provenance",
                        string_val=self.movement_events.path_timing_provenance,
                    ),
                    ChakraAttr(
                        name="movement_engine_count",
                        uint32_val=self.movement_events.engine_count,
                    ),
                    ChakraAttr(
                        name="movement_max_priority_burst",
                        uint32_val=self.movement_events.max_priority_burst,
                    ),
                    ChakraAttr(
                        name="movement_max_in_flight_page_movements",
                        uint32_val=(
                            self.movement_events.max_in_flight_page_movements
                        ),
                    ),
                    ChakraAttr(
                        name="movement_destination_tier_id",
                        uint32_val=event.destination.tier_id,
                    ),
                    ChakraAttr(
                        name="movement_destination_device_id",
                        uint32_val=event.destination.device_id,
                    ),
                    self._string_list_attr(
                        "movement_resource_ids", self.movement_events.resource_ids
                    ),
                    self._string_list_attr(
                        "movement_segment_ids",
                        tuple(
                            item.id for item in self.movement_events.path_segments
                        ),
                    ),
                    self._string_list_attr(
                        "movement_segment_kinds",
                        tuple(
                            item.kind for item in self.movement_events.path_segments
                        ),
                    ),
                    self._string_list_attr(
                        "movement_segment_resource_refs",
                        tuple(
                            item.resource_ref
                            for item in self.movement_events.path_segments
                        ),
                    ),
                    self._string_list_attr(
                        "movement_segment_operations",
                        tuple(
                            item.operation
                            for item in self.movement_events.path_segments
                        ),
                    ),
                    self._string_list_attr(
                        "movement_segment_byte_rules",
                        tuple(
                            item.byte_rule
                            for item in self.movement_events.path_segments
                        ),
                    ),
                    self._string_list_attr(
                        "movement_dependencies", event.depends_on
                    ),
                ]
            )
            if event.page_id is not None:
                node.attr.extend(
                    [
                        ChakraAttr(
                            name="movement_page_id", string_val=event.page_id
                        ),
                        ChakraAttr(
                            name="movement_transaction_id",
                            string_val=event.transaction_id,
                        ),
                        ChakraAttr(
                            name="movement_expected_residency_version",
                            uint32_val=event.expected_residency_version,
                        ),
                        ChakraAttr(
                            name="movement_home_domain_id",
                            uint32_val=event.home_domain_id,
                        ),
                    ]
                )
            nodes[event.event_id] = (event, node)
        return nodes

    def add_memory_movement_parents(
        self,
        comp_node: Any,
        layer_name: str,
        movement_nodes: dict[str, tuple[Any, Any]],
    ) -> None:
        if (self.movement_events is not None
                and self.movement_events.completion_owner != "workload"):
            raise ValueError("external preparation cannot enter a workload graph")
        for event, movement_node in movement_nodes.values():
            if layer_name in event.releases:
                self.add_parent(comp_node, movement_node)
                self._emitted_movement_releases.add((event.event_id, layer_name))

    def validate_movement_emission(self) -> None:
        if self.movement_events is None:
            return
        expected_events = {
            event.event_id for event in self.movement_events.events
        }
        if self._emitted_movement_event_ids != expected_events:
            missing = sorted(expected_events - self._emitted_movement_event_ids)
            raise ValueError(
                f"memory movement events target NPUs with no output ET: {missing}"
            )
        expected_releases = {
            (event.event_id, layer_name)
            for event in self.movement_events.events
            for layer_name in event.releases
        }
        if self._emitted_movement_releases != expected_releases:
            missing = sorted(expected_releases - self._emitted_movement_releases)
            raise ValueError(
                f"memory movement releases do not name emitted compute nodes: {missing}"
            )

    def get_weight_load_nodes(self, layer: Layer) -> List[Any]:
        if layer.weight_segments:
            return [
                self.get_memory_load_node(
                    layer.name,
                    f"WEIGHT_SEGMENT_{index}",
                    location,
                    size,
                )
                for index, (location, size) in enumerate(layer.weight_segments)
            ]
        if (
            self.local_offloading or layer.weight_memory_loc != "LOCAL"
        ) and layer.weight_memory_size > 0:
            return [
                self.get_memory_load_node(
                    layer.name,
                    "WEIGHT",
                    layer.weight_memory_loc,
                    layer.weight_memory_size,
                )
            ]
        return []

    def get_prefix_memory_nodes(
        self, layers: List[Layer]
    ) -> tuple[List[Any], List[Any], int]:
        """Return ordered load/store nodes encoded before model layers."""

        loads = []
        stores = []
        prefix_count = 0
        for layer in layers:
            if "kv_load" in layer.name:
                loads.append(
                    self.get_memory_load_node(
                        layer.name,
                        "WEIGHT",
                        layer.weight_memory_loc,
                        layer.weight_memory_size,
                    )
                )
            elif "kv_evict" in layer.name or "cache_writeback" in layer.name:
                stores.append(
                    self.get_memory_store_node(
                        layer.name,
                        "WEIGHT",
                        layer.weight_memory_loc,
                        layer.weight_memory_size,
                    )
                )
            else:
                break
            prefix_count += 1
        return loads, stores, prefix_count

    @staticmethod
    def _reset_layer_state(layers: List[Layer]) -> None:
        """Discard node references written while producing a previous ET."""
        for layer in layers:
            layer.comp_node = None
            layer.comm_node = None
            if not layer.is_expert and not layer.is_pim:
                layer.input_memory_node = None
                layer.weight_memory_node = None
                layer.weight_memory_nodes = []
                layer.output_memory_node = None

    @staticmethod
    def _marker_regions(layers: List[Layer]) -> List[Any]:
        """Return closed EXPERT/PIM regions as [start, end) Trace ranges."""
        regions = []
        active_kind = None
        active_start = None
        for index, layer in enumerate(layers):
            if not layer.is_expert and not layer.is_pim:
                continue
            kind = "EXPERT" if layer.is_expert else "PIM"
            marker = layer.expert_num if layer.is_expert else layer.pim_num
            if marker == "END":
                if active_kind != kind:
                    raise ValueError(f"unmatched {kind} END marker at Trace row {index}")
                regions.append((active_start, index + 1, kind))
                active_kind = None
                active_start = None
            elif active_kind is None:
                active_kind = kind
                active_start = index
            elif active_kind != kind:
                raise ValueError(
                    f"{kind} marker at Trace row {index} is nested inside "
                    f"{active_kind} region starting at row {active_start}"
                )
        if active_kind is not None:
            raise ValueError(
                f"unclosed {active_kind} region starting at Trace row {active_start}"
            )
        return regions

    def _validate_marker_boundaries(
        self, layers: List[Layer], stage_boundaries: List[int]
    ) -> None:
        regions = self._marker_regions(layers)
        for boundary in stage_boundaries:
            for start, end, kind in regions:
                if start <= boundary < end:
                    raise ValueError(
                        f"pp_stage_boundary {boundary} falls inside {kind} region "
                        f"[{start}, {end})"
                    )

    def get_stage_edges(self, num_layers: int, num_npu_group: int,
                        stage_boundaries: List[int]) -> List[Any]:
        """Resolve the [start, end) trace-line range owned by each pipeline stage.

        ``stage_boundaries`` comes from the trace header and holds the line
        index at which every stage after the first begins. The frontend puts
        those cuts on transformer-block boundaries, which is what makes the
        tensor crossing a stage boundary the hidden state: the sending
        layer's output_size then equals the receiving layer's input_size, and
        the SEND/RECV pair matches in ASTRA-sim (its callback tracker keys on
        chunk size, so a mismatch deadlocks the run instead of erroring).
        Splitting the line count evenly instead lands cuts inside a block --
        e.g. between qkv_proj and rotary_emb, whose declared sizes differ by
        the V projection -- so do not reintroduce that.
        """
        if num_npu_group == 1:
            if stage_boundaries:
                raise ValueError("PP1 Trace must not declare pp_stage_boundaries")
            return [(0, num_layers)]
        if len(stage_boundaries) != num_npu_group - 1:
            raise ValueError(
                f"trace declares model_parallel_NPU_group: {num_npu_group} but "
                f"carries {len(stage_boundaries)} pp_stage_boundaries "
                f"(expected {num_npu_group - 1}); regenerate the trace")
        edges = [0] + list(stage_boundaries) + [num_layers]
        for i in range(len(edges) - 1):
            if not 0 <= edges[i] < edges[i + 1] <= num_layers:
                raise ValueError(
                    f"pp_stage_boundaries {stage_boundaries} are not a strictly "
                    f"increasing split of {num_layers} layers")
        return [(edges[i], edges[i + 1]) for i in range(num_npu_group)]

    def convert_common(self, f: TextIOWrapper, num_layers: int, num_npu_group: int,
                       stage_boundaries: List[int] = None):
        layers: list[Layer] = self.get_layers(f)
        if len(layers) != num_layers:
            raise ValueError(
                f"Trace declares {num_layers} rows but contains {len(layers)}"
            )

        # vllm: check eviction or load
        loads, stores, prefix_count = self.get_prefix_memory_nodes(layers)
        layers = layers[prefix_count:]
        num_layers -= prefix_count

        if self.num_npus % num_npu_group != 0: print("Warning! num_npus % num_npu_group != 0, Some npus won't do anything!")
        npus_per_group = self.num_npus // num_npu_group
        if npus_per_group == 1: # same as pipeline parallelism, ignore all reduce
            use_comm = False
        else:
            use_comm = True
        self._validate_marker_boundaries(layers, stage_boundaries or [])
        stage_edges = self.get_stage_edges(num_layers, num_npu_group,
                                           stage_boundaries or [])
        output_paths = []

        for npu_group in range(num_npu_group):
            for npu_offset in range(npus_per_group):
                # Re-read the authoritative edges per rank: the walk below may
                # rebind layer_end if it ever overruns.
                layer_start, layer_end = stage_edges[npu_group]
                self._reset_layer_state(layers)
                npu_id = npu_group * npus_per_group + npu_offset + self.npu_offset
                movement_nodes = self.get_memory_movement_nodes(npu_id)
                output_filename = "%s.%d.et" % (self.output_filename, npu_id)
                output_paths.append(Path(output_filename))
                first_comp_node = True
                with open(output_filename, "wb") as g:
                    global_metadata = self.get_global_metadata(npu_id)
                    encode_message(g, global_metadata)
                    for _, movement_node in movement_nodes.values():
                        encode_message(g, movement_node)
                    for store in stores:
                        encode_message(g, store)
                    for load in loads:
                        encode_message(g, load)
                    if npu_group == 0:
                        # Load Input
                        input_load_node = self.get_memory_load_node(
                            layers[layer_start].name,
                            "INPUT",
                            layers[layer_start].input_memory_loc,
                            layers[layer_start].input_memory_size,
                        )
                        encode_message(g, input_load_node)
                    else:
                        if layers[layer_start].is_expert or layers[layer_start].is_pim:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start-1].name,
                                comm_type=layers[layer_start-1].comm_type,
                                comm_size=layers[layer_start-1].output_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            encode_message(g, receive_input_node)
                        else:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start].name,
                                comm_type=layers[layer_start].comm_type,
                                comm_size=layers[layer_start].input_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            encode_message(g, receive_input_node)

                    expert_start = False
                    pim_start = False
                    attn_remain = False # to handle remaining prefill attention after pim
                    pim_parent_nodes = []
                    pim_comp_nodes = []
                    past_pim_comp_nodes = []
                    last_batch_type = "BATCH_1"
                    layer_num = layer_start
                    while expert_start or pim_start or attn_remain or layer_num < layer_end:
                        if not layers[layer_num].is_expert and not layers[layer_num].is_pim:
                            layers[layer_num].weight_memory_nodes = (
                                self.get_weight_load_nodes(layers[layer_num])
                            )
                            for weight_load_node in layers[layer_num].weight_memory_nodes:
                                if expert_start:
                                    self.add_parent(weight_load_node, comp_node)
                                encode_message(g, weight_load_node)
                            # Compute
                            if layers[layer_num].comp_time != 0 and not pim_start: # pim computation is handled pim_comp_node
                                comp_node = self.get_comp_node(
                                    layers[layer_num].name,
                                    layers[layer_num].comp_time)
                                layers[layer_num].comp_node = comp_node
                                self.add_memory_movement_parents(
                                    comp_node,
                                    layers[layer_num].name,
                                    movement_nodes,
                                )

                                # handle pim parent nodes, and if prefill attention remains wait until all attention is done (before o_proj)
                                if len(pim_parent_nodes) != 0:
                                    if attn_remain:
                                        for parent in pim_parent_nodes:
                                            self.add_parent(comp_node, parent)
                                    pim_parent_nodes = [] # reset pim parent nodes

                                    if "attn" in layers[layer_num].name:
                                        attn_remain = False
                                else:
                                    if first_comp_node:
                                        if npu_group == 0:
                                            self.add_parent(comp_node, input_load_node)
                                        else:
                                            self.add_parent(comp_node, receive_input_node)
                                        for store in stores:
                                            self.add_parent(comp_node, store)
                                        for load in loads:
                                            self.add_parent(comp_node, load)
                                        for weight_node in layers[layer_num].weight_memory_nodes:
                                            self.add_parent(comp_node, weight_node)
                                        first_comp_node = False
                                    else:
                                        for weight_node in layers[layer_num].weight_memory_nodes:
                                            self.add_parent(comp_node, weight_node)
                                        if layers[layer_num - 1].comm_node != None:
                                            self.add_parent(comp_node, layers[layer_num - 1].comm_node)
                                        elif layers[layer_num - 1].comp_node != None:
                                            self.add_parent(comp_node, layers[layer_num - 1].comp_node)
                                        else:
                                            self.add_parent(comp_node, layers[layer_num - 2].comp_node)

                                # handle pim_compute_mode dependency & should not be remaining attention
                                if not attn_remain and len(pim_comp_nodes) != 0:
                                    if layers[layer_num].misc == "NONE": # no sub-batch interleaving
                                        for pim_comp in pim_comp_nodes:
                                            self.add_parent(comp_node, pim_comp)
                                        pim_comp_nodes = [] # reset pim comp nodes
                                    elif layers[layer_num].misc != last_batch_type:
                                        if len(past_pim_comp_nodes) != 0:
                                            for pim_comp in past_pim_comp_nodes:
                                                self.add_parent(comp_node, pim_comp)
                                        past_pim_comp_nodes = pim_comp_nodes # update past pim comp nodes
                                        pim_comp_nodes = [] # reset pim comp nodes
                                        last_batch_type = layers[layer_num].misc

                                encode_message(g, comp_node)

                            # PIM compute
                            if pim_start:
                                pim_comp_node = self.get_pim_compute_node(
                                    layers[layer_num].name,
                                    "PIM",
                                    layers[layer_num].comp_time,
                                    layers[layer_num].input_memory_loc,
                                    layers[layer_num].input_memory_size + layers[layer_num].output_memory_size # load/store from pim
                                )
                                pim_comp_nodes.append(pim_comp_node)
                                for parent in pim_parent_nodes:
                                    self.add_parent(pim_comp_node, parent)
                                encode_message(g, pim_comp_node)

                            # Communication (if required)
                            if layers[layer_num].comm_type != "NONE" and use_comm:
                                comm_coll_node = self.get_comm_coll_node(layers[layer_num].name, layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                # for j in range(self.num_dims):
                                # comm_coll_node.involved_dim.append(True)
                                layers[layer_num].comm_node = comm_coll_node
                                if layers[layer_num].comp_time != 0:
                                    self.add_parent(comm_coll_node, comp_node)
                                encode_message(g, comm_coll_node)
                            # add layer_num
                            layer_num += 1
                        # expert layer starts
                        elif layers[layer_num].is_expert:
                            # communication can happen even with one NPU in the group, for example, expert input gathering in data parallel
                            if expert_start == False and layers[layer_num].comm_size > 0 and layers[layer_num].comm_type != "NONE":
                                # Start of expert, add ALLTOALL communication before expert computation
                                comm_coll_node = self.get_comm_coll_node("expert_start", layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                layers[layer_num].comm_node = comm_coll_node
                                self.add_parent(comm_coll_node, comp_node)
                                encode_message(g, comm_coll_node)
                            expert_start = True
                            # check expert end
                            if layers[layer_num].expert_num == 'END':
                                expert_start = False
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                # End of expert, add ALLTOALL communication after expert computation
                                if layers[layer_num].comm_size > 0 and layers[layer_num].comm_type != "NONE":
                                    comm_coll_node = self.get_comm_coll_node("expert_end", layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                    layers[layer_num].comm_node = comm_coll_node
                                    self.add_parent(comm_coll_node, comp_node)
                                    encode_message(g, comm_coll_node)
                                layer_num += 1
                                continue
                            # round robin assignment
                            expert_id = int(layers[layer_num].expert_num) % npus_per_group
                            if npu_offset != expert_id:
                                # go to next expert
                                while True:
                                    layer_num += 1
                                    if layers[layer_num].is_expert:
                                        break
                            else:
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                layer_num += 1
                        # pim layer starts
                        elif layers[layer_num].is_pim:
                            pim_start = True
                            # check attention end
                            if layers[layer_num].pim_num == 'END':
                                pim_start = False
                                if "attn" in layers[layer_num + 1].name:
                                    attn_remain = True # prefill attn remains
                                layer_num += 1
                                continue
                            # add pim parent nodes for dependency
                            elif int(layers[layer_num].pim_num) == 0:
                                if first_comp_node:
                                    if npu_group == 0:
                                        pim_parent_nodes.append(input_load_node)
                                    else:
                                        pim_parent_nodes.append(receive_input_node)
                                    pim_parent_nodes.extend(stores)
                                    pim_parent_nodes.extend(loads)
                                    # discarded weight parent because attention has no weight
                                    first_comp_node = False
                                else:
                                    # discarded weight parent because attention has no weight
                                    if layers[layer_num - 1].comm_node != None:
                                        pim_parent_nodes.append(layers[layer_num - 1].comm_node)
                                    elif layers[layer_num - 1].comp_node != None:
                                        pim_parent_nodes.append(layers[layer_num - 1].comp_node)
                                    else:
                                        pim_parent_nodes.append(layers[layer_num - 1].comp_node)
                            # round robin assignment
                            pim_id = int(layers[layer_num].pim_num) % npus_per_group
                            if npu_offset != pim_id:
                                # go to next attention
                                while True:
                                    layer_num += 1
                                    if layers[layer_num].is_pim:
                                        break
                            else:
                                layer_num += 1

                    if layer_num != layer_end:
                        raise ValueError(
                            f"pipeline stage {npu_group} consumed through Trace row "
                            f"{layer_num}, expected frozen boundary {layer_end}"
                        )

                    if npu_group == (num_npu_group - 1):
                        # Store output (for the last layer)
                        # The last layer is the sampler: what crosses back to
                        # the host is its OUTPUT, the sampled token ids, 4 bytes
                        # per sequence.
                        #
                        # Do not switch this back to input_memory_size. That read
                        # was a deliberate down-scaling when lm_head was the last
                        # trace layer: its output is the logits
                        # (num_seqs * vocab_size * dtype) and billing those to CPU
                        # memory every iteration was a large overcharge, so the
                        # smaller input (the hidden state) stood in. Once sampler
                        # was appended as the new last layer the two swapped
                        # roles -- sampler's output is the token ids and its input
                        # is the logits -- so reading the input landed back on
                        # exactly the tensor the workaround existed to avoid.
                        output_store_node = self.get_memory_store_node(
                            layers[layer_end - 1].name,
                            "OUTPUT",
                            layers[layer_end - 1].output_memory_loc,
                            layers[layer_end - 1].output_memory_size,
                        )
                        # if pim_comp_nodes are not consumed yet, add dependency
                        if len(pim_comp_nodes) != 0:
                            for pim_comp in pim_comp_nodes:
                                self.add_parent(send_output_node, pim_comp)
                                pim_comp_nodes = []
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(output_store_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(output_store_node, comp_node)
                        else:
                            self.add_parent(output_store_node, layers[layer_end - 2].comp_node)
                        encode_message(g, output_store_node)
                    else:
                        if layers[layer_end - 1].is_expert or layers[layer_end - 1].is_pim:
                            # Send output (to the next layer in another npu group)
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end].name,
                                comm_type=layers[layer_end].comm_type,
                                comm_size=layers[layer_end].input_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        else:
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end - 1].name,
                                comm_type=layers[layer_end - 1].comm_type,
                                comm_size=layers[layer_end - 1].output_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        # if pim_comp_nodes are not consumed yet, add dependency
                        if len(pim_comp_nodes) != 0:
                            for pim_comp in pim_comp_nodes:
                                self.add_parent(send_output_node, pim_comp)
                                pim_comp_nodes = []
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(send_output_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(send_output_node, comp_node)
                        else:
                            self.add_parent(send_output_node, layers[layer_end - 2].comp_node)
                        encode_message(g, send_output_node)
        return output_paths

    def convert_prefill(self, f: TextIOWrapper, num_layers: int, num_npu_group: int,
                        stage_boundaries: List[int] = None):
        layers: list[Layer] = self.get_layers(f)
        if len(layers) != num_layers:
            raise ValueError(
                f"Trace declares {num_layers} rows but contains {len(layers)}"
            )
        # There will be no pim operation in prefill (PIM cannot perform GEMM)

        # vllm: check eviction or load
        loads, stores, prefix_count = self.get_prefix_memory_nodes(layers)
        layers = layers[prefix_count:]
        num_layers -= prefix_count

        if self.num_npus % num_npu_group != 0: print("Warning! num_npus % num_npu_group != 0, Some npus won't do anything!")
        npus_per_group = self.num_npus // num_npu_group
        if npus_per_group == 1: # same as pipeline parallelism, ignore all reduce
            use_comm = False
        else:
            use_comm = True
        self._validate_marker_boundaries(layers, stage_boundaries or [])
        stage_edges = self.get_stage_edges(num_layers, num_npu_group,
                                           stage_boundaries or [])
        output_paths = []

        for npu_group in range(num_npu_group):
            for npu_offset in range(npus_per_group):
                # Re-read the authoritative edges per rank: the walk below may
                # rebind layer_end if it ever overruns.
                layer_start, layer_end = stage_edges[npu_group]
                self._reset_layer_state(layers)
                npu_id = npu_group * npus_per_group + npu_offset + self.npu_offset
                movement_nodes = self.get_memory_movement_nodes(npu_id)
                output_filename1 = "%s.%d.et" % (self.output_filename, npu_id)
                output_filename2 = "%s.%d.et" % (self.output_filename, npu_id + self.num_npus) # sender for prefill-decode
                output_paths.extend((Path(output_filename1), Path(output_filename2)))
                first_comp_node = True
                with open(output_filename1, "wb") as g, open(output_filename2, "wb") as s:
                    global_metadata = self.get_global_metadata(npu_id)
                    encode_message(g, global_metadata)
                    encode_message(s, self.get_global_metadata(npu_id + self.num_npus))
                    for _, movement_node in movement_nodes.values():
                        encode_message(g, movement_node)
                    for store in stores:
                        encode_message(g, store)
                    for load in loads:
                        encode_message(g, load)
                    if npu_group == 0:
                        # Load Input
                        input_load_node = self.get_memory_load_node(
                            layers[layer_start].name,
                            "INPUT",
                            layers[layer_start].input_memory_loc,
                            layers[layer_start].input_memory_size,
                        )
                        encode_message(g, input_load_node)
                    else:
                        if layers[layer_start].is_expert:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start-1].name,
                                comm_type=layers[layer_start-1].comm_type,
                                comm_size=layers[layer_start-1].output_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            encode_message(g, receive_input_node)
                        else:
                            # Receive input (from the previous layer in another npu group)
                            receive_input_node = self.get_comm_node(
                                is_send=False,
                                layer_name=layers[layer_start].name,
                                comm_type=layers[layer_start].comm_type,
                                comm_size=layers[layer_start].input_memory_size,
                                comm_src=npu_id - npus_per_group,
                                comm_dst=npu_id
                            )
                            encode_message(g, receive_input_node)

                    expert_start = False
                    layer_num = layer_start
                    while expert_start or layer_num < layer_end:
                        if not layers[layer_num].is_expert:
                            layers[layer_num].weight_memory_nodes = (
                                self.get_weight_load_nodes(layers[layer_num])
                            )
                            for weight_load_node in layers[layer_num].weight_memory_nodes:
                                if expert_start:
                                    self.add_parent(weight_load_node, comp_node)
                                encode_message(g, weight_load_node)

                            # Compute
                            if layers[layer_num].comp_time != 0:
                                comp_node = self.get_comp_node(
                                    layers[layer_num].name,
                                    layers[layer_num].comp_time)
                                layers[layer_num].comp_node = comp_node
                                self.add_memory_movement_parents(
                                    comp_node,
                                    layers[layer_num].name,
                                    movement_nodes,
                                )

                                if first_comp_node:
                                    if npu_group == 0:
                                        self.add_parent(comp_node, input_load_node)
                                    else:
                                        self.add_parent(comp_node, receive_input_node)
                                    for store in stores:
                                        self.add_parent(comp_node, store)
                                    for load in loads:
                                        self.add_parent(comp_node, load)
                                    for weight_node in layers[layer_num].weight_memory_nodes:
                                        self.add_parent(comp_node, weight_node)
                                    first_comp_node = False
                                else:
                                    for weight_node in layers[layer_num].weight_memory_nodes:
                                        self.add_parent(comp_node, weight_node)
                                    if layers[layer_num - 1].comm_node != None:
                                        self.add_parent(comp_node, layers[layer_num - 1].comm_node)
                                    elif layers[layer_num - 1].comp_node != None:
                                        self.add_parent(comp_node, layers[layer_num - 1].comp_node)
                                    else:
                                        self.add_parent(comp_node, layers[layer_num - 2].comp_node)

                                encode_message(g, comp_node)

                                # Send KV cache after each kv_proj.
                                # comm_size comes from the trace's comm_size
                                # column, which the frontend fills with the
                                # per-layer, per-rank K+V bytes. It used to read
                                # output_memory_size, i.e. the whole QKV
                                # activation, which shipped Q as well and
                                # overstated the transfer by
                                # (q_dim + 2*kv_dim) / (2*kv_dim) -- 3x for
                                # Llama-3.1-8B -- and ignored kv_cache_dtype.
                                if "v_proj" in layers[layer_num].name and layers[layer_num].comm_size > 0:
                                    send_kv_node = self.get_comm_node(
                                        is_send=True,
                                        layer_name="kv_proj",
                                        comm_type=layers[layer_num].comm_type,
                                        comm_size=layers[layer_num].comm_size,
                                        comm_src=npu_id,
                                        comm_dst=npu_id + self.num_npus, # to the paired npu in decode
                                        id=layer_num
                                    )
                                    self.add_parent(send_kv_node, comp_node)
                                    encode_message(g, send_kv_node)

                                    recv_kv_node = self.get_comm_node(
                                        is_send=False,
                                        layer_name="kv_proj",
                                        comm_type=layers[layer_num].comm_type,
                                        comm_size=layers[layer_num].comm_size,
                                        comm_src=npu_id,
                                        comm_dst=npu_id + self.num_npus,
                                        id=layer_num
                                    )
                                    encode_message(s, recv_kv_node)

                            # Communication (if required)
                            if layers[layer_num].comm_type != "NONE" and use_comm:
                                comm_coll_node = self.get_comm_coll_node(layers[layer_num].name, layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                # for j in range(self.num_dims):
                                # comm_coll_node.involved_dim.append(True)
                                layers[layer_num].comm_node = comm_coll_node
                                if layers[layer_num].comp_time != 0:
                                    self.add_parent(comm_coll_node, comp_node)
                                encode_message(g, comm_coll_node)
                            # add layer_num
                            layer_num += 1
                        # expert layer starts
                        elif layers[layer_num].is_expert:
                            # communication can happen even with one NPU in the group, for example, expert input gathering in data parallel
                            if expert_start == False and layers[layer_num].comm_size > 0 and layers[layer_num].comm_type != "NONE":
                                # Start of expert, add ALLTOALL communication before expert computation
                                comm_coll_node = self.get_comm_coll_node("expert_start", layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                layers[layer_num].comm_node = comm_coll_node
                                self.add_parent(comm_coll_node, comp_node)
                                encode_message(g, comm_coll_node)
                            expert_start = True
                            # check expert end
                            if layers[layer_num].expert_num == 'END':
                                expert_start = False
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                # End of expert, add ALLTOALL communication after expert computation
                                if layers[layer_num].comm_size > 0 and layers[layer_num].comm_type != "NONE":
                                    comm_coll_node = self.get_comm_coll_node("expert_end", layers[layer_num].comm_type, layers[layer_num].comm_size, layers[layer_num].involved_dim)
                                    layers[layer_num].comm_node = comm_coll_node
                                    self.add_parent(comm_coll_node, comp_node)
                                    encode_message(g, comm_coll_node)
                                layer_num += 1
                                continue
                            # round robin assignment
                            expert_id = int(layers[layer_num].expert_num) % npus_per_group
                            if npu_offset != expert_id:
                                # go to next expert
                                while True:
                                    layer_num += 1
                                    if layers[layer_num].is_expert:
                                        break
                            else:
                                layers[layer_num].comp_node = comp_node # is latest comp_node
                                layer_num += 1

                    if layer_num != layer_end:
                        raise ValueError(
                            f"pipeline stage {npu_group} consumed through Trace row "
                            f"{layer_num}, expected frozen boundary {layer_end}"
                        )

                    if npu_group == (num_npu_group - 1):
                        # Send output (for the last layer, to the paired decode npu)
                        send_output_node = self.get_comm_node(
                            is_send=True,
                            layer_name=layers[layer_end - 1].name,
                            comm_type=layers[layer_end - 1].comm_type,
                            comm_size=layers[layer_end - 1].output_memory_size,
                            comm_src=npu_id,
                            comm_dst=npu_id + self.num_npus
                        )
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(send_output_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(send_output_node, comp_node)
                        else:
                            self.add_parent(send_output_node, layers[layer_end - 2].comp_node)
                        encode_message(g, send_output_node)
                        # paired decode npu receive output
                        recv_output_node = self.get_comm_node(
                            is_send=False,
                            layer_name=layers[layer_end - 1].name,
                            comm_type=layers[layer_end - 1].comm_type,
                            comm_size=layers[layer_end - 1].output_memory_size,
                            comm_src=npu_id,
                            comm_dst=npu_id + self.num_npus
                        )
                        encode_message(s, recv_output_node)
                    else:
                        if layers[layer_end - 1].is_expert:
                            # Send output (to the next layer in another npu group)
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end].name,
                                comm_type=layers[layer_end].comm_type,
                                comm_size=layers[layer_end].input_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        else:
                            send_output_node = self.get_comm_node(
                                is_send=True,
                                layer_name=layers[layer_end - 1].name,
                                comm_type=layers[layer_end - 1].comm_type,
                                comm_size=layers[layer_end - 1].output_memory_size,
                                comm_src=npu_id,
                                comm_dst=npu_id + npus_per_group
                            )
                        if layers[layer_end - 1].comm_type != "NONE" and use_comm:
                            self.add_parent(send_output_node, comm_coll_node)
                        elif layers[layer_end - 1].comp_node != None:
                            self.add_parent(send_output_node, comp_node)
                        else:
                            self.add_parent(send_output_node, layers[layer_end - 2].comp_node)
                        encode_message(g, send_output_node)
        return output_paths

    def convert_event(self, f: TextIOWrapper, num_layers: int):
        layers: list[Layer] = self.get_layers(f)
        if len(layers) != num_layers:
            raise ValueError(
                f"Trace declares {num_layers} rows but contains {len(layers)}"
            )
        output_paths = []
        for npu_id in range(self.num_npus):
            movement_nodes = self.get_memory_movement_nodes(npu_id)
            output_filename = "%s.%d.et" % (self.output_filename, npu_id)
            output_paths.append(Path(output_filename))
            with open(output_filename, "wb") as g:
                global_metadata = self.get_global_metadata(npu_id)
                encode_message(g, global_metadata)
                for _, movement_node in movement_nodes.values():
                    encode_message(g, movement_node)
                for idx, layer in enumerate(layers):
                    comp_node = self.get_comp_node(layer.name, layer.comp_time)
                    layer.comp_node = comp_node
                    self.add_memory_movement_parents(
                        comp_node, layer.name, movement_nodes
                    )
                    encode_message(g, comp_node)
        return output_paths

    def convert(self):
        if (self.movement_events is not None
                and self.movement_events.completion_owner != "workload"):
            raise ValueError("external preparation cannot enter a workload graph")
        self._emitted_movement_event_ids.clear()
        self._emitted_movement_releases.clear()
        with open(self.input_filename, "r") as f:
            execution_type, header = _parse_trace_header(f.readline())
            trace_schema = header.get("trace_schema")
            trace_digest = header.get("tier_manifest_digest")
            native_header = trace_schema is not None or trace_digest is not None
            if native_header:
                if trace_schema != "llm-tier-v1":
                    raise ValueError("native Trace requires trace_schema: llm-tier-v1")
                if self.manifest is None:
                    raise ValueError("native Trace requires --tier-manifest")
                if trace_digest != self.manifest.digest:
                    raise ValueError(
                        "tier_manifest_digest mismatch between Trace and manifest"
                    )
                self.native_trace = True
            elif self.manifest is not None:
                raise ValueError("legacy Trace must not be used with --tier-manifest")
            if self.services is not None:
                self.services.validate_trace(header)
            elif 'service_binding_digest' in header or 'service_activation_id' in header:
                raise ValueError("Trace service identity requires --physical-service-bindings")
            try:
                num_npu_group = int(header.get("model_parallel_NPU_group", 0))
            except ValueError as exc:
                raise ValueError("invalid model_parallel_NPU_group in Trace header") from exc
            boundary_text = header.get("pp_stage_boundaries", "")
            try:
                stage_boundaries = (
                    [int(value) for value in boundary_text.split(",")]
                    if boundary_text
                    else []
                )
            except ValueError as exc:
                raise ValueError("invalid pp_stage_boundaries in Trace header") from exc

            second_line = f.readline().strip()
            num_layers = int(second_line)

            third_line = f.readline() # This is for the table header, so just ignore it

            if execution_type == "COLOCATED":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                outputs = self.convert_common(
                    f, num_layers, num_npu_group, stage_boundaries
                )
            elif execution_type == "PREFILL":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                outputs = (
                    self.convert_common(
                        f, num_layers, num_npu_group, stage_boundaries
                    )
                    if self.pd_kv_transfer_mode == "faithful"
                    else self.convert_prefill(
                        f, num_layers, num_npu_group, stage_boundaries
                    )
                )
            elif execution_type == "DECODE":
                if num_npu_group <= 0:
                    raise ValueError(f"model_parallel_NPU_group <= 0")
                outputs = self.convert_common(
                    f, num_layers, num_npu_group, stage_boundaries
                )
            elif execution_type == "EVENT":
                outputs = self.convert_event(f, num_layers)
            else:
                raise ValueError(f"Unsupported execution type, {execution_type}")
        self.validate_movement_emission()
        validate_et_group(outputs)
        return outputs
