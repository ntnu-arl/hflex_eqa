# BSD 3-Clause License

# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""VLM based EQA planner."""

import collections
import copy
import json
import math
import os
import pathlib
import queue
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import spark_dsg._dsg_bindings as spark_dsg
import vlms_python
from spark_config import Config, config_field
from vlms_python import RoomClassificationInput

from hflex_eqa_python.conversions import DsgVLMFormat, to_vlm_format
from hflex_eqa_python.eqa_planner.eqa_planner_input import (
    EQAPlannerInput,
)
from hflex_eqa_python.eqa_planner.eqa_planner_output import (
    EQAPlannerOutput,
    str_to_output_mode,
)
from hflex_eqa_python.misc import LoggerWarn


def clean_room_name(room: str) -> str:
    """Normalize a floorplan room node name to its semantic room type."""
    room = re.sub(r"^tie:\s*", "", room).strip()
    room = re.sub(r"[_-]\d+$", "", room).strip()
    room = re.sub(r"\s+\d+$", "", room).strip()
    room = room.replace("_", " ")
    room = re.sub(r"\s+", " ", room).strip()
    return room


def room_name_parts(room: str) -> list[str]:
    """Split a possibly tied room label into normalized semantic room types."""
    room = re.sub(r"^tie:\s*", "", room).strip()
    return [clean_room_name(part) for part in room.split("&")]


def unique_room_names(room_list, add_extra_edges=False):
    seen = set()
    result = []

    extra_edges = []
    for room in room_list:
        cleaned_all = []
        for cleaned in room_name_parts(room):
            # Keep unique labels in original order
            cleaned_all.append(cleaned)
            if cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        if add_extra_edges and len(cleaned_all) > 1:
            for i in range(len(cleaned_all)):
                for j in range(i + 1, len(cleaned_all)):
                    extra_edges.append((cleaned_all[i], cleaned_all[j]))

    return result, extra_edges


def softmax(scores: list[float]) -> list[float]:
    """Compute a numerically stable softmax over a list of scores."""
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    exp_sum = sum(exp_scores)
    if exp_sum <= 0.0:
        return [1.0 / float(len(scores))] * len(scores)
    return [score / exp_sum for score in exp_scores]


@dataclass
class FloorPlanGraph(Config):
    """Graph representing the floorplan of the environment."""

    nodes: list[str] = field(default_factory=list)
    edges: list[list[str]] = field(default_factory=list)

    @classmethod
    def from_json_file(cls, path: str | os.PathLike[str]) -> "FloorPlanGraph":
        """Load a floorplan graph from a JSON file."""
        floorplan_path = pathlib.Path(path).expanduser().absolute()
        if not floorplan_path.is_file():
            raise FileNotFoundError(
                f"Floorplan JSON path does not exist or is not a file: {floorplan_path}"
            )

        with floorplan_path.open() as f:
            floorplan_data = json.load(f)

        if not isinstance(floorplan_data, dict):
            raise ValueError(
                f"Floorplan JSON '{floorplan_path}' must contain a JSON object."
            )
        if "floorplan_graph" in floorplan_data:
            floorplan_data = floorplan_data["floorplan_graph"]
        if not isinstance(floorplan_data, dict):
            raise ValueError(
                f"Floorplan JSON '{floorplan_path}' field 'floorplan_graph' "
                "must be an object."
            )

        nodes = floorplan_data.get("nodes")
        edges = floorplan_data.get("edges")
        if not isinstance(nodes, list):
            raise ValueError(
                f"Floorplan JSON '{floorplan_path}' must contain a list field 'nodes'."
            )
        if not isinstance(edges, list):
            raise ValueError(
                f"Floorplan JSON '{floorplan_path}' must contain a list field 'edges'."
            )

        if not all(isinstance(node, str) for node in nodes):
            raise ValueError(
                f"Floorplan JSON '{floorplan_path}' field 'nodes' must only "
                "contain strings."
            )
        for edge in edges:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or not all(isinstance(node, str) for node in edge)
            ):
                raise ValueError(
                    f"Floorplan JSON '{floorplan_path}' field 'edges' must contain "
                    "two-element string lists."
                )

        return cls(nodes=nodes, edges=edges)

    def to_string(self) -> str:
        """Convert the floorplan graph to a string representation."""
        graph_str = "FLOORPLAN GRAPH:\n"
        graph_str += "Nodes:\n"
        for node in self.nodes:
            graph_str += f"- {node}\n"
        graph_str += "Edges:\n"
        for edge in self.edges:
            graph_str += f"- {edge[0]} <-> {edge[1]}\n"
        return graph_str

    def to_json(self) -> str:
        """Convert the floorplan graph to a JSON string."""
        return json.dumps({"nodes": self.nodes, "edges": self.edges})


@dataclass
class EQAPlannerConfig(Config):
    """Configuration for the EQA Planner."""

    vlm: Any = config_field("eqa_planner", default="openai")
    room_classification: Any = config_field("room_classification", default="openai")
    image_sampler: Any = config_field("image_sampling", default="object")
    history_length: int = 5
    max_images_per_room: int = 5
    stitch_room_images: bool = False
    room_image_stitching: str = "none"
    debug_input: bool = False
    question: str = ""
    floorplan_graph: FloorPlanGraph = field(default_factory=FloorPlanGraph)
    floorplan_source: str = "list"
    floorplan_json_path: str = ""
    use_floorplan_prior: bool = True
    classify_rooms: bool = True
    room_classification_min_confidence: float = 0.0
    use_floorplan_rooms_as_choices: bool = False
    use_choices: bool = False
    choices: list[str] = field(default_factory=list)
    log_folder: str = ""

    def resolve_floorplan_graph(self) -> None:
        """Resolve the configured floorplan source into ``floorplan_graph``."""
        source = (self.floorplan_source or "list").lower().strip()
        if source in ("list", "lists", "manual"):
            return
        if source in ("json", "file", "json_file"):
            if not self.floorplan_json_path:
                raise ValueError(
                    "floorplan_json_path must be set when floorplan_source is 'json'."
                )
            self.floorplan_graph = FloorPlanGraph.from_json_file(
                self.floorplan_json_path
            )
            return
        raise ValueError(
            "Unknown floorplan_source "
            f"'{self.floorplan_source}'. Expected 'list' or 'json'."
        )


class EQAPlanner:
    """VLM based EQA Planner."""

    def __init__(self, config: EQAPlannerConfig) -> None:
        """Initialize the EQA Planner."""
        self._config = config
        self._config.resolve_floorplan_graph()
        self._vlm = self._config.vlm.create()
        self._room_classifier = (
            self._config.room_classification.create()
            if self._config.classify_rooms
            else None
        )
        self.question = self._config.question
        self._image_sampler = self._config.image_sampler.create()
        self._image_sampler.set_question(self.question)

        self.history: list[EQAPlannerOutput] = []

        self.input_queue = queue.Queue()

        self._started = False
        self._should_shutdown = False

        self._sinks = []

        self._ind2choice = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

        self._unique_room_names = (
            unique_room_names(self._config.floorplan_graph.nodes)[0]
            if self._config.use_floorplan_rooms_as_choices
            else None
        )
        self._log_folder = (
            pathlib.Path(self._config.log_folder).expanduser()
            if self._config.log_folder
            else None
        )

    @property
    def question(self) -> str:
        """Get the current question for the VLM."""
        return self._question

    @question.setter
    def question(self, value: str) -> None:
        """Set the question for the VLM.
        :param value: The question to set.
        """
        self._question = value

    def add_sink(self, sink: callable) -> None:
        """Add a sink to receive the output of the planner.
        :param sink: A callable that takes a EQAPlannerOutput as input.
        """
        self._sinks.append(sink)

    def start(self) -> None:
        """Start the planner."""
        if not self._started:
            self._started = True
            self._thread = threading.Thread(target=self._spin)
            self._thread.start()

    def stop(self) -> None:
        """Stop the planner."""
        self._should_shutdown = True
        if self._started:
            self._thread.join()
            self._started = False

        self._started = False
        self._should_shutdown = False
        self.history = []

    def _spin(self) -> None:
        while not self._should_shutdown:
            try:
                hlp_input = self.input_queue.get(timeout=0.1)
                self.spin_once(hlp_input)
            except queue.Empty:
                continue

    def spin_once(self, eqa_planner_input: EQAPlannerInput) -> None:
        """Process a single input through the planner and send the output to the sinks.
        :param eqa_planner_input: The input to process.
        """
        eqa_planner_output = self._reason(eqa_planner_input)
        for sink in self._sinks:
            sink(eqa_planner_output)

    def _vlm_output_to_eqa_planner_output(
        self, vlm_output: dict[str, Any], node_id_to_index: dict[str, int]
    ) -> tuple[EQAPlannerOutput, str, list[str]]:
        """Convert the VLM output to a EQAPlannerOutput.
        :param vlm_output: Output from the VLM.
        :param node_id_to_index: Mapping from node ID strings to integer indices.
        :return: An EQAPlannerOutput object, target room ID string,
                 and target object IDs string.
        """
        output = EQAPlannerOutput(
            answer=vlm_output.get("answer", ""),
            answered=vlm_output.get("answered", False),
            mode=str_to_output_mode(vlm_output.get("mode", "explore")),
            sg_description=vlm_output.get("sg_description", ""),
            floorplan_description=vlm_output.get("floorplan_description", ""),
            image_descriptions=vlm_output.get("image_descriptions", []),
            new_labels=vlm_output.get("new_labels", []),
            reasoning=vlm_output.get("reasoning", ""),
            confidence=vlm_output.get("confidence", 0.0),
            target_room_label=vlm_output.get("target_room_label"),
        )
        target_room_id_str = vlm_output.get("target_room_id") or "null"
        if vlm_output.get("target_room_id") is not None:
            output.target_room_id = node_id_to_index.get(
                vlm_output["target_room_id"], -1
            )
        target_object_ids_str = vlm_output.get("target_object_ids") or []
        if vlm_output.get("target_object_ids") is not None:
            output.target_object_ids = [
                node_id_to_index.get(obj_id)
                for obj_id in vlm_output["target_object_ids"]
                if node_id_to_index.get(obj_id) is not None
            ]
        output.validate()
        return output, target_room_id_str, target_object_ids_str

    def _floorplan_room_labels(self) -> list[str]:
        """Get the unique semantic room labels from the floorplan."""
        if self._unique_room_names:
            return copy.deepcopy(self._unique_room_names)
        return unique_room_names(self._config.floorplan_graph.nodes)[0]

    def _floorplan_instance_adjacency(self) -> dict[str, set[str]]:
        """Build floorplan adjacency over original floorplan instance nodes."""
        adjacency = {node: set() for node in self._config.floorplan_graph.nodes}
        for edge in self._config.floorplan_graph.edges:
            if len(edge) != 2:
                continue
            a, b = edge
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        return adjacency

    def _floorplan_distance_to_target_instances(
        self, start_node: str, target_label: str, adjacency: dict[str, set[str]]
    ) -> int | None:
        """Shortest distance from a floorplan node to any target-label instance."""
        target_label = clean_room_name(target_label)
        if target_label in room_name_parts(start_node):
            return 0
        if start_node not in adjacency:
            return None

        visited = {start_node}
        queue = collections.deque([(start_node, 0)])
        while queue:
            node, distance = queue.popleft()
            for neighbor in adjacency.get(node, set()):
                if neighbor in visited:
                    continue
                if target_label in room_name_parts(neighbor):
                    return distance + 1
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return None

    def _compute_floorplan_progress_scores(
        self, current_room_label: str | None, target_room_label: str | None
    ) -> tuple[list[str], list[float]]:
        """Compute next-room label distribution for the find_room strategy."""
        room_labels = self._floorplan_room_labels()
        if not room_labels:
            return [], []
        if current_room_label is None or target_room_label is None:
            return room_labels, [0.0] * len(room_labels)

        current_room_labels = set(room_name_parts(current_room_label))
        target_room_label = clean_room_name(target_room_label)
        adjacency = self._floorplan_instance_adjacency()
        current_nodes = [
            node
            for node in self._config.floorplan_graph.nodes
            if any(label in current_room_labels for label in room_name_parts(node))
        ]
        if not current_nodes:
            return room_labels, [0.0] * len(room_labels)

        candidate_distance_by_label: dict[str, int] = {}
        for current_node in current_nodes:
            for neighbor in adjacency.get(current_node, set()):
                neighbor_distance = self._floorplan_distance_to_target_instances(
                    neighbor, target_room_label, adjacency
                )
                if neighbor_distance is None:
                    continue
                for neighbor_label in room_name_parts(neighbor):
                    if neighbor_label not in room_labels:
                        continue
                    candidate_distance_by_label[neighbor_label] = min(
                        candidate_distance_by_label.get(
                            neighbor_label, neighbor_distance
                        ),
                        neighbor_distance,
                    )

        if not candidate_distance_by_label:
            return room_labels, [0.0] * len(room_labels)

        candidate_labels = list(candidate_distance_by_label.keys())
        candidate_weights = softmax(
            [-float(candidate_distance_by_label[label]) for label in candidate_labels]
        )

        score_by_label = {label: 0.0 for label in room_labels}
        for label, weight in zip(candidate_labels, candidate_weights, strict=True):
            score_by_label[label] = weight

        return room_labels, [score_by_label[label] for label in room_labels]

    def _add_find_room_output_fields(
        self,
        output: EQAPlannerOutput,
        current_room_label: str | None,
    ) -> None:
        """Populate prompt metadata and floorplan progress for find_room outputs."""
        if (
            output.mode != str_to_output_mode("find_room")
            or output.target_room_label is None
        ):
            return

        target_room_label = clean_room_name(output.target_room_label)
        floorplan_room_labels = self._floorplan_room_labels()
        if target_room_label not in floorplan_room_labels:
            LoggerWarn.warning(
                "Invalid EQA Planner Output: FIND_ROOM target_room_label "
                f"'{target_room_label}' is not present in the floorplan."
            )
            output.valid = False
            return

        output.target_room_label = target_room_label
        output.transition_prompts = [
            f"doorway to a {target_room_label}",
            f"entrance to a {target_room_label}",
            f"hallway leading to a {target_room_label}",
            f"opening into a {target_room_label}",
        ]
        (
            output.floorplan_room_labels,
            output.floorplan_progress_scores,
        ) = self._compute_floorplan_progress_scores(
            current_room_label, target_room_label
        )

    def _json_safe(self, value: Any) -> Any:
        """Convert planner data into JSON-serializable values."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, pathlib.Path):
            return str(value)
        if hasattr(value, "value"):
            return value.value
        return value

    def _write_json(self, path: pathlib.Path, payload: Any) -> None:
        """Write a JSON payload with consistent formatting."""
        with path.open("w") as f:
            json.dump(self._json_safe(payload), f, indent=2)

    def _safe_file_stem(self, value: str) -> str:
        """Create a readable filename stem from an image-ordering string."""
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
        return stem[:80] if stem else "image"

    def _scene_graph_object_classes(
        self, scene_graph_input: dict[str, Any], object_ids: list[str] | None
    ) -> dict[str, str | None]:
        """Look up object classes from the scene-graph payload sent to the VLM."""
        if not object_ids:
            return {}

        object_classes = {object_id: None for object_id in object_ids}
        for room_data in scene_graph_input.values():
            if not isinstance(room_data, dict):
                continue
            for obj in room_data.get("objects", []):
                if not isinstance(obj, dict):
                    continue
                obj_id = obj.get("id")
                if obj_id in object_classes:
                    object_classes[obj_id] = obj.get("label")
        return object_classes

    def _planner_output_to_log(
        self,
        output: EQAPlannerOutput,
        raw_vlm_output: dict[str, Any],
        success: bool,
        target_room_id_str: str,
        target_object_ids_str: list[str],
        scene_graph_input: dict[str, Any],
        node_id_to_index: dict[str, int],
    ) -> dict[str, Any]:
        """Build the output payload saved for an iteration."""
        converted_output = copy.deepcopy(output.__dict__)
        converted_output["mode"] = str(output.mode)

        payload = {
            "success": success,
            "raw_vlm_output": raw_vlm_output,
            "converted_output": converted_output,
            "target_room_id": {
                "vlm_id": target_room_id_str,
                "node_id": output.target_room_id,
            },
        }

        if output.mode == str_to_output_mode("go_to_objects"):
            object_classes = self._scene_graph_object_classes(
                scene_graph_input, target_object_ids_str
            )
            payload["target_objects"] = [
                {
                    "vlm_id": obj_id,
                    "node_id": node_id_to_index.get(obj_id),
                    "class": object_classes.get(obj_id),
                }
                for obj_id in target_object_ids_str
            ]
        else:
            payload["target_object_ids"] = target_object_ids_str

        return payload

    def _log_iteration(
        self,
        iteration: int,
        dsg_vlm_format: DsgVLMFormat,
        floorplan_input: dict[str, Any],
        output_payload: dict[str, Any],
    ) -> None:
        """Persist the VLM inputs and planner output for one iteration."""
        if self._log_folder is None:
            return

        iteration_folder = self._log_folder / str(iteration)
        images_folder = iteration_folder / "images"
        images_folder.mkdir(parents=True, exist_ok=True)

        scene_graph_input = json.loads(dsg_vlm_format.dsg)
        self._write_json(iteration_folder / "scene_graph_input.json", scene_graph_input)
        self._write_json(iteration_folder / "floorplan_input.json", floorplan_input)
        self._write_json(iteration_folder / "output.json", output_payload)

        image_manifest = []
        for i, (image, ordering) in enumerate(
            zip(
                dsg_vlm_format.images,
                dsg_vlm_format.images_ordering,
                strict=True,
            )
        ):
            filename = f"{i:03d}_{self._safe_file_stem(ordering)}.png"
            image_path = images_folder / filename
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image_manifest.append(
                {
                    "index": i,
                    "ordering": ordering,
                    "path": str(pathlib.Path("images") / filename),
                }
            )

        self._write_json(iteration_folder / "images.json", image_manifest)

    def _add_history_to_prompt(self) -> str:
        """Add the history of previous outputs to the prompt.
        :return: A string describing the history of previous outputs.
        """
        history_str = "HISTORY OF PREVIOUS OUTPUTS:\n"
        for i, output in enumerate(self.history[-self._config.history_length :]):
            history_str += f"Output {i + 1}. "
            history_str += f"Answered: {output.answered}, Answer: {output.answer}, "
            history_str += f"Mode: {output.mode}, "
            if output.current_agent_state is not None:
                history_str += f"Agent State: {output.current_agent_state}, "
            if output.target_room_id is not None:
                history_str += f"Target Room ID: {output.target_room_id}, "
            if output.target_room_label is not None:
                history_str += f"Target Room Label: {output.target_room_label}, "
            if output.target_object_ids is not None:
                history_str += f"Target Object IDs: {output.target_object_ids}, "
            history_str += f"Confidence: {output.confidence}\n"
        return history_str

    def _get_room_parent_id(
        self, graph: spark_dsg.DynamicSceneGraph, node: Any
    ) -> int | None:
        """Resolve the room parent for an object/agent node through its nav parent."""
        if not node.has_parent():
            return None

        nav_node = graph.get_node(node.get_parent())
        if not nav_node.has_parent():
            return None

        room_id = nav_node.get_parent()
        return room_id if graph.has_node(room_id) else None

    def _get_room_label(self, room_node: Any) -> str | None:
        """Read the room label using whichever attribute the bindings expose."""
        for attr_name in ("label", "name"):
            room_label = getattr(room_node.attributes, attr_name, None)
            if room_label:
                return room_label
        return None

    def _set_room_label(self, room_node: Any, room_label: str) -> None:
        """Write the room label back to the DSG room node."""
        wrote_label = False
        for attr_name in ("label", "name"):
            if hasattr(room_node.attributes, attr_name):
                setattr(room_node.attributes, attr_name, room_label)
                wrote_label = True

        if not wrote_label:
            LoggerWarn.warning(
                f"Room node {room_node.id.str()} has no writable label attribute"
            )

    def _build_room_classification_inputs(
        self, graph: spark_dsg.DynamicSceneGraph
    ) -> list[RoomClassificationInput]:
        """Build room-classification inputs from the current graph."""
        if not graph.has_layer(spark_dsg.DsgLayers.ROOMS):
            return []

        room_layer = graph.get_layer(spark_dsg.DsgLayers.ROOMS)
        room_data = {
            node.id.value: {"objects": [], "images": [], "image_features": []}
            for node in room_layer.nodes
        }
        if not room_data:
            return []

        if graph.has_layer(spark_dsg.DsgLayers.OBJECTS):
            layer_names = graph.layer_names
            labelspace = graph.get_labelspace(
                layer_names[spark_dsg.DsgLayers.OBJECTS].layer
            )
            object_layer = graph.get_layer(spark_dsg.DsgLayers.OBJECTS)
            for node in object_layer.nodes:
                room_id = self._get_room_parent_id(graph, node)
                if room_id is None or room_id not in room_data:
                    continue
                room_data[room_id]["objects"].append(
                    labelspace.get_category(node.attributes.semantic_label)
                )

        if graph.has_layer(spark_dsg.DsgLayers.AGENTS):
            agent_layer_id = graph.get_layer_key(spark_dsg.DsgLayers.AGENTS).layer
            agent_partition = None
            for layer_key in graph.layer_keys:
                if layer_key.partition != 0:
                    agent_partition = layer_key.partition
                    break

            if agent_partition is not None:
                agent_layer = graph.get_layer(agent_layer_id, agent_partition)
                for node in agent_layer.nodes:
                    attrs = node.attributes
                    if attrs.image.shape[0] == 0:
                        continue

                    room_id = self._get_room_parent_id(graph, node)
                    if room_id is None or room_id not in room_data:
                        continue

                    room_data[room_id]["images"].append(attrs.image)
                    room_data[room_id]["image_features"].append(attrs.image_feature)

        room_inputs = []
        for room_id, data in room_data.items():
            selected_images = self._image_sampler.sample_images(
                data["images"], data["image_features"]
            )
            if not data["objects"] and not selected_images:
                continue

            room_inputs.append(
                RoomClassificationInput(
                    room_id=room_id,
                    images=selected_images,
                    objects=data["objects"],
                )
            )

        return room_inputs

    def _classify_rooms(self, graph: spark_dsg.DynamicSceneGraph) -> None:
        """Classify rooms and assign the predicted labels directly to the graph."""
        if self._room_classifier is None:
            return

        room_inputs = self._build_room_classification_inputs(graph)
        if not room_inputs:
            return

        try:
            result = self._room_classifier.classify(
                room_inputs, self._unique_room_names
            )
        except Exception as exc:
            LoggerWarn.warning(f"Room classification failed: {exc}")
            return

        if not (len(result.id) == len(result.labels) == len(result.confidence)):
            LoggerWarn.warning("Room classification output has mismatched lengths")
            return

        for room_id, room_label, confidence in zip(
            result.id, result.labels, result.confidence, strict=False
        ):
            if confidence < self._config.room_classification_min_confidence:
                continue
            if not graph.has_node(room_id):
                LoggerWarn.warning(
                    f"Classified room {room_id} does not exist in planner graph"
                )
                continue

            room_node = graph.get_node(room_id)
            self._set_room_label(room_node, room_label)

    def _reason(self, eqa_planner_input: EQAPlannerInput) -> EQAPlannerOutput:
        """Reason about the input and produce an output.
        :param eqa_planner_input: The input to reason about.
        :return: An EQAPlannerOutput object containing the answer and reasoning.
        """
        prompt = ""
        iteration = len(self.history)
        self._classify_rooms(eqa_planner_input.graph)
        floorplan_input = {
            "use_floorplan_prior": self._config.use_floorplan_prior,
            "floorplan_source": self._config.floorplan_source,
            "floorplan_json_path": self._config.floorplan_json_path,
            "floorplan_graph": {
                "nodes": self._config.floorplan_graph.nodes,
                "edges": self._config.floorplan_graph.edges,
            },
        }

        # Add floorplan prior to the prompt if configured to do so
        if self._config.use_floorplan_prior and self._config.floorplan_graph.nodes:
            prompt += (
                f"FLOORPLAN PRIOR: \n {self._config.floorplan_graph.to_json()} \n\n"
            )

        # 1. Parse input and format it as prompt and images for the VLM
        dsg_vlm_format: DsgVLMFormat = to_vlm_format(
            eqa_planner_input.graph,
            self._image_sampler,
            stitch_room_images=self._config.stitch_room_images,
            room_image_stitching=self._config.room_image_stitching,
        )
        prompt += f"SCENE GRAPH: \n {dsg_vlm_format.dsg} \n\n"

        # 2. Add current image
        dsg_vlm_format.images.append(eqa_planner_input.current_view)
        dsg_vlm_format.images_ordering.append("current_view")

        # 3. Add history of previous outputs to the prompt
        if len(self.history) > 0:
            history_str = self._add_history_to_prompt()
            prompt += history_str + "\n\n"

        # 4. Add object views if available
        if (
            eqa_planner_input.objects_views is not None
            and len(eqa_planner_input.objects_views) > 0
        ):
            dsg_vlm_format.images.extend(list(eqa_planner_input.objects_views.values()))
            dsg_vlm_format.images_ordering.extend(
                f"object_view_{i}" for i in range(len(eqa_planner_input.objects_views))
            )
            prompt += (
                f"Added {len(eqa_planner_input.objects_views)} "
                f"object views to the input, "
                f"since your previous mode was 'go_to_objects' "
                f"and you specified target_object_ids.\n\n"
            )

        if self._config.debug_input:
            LoggerWarn.warning(f"Prompt for VLM:\n{dsg_vlm_format.images_ordering}")

        # 5. Add current pose to prompt
        current_state = None
        current_room_label = None
        if dsg_vlm_format.agent_layer_partition is not None:
            agent_layer = eqa_planner_input.graph.get_layer(
                dsg_vlm_format.agent_layer_id, dsg_vlm_format.agent_layer_partition
            )
            if agent_layer.num_nodes() > 0:
                current_agent_node = list(agent_layer.nodes)[-1]
                current_state = current_agent_node.attributes.position
                # Try to get room id and room label for current pose
                room_id = None
                room_label = None
                if current_agent_node.has_parent():
                    nav_node = eqa_planner_input.graph.get_node(
                        current_agent_node.get_parent()
                    )
                    if nav_node.has_parent():
                        room_node = eqa_planner_input.graph.get_node(
                            nav_node.get_parent()
                        )
                        room_id = room_node.id.str()
                        room_label = self._get_room_label(room_node)
                        current_room_label = room_label
                current_state = (
                    f"CURRENT AGENT STATE: "
                    f"position {current_state.tolist()}, "
                    f"room_id {room_id if room_id else 'null'}, "
                    f"room_label {room_label if room_label else 'null'}"
                )
                prompt += current_state + "\n\n"

        # Add choices to the prompt if configured to do so
        if self._config.use_choices and self._config.choices:
            choices_str = "ANSWER CHOICES:\n"
            for i, choice in enumerate(self._config.choices):
                choices_str += f"{self._ind2choice.get(i, str(i))}: {choice}\n"
            prompt += choices_str + "\n\n"

        # 6. Generate output using the VLM
        prompt += f"QUESTION: {self.question}"
        output, success = self._vlm.answer(
            prompt, dsg_vlm_format.images, dsg_vlm_format.images_ordering
        )

        # 7. Convert the VLM output to HighLevelPlannerOutput
        target_room_id_str = "null"
        target_object_ids_str = []
        if success:
            eqa_planner_output, target_room_id_str, target_object_ids_str = (
                self._vlm_output_to_eqa_planner_output(
                    output, dsg_vlm_format.node_id_to_index
                )
            )
        else:
            eqa_planner_output = EQAPlannerOutput(
                answer="", answered=False, valid=False
            )

        self._add_find_room_output_fields(eqa_planner_output, current_room_label)
        eqa_planner_output.valid = eqa_planner_output.valid and success
        eqa_planner_output.timestamp = eqa_planner_input.timestamp
        eqa_planner_output.frame = eqa_planner_input.frame
        eqa_planner_output.current_agent_state = current_state
        scene_graph_input = json.loads(dsg_vlm_format.dsg)
        output_payload = self._planner_output_to_log(
            eqa_planner_output,
            output if success else {},
            success,
            target_room_id_str,
            target_object_ids_str,
            scene_graph_input,
            dsg_vlm_format.node_id_to_index,
        )
        self._log_iteration(
            iteration,
            dsg_vlm_format,
            floorplan_input,
            output_payload,
        )
        self.history.append(copy.deepcopy(eqa_planner_output))
        self.history[-1].target_room_id = target_room_id_str
        self.history[-1].target_object_ids = target_object_ids_str

        # if (eqa_planner_output.mode == str_to_output_mode("find_room") and
        #    eqa_planner_output.target_room_label is not None):
        #     # Save graph to file for debugging. Save floorplan rooms. Save question.
        #     debug_output = {
        #         "floorplan_rooms": self._unique_room_names,
        #         "floorplan_nodes": self._config.floorplan_graph.nodes,
        #         "floorplan_edges": self._config.floorplan_graph.edges,
        #         "question": self.question,
        #         "target_room": clean_room_name(eqa_planner_output.target_room_label)
        #     }
        #     with open("eqa_planner_output_debug.json", "w") as f:
        #         json.dump(debug_output, f, indent=2)
        #         eqa_planner_input.graph.save("eqa_planner_output_debug_dsg.sparkdsg",
        #                                      include_mesh=True)

        return eqa_planner_output
