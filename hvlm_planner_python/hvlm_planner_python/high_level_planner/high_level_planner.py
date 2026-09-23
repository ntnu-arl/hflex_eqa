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
"""VLM based High-Level Planner"""

import json
import queue
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import vlms_python
from spark_config import Config, config_field

from hvlm_planner_python.conversions import DsgVLMFormat, to_vlm_format
from hvlm_planner_python.high_level_planner.high_level_input import (
    HighLevelPlannerInput,
)
from hvlm_planner_python.high_level_planner.high_level_output import (
    HighLevelPlannerOutput,
    str_to_output_mode,
)
from hvlm_planner_python.misc import LoggerWarn


@dataclass
class HighLevelPlannerConfig(Config):
    """Configuration for the High-Level Planner."""

    vlm: Any = config_field("vlm_planner", default="openai")
    image_sampler: Any = config_field("image_sampling", default="object")
    history_length: int = 5
    max_images_per_room: int = 5
    debug_input: bool = False
    question: str = ""


class HighLevelPlanner:
    """High-Level Planner using VLMs."""

    def __init__(self, config: HighLevelPlannerConfig) -> None:
        """Construct High-Level Planner.
        :param config: Configuration for the High-Level Planner.
        """

        self._config = config
        self._vlm = self._config.vlm.create()
        self.question = self._config.question
        self._image_sampler = self._config.image_sampler.create()
        self._image_sampler.set_question(self.question)

        self.history: list[HighLevelPlannerOutput] = []

        self.input_queue = queue.Queue()

        self._started = False
        self._should_shutdown = False

        self._sinks = []

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
        :param sink: A callable that takes a HighLevelPlannerOutput as input.
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

    def spin_once(self, hlp_input: HighLevelPlannerInput) -> None:
        """Process a single input through the planner and send the output to the sinks.
        :param hlp_input: The input to process.
        """
        hlp_output = self._plan(hlp_input)
        for sink in self._sinks:
            sink(hlp_output)

    def _vlm_output_to_hlp_output(
        self, vlm_output: dict[str, Any], node_id_to_index: dict[str, int]
    ) -> HighLevelPlannerOutput:
        """Convert the VLM output to a HighLevelPlannerOutput.
        :param vlm_output: Output from the VLM.
        :param node_id_to_index: Mapping from node ID strings to integer indices.
        :return: A HighLevelPlannerOutput object.
        """
        # TODO: convert target_room_id and target_object_id to int
        output = HighLevelPlannerOutput(
            mode=str_to_output_mode(vlm_output.get("mode", "explore")),
            sg_description=vlm_output.get("sg_description", ""),
            images_description=vlm_output.get("images_description", []),
            reasoning=vlm_output.get("reasoning", ""),
            confidence=vlm_output.get("confidence", 0.0),
        )
        if vlm_output.get("target_room_id") is not None:
            output.target_room_id = node_id_to_index.get(
                vlm_output["target_room_id"], None
            )
        if vlm_output.get("target_object_ids") is not None:
            output.target_object_ids = [
                node_id_to_index.get(obj_id)
                for obj_id in vlm_output["target_object_ids"]
            ]
        output.validate()
        return output

    def _add_history_to_prompt(self) -> str:
        """Add the history of previous outputs to the prompt.
        :return: A string describing the history of previous outputs.
        """
        history_str = "HISTORY OF PREVIOUS OUTPUTS:\n"
        for i, output in enumerate(self.history[-self._config.history_length :]):
            history_str += f"Output {i + 1}. Mode: {output.mode}, "
            if output.current_agent_state is not None:
                history_str += f"Agent State: {output.current_agent_state}, "
            if output.target_room_id is not None:
                history_str += f"Target Room ID: {output.target_room_id}, "
            if output.target_object_ids is not None:
                history_str += f"Target Object IDs: {output.target_object_ids}, "
            history_str += f"Confidence: {output.confidence}\n"
        return history_str

    def _plan(self, hlp_input: HighLevelPlannerInput) -> HighLevelPlannerOutput:
        """Plan the next action based on the input.
        :param hlp_input: Input for the High-Level Planner.
        :return: Output for the High-Level Planner.
        """
        # 1. Parse input and format it as prompt and images for the VLM
        dsg_vlm_format: DsgVLMFormat = to_vlm_format(
            hlp_input.graph, self._image_sampler
        )
        prompt = f"SCENE GRAPH: \n {dsg_vlm_format.dsg} \n\n"

        # 2. Add current image
        dsg_vlm_format.images.append(hlp_input.current_view)
        dsg_vlm_format.images_ordering.append("current_view")

        # 3. Add history of previous outputs to the prompt
        if len(self.history) > 0:
            history_str = self._add_history_to_prompt()
            prompt += history_str + "\n\n"
        if hlp_input.objects_views is not None and len(hlp_input.objects_views) > 0:
            dsg_vlm_format.images.extend(hlp_input.objects_views)
            dsg_vlm_format.images_ordering.extend(
                f"object_view_{i}" for i in range(len(hlp_input.objects_views))
            )
            prompt += (
                f"Added {len(hlp_input.objects_views)} object views to the input, "
                f"since your previous mode was 'go_to_objects' "
                f"and you specified target_object_ids.\n\n"
            )

        if self._config.debug_input:
            with open("hlp_input_debug.json", "w") as f:
                json.dump(eval(dsg_vlm_format.dsg), f, indent=2)
            for i, img in enumerate(dsg_vlm_format.images):
                cv2.imwrite(
                    f"hlp_input_image_{dsg_vlm_format.images_ordering[i]}_{i}.png",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
            LoggerWarn.warning("Saved HLP input DSG JSON to hlp_input_debug.json")
            LoggerWarn.warning(f"Prompt for VLM:\n{dsg_vlm_format.images_ordering}")

        # 4. Add current pose to prompt
        current_state = None
        if dsg_vlm_format.agent_layer_partition is not None:
            agent_layer = hlp_input.graph.get_layer(
                dsg_vlm_format.agent_layer_id, dsg_vlm_format.agent_layer_partition
            )
            if agent_layer.num_nodes() > 0:
                current_agent_node = list(agent_layer.nodes)[-1]
                current_state = current_agent_node.attributes.position
                # Try to get room id and room label for current pose
                room_id = None
                room_label = None
                if current_agent_node.has_parent():
                    nav_node = hlp_input.graph.get_node(current_agent_node.get_parent())
                    if nav_node.has_parent():
                        room_node = hlp_input.graph.get_node(nav_node.get_parent())
                        room_id = room_node.id.str()
                        room_label = room_node.attributes.name
                current_state = (
                    f"CURRENT AGENT STATE: "
                    f"position {current_state.tolist()}, "
                    f"room_id {room_id if room_id else 'null'}, "
                    f"room_label {room_label if room_label else 'null'}"
                )
                prompt += current_state + "\n\n"

        # 5. Generate output using the VLM
        prompt += f"QUESTION: {self.question}"
        output, success = self._vlm.plan(
            prompt, dsg_vlm_format.images, dsg_vlm_format.images_ordering
        )

        # 6. Convert the VLM output to HighLevelPlannerOutput
        if success:
            hlp_output = self._vlm_output_to_hlp_output(
                output, dsg_vlm_format.node_id_to_index
            )
        hlp_output.valid = hlp_output.valid and success
        hlp_output.timestamp = hlp_input.timestamp
        hlp_output.frame = hlp_input.frame
        hlp_output.current_agent_state = current_state
        self.history.append(hlp_output)
        return hlp_output
