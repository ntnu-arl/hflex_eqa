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
"""VLM based Embodied Question Answering."""

import json
import queue
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import vlms_python
from spark_config import Config, config_field

from hflex_eqa_python.conversions import DsgVLMFormat, to_vlm_format
from hflex_eqa_python.eqa.eqa_input import (
    EQAInput,
)
from hflex_eqa_python.eqa.eqa_output import (
    EQAOutput,
)
from hflex_eqa_python.misc import LoggerWarn


@dataclass
class EmbodiedQuestionAnsweringConfig(Config):
    """Configuration for the Embodied Question Answering module."""

    vlm: Any = config_field("eqa_planner", default="openai")
    image_sampler: Any = config_field("image_sampling", default="object")
    max_images_per_room: int = 5
    debug_input: bool = False
    question: str = ""


class EmbodiedQuestionAnswering:
    """Embodied Question Answering module."""

    def __init__(self, config: EmbodiedQuestionAnsweringConfig) -> None:
        """Initialize the module.
        :param config: Configuration for the module.
        """
        self._config = config
        self._vlm = self._config.vlm.create()
        self.question = self._config.question
        self._image_sampler = self._config.image_sampler.create()
        self._image_sampler.set_question(self.question)

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
        """Add a sink to receive the output.
        :param sink: A callable that takes an EQAOutput as input.
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

    def _spin(self) -> None:
        while not self._should_shutdown:
            try:
                eqa_input = self.input_queue.get(timeout=0.1)
                self.spin_once(eqa_input)
            except queue.Empty:
                continue

    def spin_once(self, eqa_input: EQAInput) -> None:
        """Process a single input and produce output.
        :param eqa_input: The input to process.
        """
        eqa_output = self._answer(eqa_input)
        for sink in self._sinks:
            sink(eqa_output)

    def _vlm_output_to_eqa_output(self, vlm_output: str) -> EQAOutput:
        """Convert the VLM output to an EQAOutput.
        :param vlm_output: The output from the VLM.
        :return: The converted EQAOutput.
        """
        return EQAOutput(**vlm_output)

    def _answer(self, eqa_input: EQAInput) -> EQAOutput:
        """Answer the question based on the input.
        :param eqa_input: The input to process.
        :return: The output containing the answer.
        """
        # 1. Parse input and format it as prompt and images for the VLM
        dsg_vlm_format: DsgVLMFormat = to_vlm_format(
            eqa_input.graph, self._image_sampler
        )
        prompt = f"SCENE GRAPH: \n {dsg_vlm_format.dsg} \n\n"

        # 2. Add current image
        dsg_vlm_format.images.append(eqa_input.current_view)
        dsg_vlm_format.images_ordering.append("current_view")
        if eqa_input.objects_views is not None and len(eqa_input.objects_views) > 0:
            dsg_vlm_format.images.extend(eqa_input.objects_views)
            dsg_vlm_format.images_ordering.extend(
                f"object_view_{i}" for i in range(len(eqa_input.objects_views))
            )
            prompt += (
                f"Added {len(eqa_input.objects_views)} object views to the input.\n\n"
            )
        if self._config.debug_input:
            with open("eqa_input_debug.json", "w") as f:
                json.dump(eval(dsg_vlm_format.dsg), f, indent=2)
            for i, img in enumerate(dsg_vlm_format.images):
                cv2.imwrite(
                    f"eqa_input_image_{dsg_vlm_format.images_ordering[i]}_{i}.png",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
            LoggerWarn.warning("Saved EQA input DSG JSON to eqa_input_debug.json")
            LoggerWarn.warning(f"Prompt for VLM:\n{dsg_vlm_format.images_ordering}")

        # 3. Generate output using the VLM
        prompt += f"QUESTION: {self.question}"
        output, success = self._vlm.answer(
            prompt, dsg_vlm_format.images, dsg_vlm_format.images_ordering
        )

        # 4. Format output
        if success:
            eqa_output = self._vlm_output_to_eqa_output(output)
        eqa_output.valid = success
        eqa_output.frame = eqa_input.frame
        eqa_output.timestamp = eqa_input.timestamp

        return eqa_output
