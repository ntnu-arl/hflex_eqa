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
"""VLM based High-Level Planner Input."""

import enum
from dataclasses import dataclass

import numpy as np

from hflex_eqa_python.misc import LoggerWarn


class OutputMode(enum.Enum):
    """Output mode for the High-Level Planner."""

    EXPLORE_ROOM = "explore_room"
    FIND_ROOM = "find_room"
    GO_TO_OBJECTS = "go_to_objects"

    def __str__(self) -> str:
        return self.value


def str_to_output_mode(mode_str: str) -> OutputMode:
    """Convert a string to an OutputMode enum value."""
    try:
        return OutputMode(mode_str)
    except ValueError:
        LoggerWarn.warning(
            f"Invalid output mode string: {mode_str}. Defaulting to EXPLORE_ROOM."
        )
        return OutputMode.EXPLORE_ROOM


@dataclass
class EQAPlannerOutput:
    """Output for the EQA Planner."""

    answer: str
    answered: bool
    mode: OutputMode = OutputMode.EXPLORE_ROOM
    target_room_id: int | None = None
    target_room_label: str | None = None
    target_object_ids: list[int] | None = None
    transition_prompts: list[str] | None = None
    transition_prompt_embeddings: list[np.ndarray] | None = None
    floorplan_room_labels: list[str] | None = None
    floorplan_progress_scores: list[float] | None = None
    floorplan_room_embeddings: list[np.ndarray] | None = None
    sg_description: str = ""
    floorplan_description: str = ""
    image_descriptions: list[str] | None = None
    new_labels: list[str] | None = None
    reasoning: str = ""
    confidence: float = 0.0
    valid: bool = False
    timestamp: int | None = None
    current_agent_state: np.ndarray | None = None
    frame: str = ""

    def validate(self) -> None:
        """Post-initialization checks."""
        if self.answered:
            self.valid = True
            return
        if self.mode == OutputMode.EXPLORE_ROOM and self.target_room_id is None:
            self.valid = False
            LoggerWarn.warning(
                "Invalid EQA Planner Output: "
                "EXPLORE_ROOM mode requires a target_room_id."
            )
            return
        if self.mode == OutputMode.GO_TO_OBJECTS and self.target_object_ids is None:
            self.valid = False
            LoggerWarn.warning(
                "Invalid EQA Planner Output: GO_TO_OBJECTS "
                "mode requires a target_object_ids."
            )
            return
        if self.mode == OutputMode.FIND_ROOM and self.target_room_label is None:
            self.valid = False
            LoggerWarn.warning(
                "Invalid EQA Planner Output: FIND_ROOM "
                "mode requires a target_room_label."
            )
            return
        self.valid = True
