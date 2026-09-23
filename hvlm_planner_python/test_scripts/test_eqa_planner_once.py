#!/usr/bin/env python3
# BSD 3-Clause License
#
# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
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
"""Run the EQA planner once on a saved DSG."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import spark_dsg._dsg_bindings as spark_dsg

from hvlm_planner_python import (  # noqa: E402
    EQAPlanner,
    EQAPlannerConfig,
    EQAPlannerInput,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a saved graph and run EQAPlanner.spin_once on it."
    )
    parser.add_argument(
        "--graph", type=Path, required=True, help="Path to a saved DSG file"
    )
    parser.add_argument(
        "--question",
        default="What room am I in?",
        type=str,
        help="Question to ask the EQA planner",
    )
    parser.add_argument(
        "--frame",
        default="world",
        type=str,
        help="Frame id to store in the fabricated EQAPlannerInput",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Dummy current-view image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Dummy current-view image width",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        type=str,
        help="OpenAI model name for both planner answering and room classification",
    )
    parser.add_argument(
        "--debug-input",
        action="store_true",
        help="Enable EQA planner debug input dumping",
    )
    return parser.parse_args()


def _get_room_label(room_node) -> str | None:
    for attr_name in ("label", "name"):
        room_label = getattr(room_node.attributes, attr_name, None)
        if room_label:
            return room_label
    return None


def _collect_room_labels(graph: spark_dsg.DynamicSceneGraph) -> dict[str, str | None]:
    if not graph.has_layer(spark_dsg.DsgLayers.ROOMS):
        return {}

    room_labels = {}
    room_layer = graph.get_layer(spark_dsg.DsgLayers.ROOMS)
    for room_node in room_layer.nodes:
        room_labels[room_node.id.str()] = _get_room_label(room_node)
    return room_labels


def _make_dummy_rgb_image(height: int, width: int) -> np.ndarray:
    """Create a simple synthetic RGB image for the planner input."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 96
    image[:, :, 2] = 160
    return image


def _build_config(args: argparse.Namespace) -> EQAPlannerConfig:
    config = EQAPlannerConfig()
    config.update(
        {
            "question": args.question,
            "debug_input": args.debug_input,
            "classify_rooms": True,
            "vlm": {
                "type": "openai",
                "client_config": {"model": args.model},
            },
            "room_classification": {
                "type": "openai",
                "client_config": {"model": args.model},
                "system_prompt_path": "/developer/ros2_hydra_ws/src/vlms_ros/vlms_ros/config/room_classification/system_prompt.txt",
                "log": True,
            },
            "image_sampler": {
                "type": "question",
                "max_images_per_room": 3,
                "clip_model": {"type": "clip", "model_name": "ViT-B/32"},
            },
        }
    )
    return config


def main() -> None:
    args = _parse_args()
    graph_path = Path(args.graph).expanduser().absolute()
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph path does not exist: {graph_path}")

    print(f"Loading graph from: {graph_path}")
    graph = spark_dsg.DynamicSceneGraph.load(str(graph_path))
    if graph is None:
        raise RuntimeError(f"Failed to load graph from {graph_path}")

    before_labels = _collect_room_labels(graph)
    print("Room labels before planner call:")
    print(json.dumps(before_labels, indent=2))

    config = _build_config(args)
    planner = EQAPlanner(config)
    planner.question = args.question
    planner._image_sampler.set_question(args.question)

    outputs = []
    planner.add_sink(outputs.append)

    planner_input = EQAPlannerInput(
        graph=graph,
        current_view=_make_dummy_rgb_image(args.height, args.width),
        objects_views={},
        timestamp=time.time_ns(),
        frame=args.frame,
    )

    print("Running EQAPlanner.spin_once(...)")
    planner.spin_once(planner_input)

    after_labels = _collect_room_labels(graph)
    print("Room labels after planner call:")
    print(json.dumps(after_labels, indent=2))

    if not outputs:
        print("No planner output was captured from sinks.")
        return

    output = outputs[-1]
    print("Planner output summary:")
    print(
        json.dumps(
            {
                "answer": output.answer,
                "answered": output.answered,
                "mode": str(output.mode),
                "target_room_id": output.target_room_id,
                "target_object_ids": output.target_object_ids,
                "confidence": output.confidence,
                "valid": output.valid,
                "current_agent_state": output.current_agent_state,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
