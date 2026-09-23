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
"""Replay EQA planner VLM calls from logged prompt inputs."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_SRC = Path(__file__).resolve().parents[3]
for package_path in (
    WORKSPACE_SRC / "vlms_ros" / "vlms_python",
    WORKSPACE_SRC / "hflex_eqa" / "hflex_eqa_python",
):
    if package_path.is_dir():
        sys.path.insert(0, str(package_path))


CHOICE_LABELS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay EQA planner VLM answers from an eqa_log directory containing "
            "one numbered subdirectory per planner iteration."
        )
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Main eqa_log directory containing iteration subdirectories.",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        required=True,
        help="System prompt file to use for the replayed VLM calls.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to append to each replayed prompt.",
    )
    parser.add_argument(
        "--choices",
        nargs="*",
        default=None,
        help=(
            "Optional multiple-choice answers. Quote each full choice, e.g. "
            "--choices 'on the sofa' 'in the closet'."
        ),
    )
    parser.add_argument(
        "--choice",
        action="append",
        dest="choice_items",
        default=None,
        help="Optional answer choice. Can be passed multiple times.",
    )
    parser.add_argument(
        "--use-floorplan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to include floorplan_input.json as FLOORPLAN PRIOR. If not "
            "set, each iteration uses the logged use_floorplan_prior value."
        ),
    )
    parser.add_argument(
        "--history-source",
        choices=("logged", "replayed", "none"),
        default="logged",
        help=(
            "Source for HISTORY OF PREVIOUS OUTPUTS. 'logged' best recreates the "
            "original prompt; 'replayed' makes later prompts depend on replayed "
            "answers."
        ),
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=1000,
        help="Number of previous outputs to include, matching EQAPlannerConfig.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.5",
        help="OpenAI model name for the replayed EQA VLM.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-iteration replay JSON files. Defaults to "
            "<log-dir>/vlm_replay_outputs."
        ),
    )
    parser.add_argument(
        "--aggregate-name",
        default="replayed_outputs.json",
        help="Aggregate JSON filename written inside --output-dir.",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Also write the reconstructed prompt for each iteration.",
    )
    parser.add_argument(
        "--api-log",
        action="store_true",
        help="Enable the VLM wrapper's prompt/response logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build outputs and prompts without calling the VLM API.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")


def numeric_name_key(path: Path) -> tuple[int, int | str]:
    if path.name.isdigit():
        return (0, int(path.name))
    return (1, path.name)


def find_iteration_dirs(log_dir: Path) -> list[Path]:
    iteration_dirs = [
        path
        for path in log_dir.iterdir()
        if path.is_dir()
        and (path / "scene_graph_input.json").is_file()
        and (path / "images.json").is_file()
    ]
    return sorted(iteration_dirs, key=numeric_name_key)


def resolve_choices(args: argparse.Namespace) -> list[str]:
    choices: list[str] = []
    if args.choices:
        choices.extend(args.choices)
    if args.choice_items:
        choices.extend(args.choice_items)
    return choices


def load_floorplan_graph(iteration_dir: Path) -> dict[str, Any]:
    floorplan_path = iteration_dir / "floorplan_input.json"
    if not floorplan_path.is_file():
        return {}

    floorplan_input = load_json(floorplan_path)
    graph = floorplan_input.get("floorplan_graph", {})
    if graph:
        return graph

    json_path = floorplan_input.get("floorplan_json_path", "")
    if not json_path:
        return {}

    candidate = Path(json_path).expanduser()
    if not candidate.is_file():
        return {}

    loaded = load_json(candidate)
    return loaded.get("floorplan_graph", loaded)


def should_use_floorplan(iteration_dir: Path, override: bool | None) -> bool:
    if override is not None:
        return override

    floorplan_path = iteration_dir / "floorplan_input.json"
    if not floorplan_path.is_file():
        return False

    floorplan_input = load_json(floorplan_path)
    return bool(floorplan_input.get("use_floorplan_prior", False))


def load_images(iteration_dir: Path) -> tuple[list[str], list[str]]:
    image_entries = load_json(iteration_dir / "images.json")
    image_entries = sorted(image_entries, key=lambda entry: entry.get("index", 0))

    image_paths = []
    image_ordering = []
    for entry in image_entries:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = iteration_dir / path
        image_paths.append(str(path))
        image_ordering.append(entry.get("ordering", path.name))

    return image_paths, image_ordering


def output_summary(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": output.get("answer", ""),
        "answered": output.get("answered", False),
        "confidence": output.get("confidence", 0.0),
        "mode": output.get("mode", ""),
        "reasoning": output.get("reasoning", ""),
        "descriptions": {
            "scene_graph": output.get("sg_description", ""),
            "floorplan": output.get("floorplan_description", ""),
            "images": output.get("image_descriptions", []),
        },
    }


def logged_raw_output(iteration_dir: Path) -> dict[str, Any]:
    output_path = iteration_dir / "output.json"
    if not output_path.is_file():
        return {}
    return load_json(output_path).get("raw_vlm_output", {})


def logged_current_agent_state(iteration_dir: Path) -> str | None:
    output_path = iteration_dir / "output.json"
    if not output_path.is_file():
        return None
    return load_json(output_path).get("converted_output", {}).get("current_agent_state")


def make_history_entry(output: dict[str, Any], current_agent_state: str | None) -> dict:
    return {
        "answered": output.get("answered", False),
        "answer": output.get("answer", ""),
        "mode": output.get("mode", ""),
        "current_agent_state": current_agent_state,
        "target_room_id": output.get("target_room_id"),
        "target_room_label": output.get("target_room_label"),
        "target_object_ids": output.get("target_object_ids"),
        "confidence": output.get("confidence", 0.0),
    }


def add_history_to_prompt(history: list[dict[str, Any]], history_length: int) -> str:
    history_str = "HISTORY OF PREVIOUS OUTPUTS:\n"
    for index, output in enumerate(history[-history_length:]):
        history_str += f"Output {index + 1}. "
        history_str += (
            f"Answered: {output.get('answered', False)}, "
            f"Answer: {output.get('answer', '')}, "
        )
        history_str += f"Mode: {output.get('mode', '')}, "
        if output.get("current_agent_state") is not None:
            history_str += f"Agent State: {output['current_agent_state']}, "
        if output.get("target_room_id") is not None:
            history_str += f"Target Room ID: {output['target_room_id']}, "
        if output.get("target_room_label") is not None:
            history_str += f"Target Room Label: {output['target_room_label']}, "
        if output.get("target_object_ids") is not None:
            history_str += f"Target Object IDs: {output['target_object_ids']}, "
        history_str += f"Confidence: {output.get('confidence', 0.0)}\n"
    return history_str


def build_prompt(
    iteration_dir: Path,
    question: str,
    choices: list[str],
    use_floorplan: bool,
    history: list[dict[str, Any]],
    history_length: int,
) -> str:
    prompt = ""

    if use_floorplan:
        floorplan_graph = load_floorplan_graph(iteration_dir)
        if floorplan_graph.get("nodes"):
            prompt += f"FLOORPLAN PRIOR: \n {json.dumps(floorplan_graph)} \n\n"

    scene_graph = load_json(iteration_dir / "scene_graph_input.json")
    prompt += f"SCENE GRAPH: \n {json.dumps(scene_graph)} \n\n"

    if history:
        prompt += add_history_to_prompt(history, history_length) + "\n\n"

    current_agent_state = logged_current_agent_state(iteration_dir)
    if current_agent_state:
        prompt += current_agent_state + "\n\n"

    if choices:
        prompt += "ANSWER CHOICES:\n"
        for index, choice in enumerate(choices):
            prompt += f"{CHOICE_LABELS.get(index, str(index))}: {choice}\n"
        prompt += "\n"

    prompt += f"QUESTION: {question}"
    return prompt


def make_vlm(system_prompt: Path, model: str, api_log: bool) -> Any:
    from vlms_python.eqa_models import OpenAIEQA, OpenAIEQAConfig

    config = OpenAIEQAConfig()
    config.update(
        {
            "client_config": {"model": model},
            "system_prompt_path": str(system_prompt),
            "log": api_log,
        }
    )
    return OpenAIEQA(config)


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.expanduser().absolute()
    system_prompt = args.system_prompt.expanduser().absolute()
    output_dir = (
        args.output_dir.expanduser().absolute()
        if args.output_dir is not None
        else log_dir / "vlm_replay_outputs"
    )

    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")
    if not system_prompt.is_file():
        raise FileNotFoundError(f"System prompt file does not exist: {system_prompt}")

    choices = resolve_choices(args)
    iteration_dirs = find_iteration_dirs(log_dir)
    if not iteration_dirs:
        raise RuntimeError(f"No replayable iteration directories found in {log_dir}")

    vlm = None if args.dry_run else make_vlm(system_prompt, args.model, args.api_log)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "prompts"
    if args.save_prompts:
        prompts_dir.mkdir(parents=True, exist_ok=True)

    logged_history: list[dict[str, Any]] = []
    replayed_history: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {
        "log_dir": str(log_dir),
        "system_prompt": str(system_prompt),
        "question": args.question,
        "choices": choices,
        "model": args.model,
        "history_source": args.history_source,
        "iterations": [],
    }

    for iteration_dir in iteration_dirs:
        if args.history_source == "logged":
            history = logged_history
        elif args.history_source == "replayed":
            history = replayed_history
        else:
            history = []

        use_floorplan = should_use_floorplan(iteration_dir, args.use_floorplan)
        prompt = build_prompt(
            iteration_dir=iteration_dir,
            question=args.question,
            choices=choices,
            use_floorplan=use_floorplan,
            history=history,
            history_length=args.history_length,
        )
        images, image_ordering = load_images(iteration_dir)

        if args.save_prompts:
            prompt_path = prompts_dir / f"{iteration_dir.name}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

        if args.dry_run:
            replayed_output: dict[str, Any] = {}
            success = False
        else:
            replayed_output, success = vlm.answer(prompt, images, image_ordering)

        logged_output = logged_raw_output(iteration_dir)
        record = {
            "iteration": iteration_dir.name,
            "success": success,
            "use_floorplan": use_floorplan,
            "images": [
                {"path": path, "ordering": ordering}
                for path, ordering in zip(images, image_ordering, strict=True)
            ],
            **output_summary(replayed_output),
            "raw_vlm_output": replayed_output,
            "logged": output_summary(logged_output) if logged_output else None,
        }
        if args.save_prompts:
            record["prompt_path"] = str(prompts_dir / f"{iteration_dir.name}.txt")

        dump_json(output_dir / f"{iteration_dir.name}.json", record)
        aggregate["iterations"].append(record)

        current_agent_state = logged_current_agent_state(iteration_dir)
        logged_history.append(make_history_entry(logged_output, current_agent_state))
        replayed_history.append(
            make_history_entry(replayed_output, current_agent_state)
        )

        status = "dry-run" if args.dry_run else ("ok" if success else "failed")
        print(
            f"[{iteration_dir.name}] {status}: "
            f"answer={record['answer']!r}, mode={record['mode']!r}, "
            f"confidence={record['confidence']}"
        )

    dump_json(output_dir / args.aggregate_name, aggregate)
    print(f"Wrote replay outputs to: {output_dir}")


if __name__ == "__main__":
    main()
