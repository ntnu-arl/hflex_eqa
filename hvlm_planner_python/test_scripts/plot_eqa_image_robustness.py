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
"""Replay logged EQA VLM calls with image perturbations and plot robustness."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import replay_eqa_vlm_answers as replay
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

PERTURBATION_LABELS = {
    "blur": "Blur",
    "illumination": "Illumination",
    "gamma": "Gamma correction",
}


def configure_matplotlib_pdf_fonts(matplotlib: Any) -> None:
    """Embed TrueType fonts in vector exports to avoid Papercept Type 3 errors."""
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


# Edit this block for the multi-episode robustness plot. When --episodes-parent is
# provided, each episode is resolved as:
#   <episodes-parent>/<episode>/eqa_log
# unless log_dir is provided in the entry.
EPISODE_CONFIGS: list[dict[str, Any]] = [
    {
        "episode": "chairs_autonomous",
        "question": "How many chairs are there to have lunch?",
        "choices": ["Three", "Five", "Seven", "Two"],
    },
    {
        "episode": "meeting_room_whiteboard_autonomous_good",
        "question": "What is written on the whiteboard?",
        "choices": ["Robotics", "Deep learning", "Hello World", "University"],
    },
    {
        "episode": "meeting_screen_good",
        "question": "Is there any visual support in the meeting room?",
        "choices": [
            "Yes",
            "No",
        ],
    },
    {
        "episode": "micro_autonomous2",
        "question": "What is the microwave's color?",
        "choices": ["White", "Black", "Blue", "Yellow"],
    },
    {
        "episode": "office_jacket",
        "question": "Where can I put my jacket?",
        "choices": [
            "On the hanger in the office",
            "In the cabinet",
            "On the chair",
        ],
    },
    {
        "episode": "office_laptop_autonomous",
        "question": "Where can I find my laptop?",
        "choices": [
            "On the desk",
            "In the cabinet",
            "On the chair",
        ],
    },
    {
        "episode": "sequential",
        "question": "Are the fridge and the chair the same color?",
        "choices": [
            "Yes",
            "No",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an offline EQA robustness plot by replaying logged VLM "
            "planner inputs with perturbed images. The scene graph remains fixed."
        )
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Main eqa_log directory containing iteration subdirectories.",
    )
    parser.add_argument(
        "--episodes-parent",
        type=Path,
        default=None,
        help=(
            "Parent directory containing one subdirectory per EQA episode. Uses "
            "EPISODE_CONFIGS near the top of this script for questions/choices."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        required=True,
        help="System prompt file to use for replayed VLM calls.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Question to append to each replayed prompt.",
    )
    parser.add_argument(
        "--choices",
        nargs="*",
        default=None,
        help="Multiple-choice answers. Used by default unless --no-choices is set.",
    )
    parser.add_argument(
        "--choice",
        action="append",
        dest="choice_items",
        default=None,
        help="Multiple-choice answer. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-choices",
        action="store_true",
        help="Do not include answer choices in the replayed prompt.",
    )
    parser.add_argument(
        "--blur-sigmas",
        type=float,
        nargs="+",
        default=[0.0, 1.5, 3.0, 5.0],
        help="Gaussian blur radii. Default: 0.0 1.5 3.0 5.0.",
    )
    parser.add_argument(
        "--illumination-factors",
        type=float,
        nargs="+",
        default=[1.0, 0.75, 0.5, 0.35],
        help="Brightness scale factors. Default: 1.0 0.75 0.5 0.35.",
    )
    parser.add_argument(
        "--gamma-values",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5],
        help=(
            "Gamma correction values using output = input ** (1 / gamma). "
            "Default: 1.0 1.5 2.0 2.5."
        ),
    )
    parser.add_argument(
        "--history-source",
        choices=("logged", "replayed", "none"),
        default="logged",
        help="History source for prompt reconstruction. Default: logged.",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=1000,
        help="Number of previous outputs to include in the prompt history.",
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
            "Directory for cached replay JSON and plots. Defaults to "
            "<log-dir>/image_robustness_outputs."
        ),
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Plot output path. Defaults to <output-dir>/eqa_image_robustness.png.",
    )
    parser.add_argument(
        "--combined-plot-path",
        type=Path,
        default=None,
        help=(
            "Combined plot output path. Defaults to "
            "<output-dir>/eqa_image_robustness_combined.png."
        ),
    )
    parser.add_argument(
        "--aggregate-name",
        default="robustness_results.json",
        help="Aggregate JSON filename written inside --output-dir.",
    )
    parser.add_argument(
        "--csv-name",
        default="robustness_metrics.csv",
        help="Mean/std metrics CSV filename written inside --output-dir.",
    )
    parser.add_argument(
        "--episode-csv-name",
        default="robustness_episode_metrics.csv",
        help="Per-episode metrics CSV filename written inside --output-dir.",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Write reconstructed prompts for each perturbation and iteration.",
    )
    parser.add_argument(
        "--api-log",
        action="store_true",
        help="Enable the VLM wrapper's prompt/response logging.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Ignore cached per-iteration VLM outputs and call the API again.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts, perturb images, and generate placeholder metrics only.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation and only write JSON/CSV outputs.",
    )
    return parser.parse_args()


def safe_value(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def pdf_sibling_path(path: Path) -> Path:
    return path.with_suffix(".pdf")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "episode"


def resolve_choices_from_values(
    choices: list[str] | None, choice_items: list[str] | None
) -> list[str]:
    result: list[str] = []
    if choices:
        result.extend(choices)
    if choice_items:
        result.extend(choice_items)
    return result


def find_episode_log_dir(episodes_parent: Path, entry: dict[str, Any]) -> Path | None:
    configured_log_dir = entry.get("log_dir")
    if configured_log_dir:
        candidate = Path(configured_log_dir).expanduser()
        if not candidate.is_absolute():
            candidate = episodes_parent / candidate
        if candidate.is_dir():
            return candidate.absolute()

    episode_name = entry["episode"]
    candidates = [
        episodes_parent / episode_name / "eqa_log",
        episodes_parent / episode_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.absolute()
    return None


def discover_episode_log_dirs(episodes_parent: Path) -> list[Path]:
    candidates = []
    if (episodes_parent / "scene_graph_input.json").is_file():
        candidates.append(episodes_parent)
    candidates.extend(
        path for path in episodes_parent.rglob("eqa_log") if path.is_dir()
    )
    return sorted(
        {path.absolute() for path in candidates if replay.find_iteration_dirs(path)},
        key=lambda path: str(path),
    )


def resolve_episode_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.log_dir is None and args.episodes_parent is None:
        raise ValueError("Set either --log-dir for one episode or --episodes-parent.")
    if args.log_dir is not None and args.episodes_parent is not None:
        raise ValueError("Use either --log-dir or --episodes-parent, not both.")

    if args.log_dir is not None:
        if args.question is None:
            raise ValueError("--question is required when using --log-dir.")
        return [
            {
                "episode": safe_name(args.log_dir.expanduser().absolute().parent.name),
                "log_dir": args.log_dir.expanduser().absolute(),
                "question": args.question,
                "choices": []
                if args.no_choices
                else resolve_choices_from_values(args.choices, args.choice_items),
            }
        ]

    episodes_parent = args.episodes_parent.expanduser().absolute()
    if not episodes_parent.is_dir():
        raise FileNotFoundError(f"Episodes parent does not exist: {episodes_parent}")

    specs = []
    for entry in EPISODE_CONFIGS:
        log_dir = find_episode_log_dir(episodes_parent, entry)
        if log_dir is None:
            print(f"Skipping missing episode log: {entry['episode']}")
            continue
        specs.append(
            {
                "episode": entry["episode"],
                "log_dir": log_dir,
                "question": entry.get("question", args.question),
                "choices": []
                if args.no_choices
                else entry.get(
                    "choices",
                    resolve_choices_from_values(args.choices, args.choice_items),
                ),
            }
        )

    if specs:
        missing_question = [spec["episode"] for spec in specs if not spec["question"]]
        if missing_question:
            raise ValueError(
                "Missing question for episodes: " + ", ".join(missing_question)
            )
        return specs

    discovered = discover_episode_log_dirs(episodes_parent)
    if not discovered:
        raise RuntimeError(
            f"No replayable eqa_log directories found in {episodes_parent}"
        )
    if args.question is None:
        raise ValueError(
            "--question is required when EPISODE_CONFIGS is empty or no configured "
            "episodes were found."
        )

    return [
        {
            "episode": safe_name(path.parent.name),
            "log_dir": path,
            "question": args.question,
            "choices": []
            if args.no_choices
            else resolve_choices_from_values(args.choices, args.choice_items),
        }
        for path in discovered
    ]


def output_path_for(
    output_dir: Path, perturbation: str, level_index: int, value: float, iteration: str
) -> Path:
    level_name = f"{level_index:02d}_{safe_value(value)}"
    return output_dir / perturbation / level_name / f"{iteration}.json"


def prompt_path_for(
    output_dir: Path, perturbation: str, level_index: int, value: float, iteration: str
) -> Path:
    level_name = f"{level_index:02d}_{safe_value(value)}"
    return output_dir / "prompts" / perturbation / level_name / f"{iteration}.txt"


def load_rgb_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def perturb_image(image: Image.Image, perturbation: str, value: float) -> np.ndarray:
    if perturbation == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=value))
    elif perturbation == "illumination":
        image = ImageEnhance.Brightness(image).enhance(value)
    elif perturbation == "gamma":
        if value <= 0.0:
            raise ValueError("Gamma values must be positive.")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.power(arr, 1.0 / value)
        return np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown perturbation: {perturbation}")

    return np.asarray(image, dtype=np.uint8)


def perturb_images(image_paths: list[str], perturbation: str, value: float):
    return [
        perturb_image(load_rgb_image(path), perturbation, value) for path in image_paths
    ]


def normalize_text(text: Any) -> str:
    text_str = "" if text is None else str(text)
    text_str = text_str.lower()
    text_str = re.sub(r"[^a-z0-9]+", " ", text_str)
    return re.sub(r"\s+", " ", text_str).strip()


def normalize_answer(answer: Any, choices: list[str]) -> str:
    if answer is None:
        return ""

    raw_answer = str(answer).strip()
    prefix_match = re.match(
        r"^(?:answer\s*)?([a-e])(?:[\).:\s-]+|$)", raw_answer, flags=re.IGNORECASE
    )
    if prefix_match:
        choice_index = ord(prefix_match.group(1).lower()) - ord("a")
        if choice_index < len(choices):
            return normalize_text(choices[choice_index])
        raw_answer = raw_answer[prefix_match.end() :]

    answer_str = raw_answer.lower()
    answer_str = re.sub(r"^\s*(?:answer\s*)?[a-e][\).:\s-]+", "", answer_str)
    answer_str = answer_str.lower()
    return normalize_text(answer_str)


def answer_matches(
    candidate: dict[str, Any], reference: dict[str, Any], choices: list[str]
) -> bool:
    if not candidate.get("answered", False):
        return False
    if not reference.get("answered", False):
        return False
    return normalize_answer(candidate.get("answer", ""), choices) == normalize_answer(
        reference.get("answer", ""), choices
    )


def select_final_answer(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [output for output in outputs if output.get("answered", False)]
    if answered:
        return answered[-1]
    return outputs[-1] if outputs else {}


def output_summary_without_confidence(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": output.get("answer", ""),
        "answered": output.get("answered", False),
        "mode": output.get("mode", ""),
        "reasoning": output.get("reasoning", ""),
        "descriptions": {
            "scene_graph": output.get("sg_description", ""),
            "floorplan": output.get("floorplan_description", ""),
            "images": output.get("image_descriptions", []),
        },
    }


def make_history_entry(output: dict[str, Any], iteration_dir: Path) -> dict[str, Any]:
    return replay.make_history_entry(
        output, replay.logged_current_agent_state(iteration_dir)
    )


def write_episode_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "log_dir",
        "perturbation",
        "level_index",
        "value",
        "mode_consistency",
        "answer_success",
        "num_iterations",
        "num_mode_matches",
        "reference_answer",
        "perturbed_answer",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_summary_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "perturbation",
        "level_index",
        "value",
        "num_episodes",
        "mode_consistency_mean",
        "mode_consistency_std",
        "answer_success_mean",
        "answer_success_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key) for key in fieldnames})


def summarize_episode_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    for row in metrics:
        key = (row["perturbation"], row["level_index"], row["value"])
        grouped.setdefault(key, []).append(row)

    perturbation_order = {
        name: index for index, name in enumerate(PERTURBATION_LABELS.keys())
    }
    summary = []
    for (perturbation, level_index, value), rows in sorted(
        grouped.items(),
        key=lambda item: (perturbation_order.get(item[0][0], 99), item[0][1]),
    ):
        mode_values = np.asarray(
            [row["mode_consistency"] for row in rows], dtype=np.float64
        )
        answer_values = np.asarray(
            [row["answer_success"] for row in rows], dtype=np.float64
        )
        summary.append(
            {
                "perturbation": perturbation,
                "level_index": level_index,
                "value": value,
                "num_episodes": len(rows),
                "mode_consistency_mean": float(np.mean(mode_values)),
                "mode_consistency_std": float(np.std(mode_values)),
                "answer_success_mean": float(np.mean(answer_values)),
                "answer_success_std": float(np.std(answer_values)),
            }
        )
    return summary


def format_plot_value(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def severity_table_rows(metrics: list[dict[str, Any]]) -> list[list[str]]:
    values_by_level: dict[int, dict[str, Any]] = {}
    for row in metrics:
        level_index = int(row["level_index"])
        values_by_level.setdefault(level_index, {})[row["perturbation"]] = row.get(
            "value"
        )

    rows = []
    for level_index in sorted(values_by_level):
        values = values_by_level[level_index]
        rows.append(
            [
                str(level_index),
                format_plot_value(values.get("blur")),
                format_plot_value(values.get("illumination")),
                format_plot_value(values.get("gamma")),
            ]
        )
    return rows


def severity_table_text(metrics: list[dict[str, Any]]) -> str:
    rows = severity_table_rows(metrics)
    if not rows:
        return "Severity levels: no perturbation values available"

    table_rows = [["Level", "Blur sigma", "Illum. factor", "Gamma"], *rows]
    widths = [
        max(len(row[column]) for row in table_rows)
        for column in range(len(table_rows[0]))
    ]
    lines = ["Severity level parameters:"]
    for row in table_rows:
        lines.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
    return "\n".join(lines)


def add_severity_box(ax: Any, metrics: list[dict[str, Any]]) -> None:
    rows = severity_table_rows(metrics)
    if not rows:
        return

    table = ax.table(
        cellText=rows,
        colLabels=[r"$\ell$", r"$\sigma_b$", r"$\alpha_I$", r"$\gamma$"],
        cellLoc="center",
        colLoc="center",
        bbox=(0.58, 0.05, 0.39, 0.30),
        zorder=10,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for cell in table.get_celld().values():
        cell.set_edgecolor("0.55")
        cell.set_facecolor((1.0, 1.0, 1.0, 0.86))
        cell.set_linewidth(0.6)
    ax.text(
        0.775,
        0.36,
        r"Severity: $\sigma_b,\ \alpha_I,\ \gamma$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.55",
            "alpha": 0.86,
        },
    )


def severity_levels(metrics: list[dict[str, Any]]) -> list[int]:
    return sorted({int(row["level_index"]) for row in metrics})


def draw_severity_table(
    draw: ImageDraw.ImageDraw,
    metrics: list[dict[str, Any]],
    origin: tuple[int, int],
    fill: tuple[int, int, int] = (35, 35, 35),
    line_height: int = 22,
) -> None:
    x, y = origin
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    headers = ["level", "sigma_b", "alpha_I", "gamma"]
    rows = severity_table_rows(metrics)
    col_widths = [54, 78, 88, 64]
    draw.text((x, y), "Severity: sigma_b, alpha_I, gamma", fill=fill, font=font)
    y += line_height
    col_x = x
    for header, width in zip(headers, col_widths, strict=True):
        draw.text((col_x, y), header, fill=fill, font=font)
        col_x += width
    for row_index, row in enumerate(rows, start=1):
        col_x = x
        row_y = y + row_index * line_height
        for value, width in zip(row, col_widths, strict=True):
            draw.text((col_x, row_y), value, fill=fill, font=font)
            col_x += width


def plot_metrics(metrics: list[dict[str, Any]], plot_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        configure_matplotlib_pdf_fonts(matplotlib)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plot_metrics_with_pil(metrics, plot_path)
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharex=True, sharey=True)
    colors = {
        "blur": "#3567b7",
        "illumination": "#c65f2e",
        "gamma": "#2b8a5f",
    }
    markers = {"blur": "o", "illumination": "s", "gamma": "^"}
    x_ticks = severity_levels(metrics)

    for ax, metric_name, title in (
        (axes[0], "mode_consistency", "Mode consistency"),
        (axes[1], "answer_success", "Answer success"),
    ):
        mean_key = f"{metric_name}_mean"
        std_key = f"{metric_name}_std"
        for perturbation in PERTURBATION_LABELS:
            rows = [row for row in metrics if row["perturbation"] == perturbation]
            rows = sorted(rows, key=lambda row: row["level_index"])
            if not rows:
                continue
            x_values = [row["level_index"] for row in rows]
            y_values = [row.get(mean_key, row.get(metric_name, 0.0)) for row in rows]
            y_std = [row.get(std_key, 0.0) for row in rows]
            ax.plot(
                x_values,
                y_values,
                marker=markers[perturbation],
                linewidth=2.0,
                color=colors[perturbation],
                label=PERTURBATION_LABELS[perturbation],
            )
            ax.fill_between(
                x_values,
                np.clip(np.asarray(y_values) - np.asarray(y_std), 0.0, 1.0),
                np.clip(np.asarray(y_values) + np.asarray(y_std), 0.0, 1.0),
                color=colors[perturbation],
                alpha=0.16,
                linewidth=0,
            )

        ax.set_title(title)
        ax.set_xlabel(r"Perturbation severity level ($\ell$)")
        ax.set_xticks(x_ticks)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mode consistency / answer success")
    axes[1].legend(loc="lower left", frameon=False)
    add_severity_box(axes[1], metrics)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_sibling_path(plot_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_with_pil(metrics: list[dict[str, Any]], plot_path: Path) -> None:
    width, height = 1600, 680
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    colors = {
        "blur": (53, 103, 183),
        "illumination": (198, 95, 46),
        "gamma": (43, 138, 95),
    }
    panels = [
        (80, 70, 760, 560, "mode_consistency", "Mode consistency"),
        (860, 70, 1540, 560, "answer_success", "Answer success"),
    ]
    max_level = max((row["level_index"] for row in metrics), default=0)

    for left, top, right, bottom, metric_name, title in panels:
        mean_key = f"{metric_name}_mean"
        std_key = f"{metric_name}_std"
        draw.rectangle((left, top, right, bottom), outline=(40, 40, 40), width=2)
        draw.text((left, top - 35), title, fill=(20, 20, 20))
        draw.text(
            (left + 220, bottom + 45),
            "Perturbation severity level (l)",
            fill=(20, 20, 20),
        )
        if metric_name == "mode_consistency":
            draw.text((left - 65, top + 180), "Score", fill=(20, 20, 20))

        for y_value in np.linspace(0.0, 1.0, 6):
            y = bottom - int(y_value * (bottom - top))
            draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
            draw.text((left - 35, y - 8), f"{y_value:.1f}", fill=(80, 80, 80))

        for perturbation in PERTURBATION_LABELS:
            rows = [row for row in metrics if row["perturbation"] == perturbation]
            rows = sorted(rows, key=lambda row: row["level_index"])
            if not rows:
                continue

            points = []
            error_bars = []
            for row in rows:
                x_ratio = row["level_index"] / max(max_level, 1)
                x = left + int(x_ratio * (right - left))
                mean_value = float(row.get(mean_key, row.get(metric_name, 0.0)))
                std_value = float(row.get(std_key, 0.0))
                y = bottom - int(mean_value * (bottom - top))
                y_low = bottom - int(max(mean_value - std_value, 0.0) * (bottom - top))
                y_high = bottom - int(min(mean_value + std_value, 1.0) * (bottom - top))
                points.append((x, y))
                error_bars.append((x, y_low, y_high))

            if len(points) > 1:
                draw.line(points, fill=colors[perturbation], width=4)
            for x, y_low, y_high in error_bars:
                draw.line((x, y_low, x, y_high), fill=colors[perturbation], width=2)
                draw.line(
                    (x - 6, y_low, x + 6, y_low),
                    fill=colors[perturbation],
                    width=2,
                )
                draw.line(
                    (x - 6, y_high, x + 6, y_high),
                    fill=colors[perturbation],
                    width=2,
                )
            for x, y in points:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=colors[perturbation])

        for level_index in severity_levels(metrics):
            x_ratio = level_index / max(max_level, 1)
            x = left + int(x_ratio * (right - left))
            draw.text((x - 5, bottom + 12), str(level_index), fill=(80, 80, 80))

    legend_x, legend_y = 900, 405
    for index, perturbation in enumerate(PERTURBATION_LABELS):
        y = legend_y + 24 * index
        draw.line(
            (legend_x, y + 8, legend_x + 35, y + 8),
            fill=colors[perturbation],
            width=4,
        )
        draw.ellipse(
            (legend_x + 13, y + 2, legend_x + 25, y + 14),
            fill=colors[perturbation],
        )
        draw.text(
            (legend_x + 48, y), PERTURBATION_LABELS[perturbation], fill=(20, 20, 20)
        )

    table_x, table_y = 1190, 405
    draw.rectangle(
        (table_x - 10, table_y - 8, 1530, 548),
        fill=(255, 255, 255),
        outline=(150, 150, 150),
    )
    draw_severity_table(draw, metrics, (table_x, table_y), line_height=18)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(plot_path)
    image.save(pdf_sibling_path(plot_path), "PDF", resolution=100.0)


def plot_combined_metrics(metrics: list[dict[str, Any]], plot_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        configure_matplotlib_pdf_fonts(matplotlib)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plot_combined_metrics_with_pil(metrics, plot_path)
        return

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    curve_colors = {
        ("mode_consistency", "blur"): "#1f77b4",
        ("mode_consistency", "illumination"): "#2ca02c",
        ("mode_consistency", "gamma"): "#9467bd",
        ("answer_success", "blur"): "#ff7f0e",
        ("answer_success", "illumination"): "#d62728",
        ("answer_success", "gamma"): "#8c564b",
    }
    perturbation_styles = {
        "blur": {"marker": "o"},
        "illumination": {"marker": "s"},
        "gamma": {"marker": "^"},
    }
    metric_specs = [
        ("mode_consistency", "Mode consistency", "-"),
        ("answer_success", "Answer success", "--"),
    ]

    for metric_name, metric_label, line_style in metric_specs:
        for perturbation in PERTURBATION_LABELS:
            rows = [row for row in metrics if row["perturbation"] == perturbation]
            rows = sorted(rows, key=lambda row: row["level_index"])
            if not rows:
                continue
            style = perturbation_styles[perturbation]
            color = curve_colors[(metric_name, perturbation)]
            x_values = [row["level_index"] for row in rows]
            mean_key = f"{metric_name}_mean"
            std_key = f"{metric_name}_std"
            y_values = [row.get(mean_key, row.get(metric_name, 0.0)) for row in rows]
            y_std = [row.get(std_key, 0.0) for row in rows]
            ax.plot(
                x_values,
                y_values,
                linestyle=line_style,
                marker=style["marker"],
                linewidth=2.0,
                color=color,
                label=f"{PERTURBATION_LABELS[perturbation]} - {metric_label}",
            )
            ax.fill_between(
                x_values,
                np.clip(np.asarray(y_values) - np.asarray(y_std), 0.0, 1.0),
                np.clip(np.asarray(y_values) + np.asarray(y_std), 0.0, 1.0),
                color=color,
                alpha=0.13,
                linewidth=0,
            )

    ax.set_xlabel(r"Perturbation severity level ($\ell$)")
    ax.set_ylabel("Mode consistency / answer success")
    ax.set_xticks(severity_levels(metrics))
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.3)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=curve_colors[(metric_name, perturbation)],
            linestyle=(0, (5, 2)) if metric_name == "answer_success" else "-",
            marker=perturbation_styles[perturbation]["marker"],
            markersize=5.5,
            linewidth=2.8,
            label=f"{PERTURBATION_LABELS[perturbation]} - {metric_label}",
        )
        for metric_name, metric_label, line_style in metric_specs
        for perturbation in PERTURBATION_LABELS
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=False,
        fontsize=7,
        ncol=1,
        handlelength=6.5,
        handletextpad=0.8,
        numpoints=1,
    )
    add_severity_box(ax, metrics)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_sibling_path(plot_path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    fill: tuple[int, int, int],
    width: int,
    dash_length: int = 16,
    gap_length: int = 10,
) -> None:
    for start, end in zip(points[:-1], points[1:], strict=True):
        x0, y0 = start
        x1, y1 = end
        segment_length = float(np.hypot(x1 - x0, y1 - y0))
        if segment_length <= 0.0:
            continue
        dx = (x1 - x0) / segment_length
        dy = (y1 - y0) / segment_length
        distance = 0.0
        while distance < segment_length:
            dash_end = min(distance + dash_length, segment_length)
            draw.line(
                (
                    int(x0 + dx * distance),
                    int(y0 + dy * distance),
                    int(x0 + dx * dash_end),
                    int(y0 + dy * dash_end),
                ),
                fill=fill,
                width=width,
            )
            distance += dash_length + gap_length


def plot_combined_metrics_with_pil(
    metrics: list[dict[str, Any]], plot_path: Path
) -> None:
    width, height = 1100, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    left, top, right, bottom = 90, 70, 830, 570
    curve_colors = {
        ("mode_consistency", "blur"): (31, 119, 180),
        ("mode_consistency", "illumination"): (44, 160, 44),
        ("mode_consistency", "gamma"): (148, 103, 189),
        ("answer_success", "blur"): (255, 127, 14),
        ("answer_success", "illumination"): (214, 39, 40),
        ("answer_success", "gamma"): (140, 86, 75),
    }
    perturbation_markers = {
        "blur": "circle",
        "illumination": "square",
        "gamma": "triangle",
    }
    metric_specs = [
        ("mode_consistency", "Mode consistency", "solid"),
        ("answer_success", "Answer success", "dashed"),
    ]

    draw.rectangle((left, top, right, bottom), outline=(40, 40, 40), width=2)
    draw.text(
        (left, top - 35), "Mode consistency and answer success", fill=(20, 20, 20)
    )
    draw.text(
        (left + 240, bottom + 45),
        "Perturbation severity level (l)",
        fill=(20, 20, 20),
    )
    draw.text((left - 65, top + 185), "Score", fill=(20, 20, 20))

    for y_value in np.linspace(0.0, 1.0, 6):
        y = bottom - int(y_value * (bottom - top))
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((left - 35, y - 8), f"{y_value:.1f}", fill=(80, 80, 80))

    max_level = max((row["level_index"] for row in metrics), default=0)
    for level_index in severity_levels(metrics):
        x_ratio = level_index / max(max_level, 1)
        x = left + int(x_ratio * (right - left))
        draw.text((x - 5, bottom + 12), str(level_index), fill=(80, 80, 80))

    for metric_name, _, line_style in metric_specs:
        for perturbation in PERTURBATION_LABELS:
            rows = [row for row in metrics if row["perturbation"] == perturbation]
            rows = sorted(rows, key=lambda row: row["level_index"])
            if not rows:
                continue
            color = curve_colors[(metric_name, perturbation)]
            mean_key = f"{metric_name}_mean"
            std_key = f"{metric_name}_std"
            points = []
            error_bars = []
            for row in rows:
                x_ratio = row["level_index"] / max(max_level, 1)
                x = left + int(x_ratio * (right - left))
                mean_value = float(row.get(mean_key, row.get(metric_name, 0.0)))
                std_value = float(row.get(std_key, 0.0))
                y = bottom - int(mean_value * (bottom - top))
                y_low = bottom - int(max(mean_value - std_value, 0.0) * (bottom - top))
                y_high = bottom - int(min(mean_value + std_value, 1.0) * (bottom - top))
                points.append((x, y))
                error_bars.append((x, y_low, y_high))
            if len(points) > 1:
                if line_style == "solid":
                    draw.line(points, fill=color, width=4)
                else:
                    draw_dashed_line(draw, points, color, width=4)
            for x, y_low, y_high in error_bars:
                draw.line((x, y_low, x, y_high), fill=color, width=2)
                draw.line((x - 5, y_low, x + 5, y_low), fill=color, width=2)
                draw.line((x - 5, y_high, x + 5, y_high), fill=color, width=2)
            for x, y in points:
                draw.ellipse(
                    (x - 5, y - 5, x + 5, y + 5),
                    fill=color,
                )

    legend_x, legend_y = 110, 400
    legend_items = [
        (metric_name, metric_label, line_style, perturbation)
        for metric_name, metric_label, line_style in metric_specs
        for perturbation in PERTURBATION_LABELS
    ]
    for index, (metric_name, metric_label, line_style, perturbation) in enumerate(
        legend_items
    ):
        y = legend_y + 24 * index
        color = curve_colors[(metric_name, perturbation)]
        sample_end_x = legend_x + 90
        if line_style == "solid":
            draw.line((legend_x, y + 8, sample_end_x, y + 8), fill=color, width=4)
        else:
            draw_dashed_line(
                draw,
                [(legend_x, y + 8), (sample_end_x, y + 8)],
                color,
                width=4,
                dash_length=14,
                gap_length=10,
            )
        marker_x = legend_x + 45
        marker_y = y + 8
        marker = perturbation_markers[perturbation]
        if marker == "circle":
            draw.ellipse(
                (marker_x - 6, marker_y - 6, marker_x + 6, marker_y + 6),
                fill=color,
            )
        elif marker == "square":
            draw.rectangle(
                (marker_x - 6, marker_y - 6, marker_x + 6, marker_y + 6),
                fill=color,
            )
        else:
            draw.polygon(
                [
                    (marker_x, marker_y - 7),
                    (marker_x - 7, marker_y + 7),
                    (marker_x + 7, marker_y + 7),
                ],
                fill=color,
            )
        draw.text(
            (legend_x + 104, y),
            f"{PERTURBATION_LABELS[perturbation]} - {metric_label}",
            fill=(20, 20, 20),
        )

    table_x, table_y = 530, 400
    draw.rectangle(
        (table_x - 10, table_y - 8, 815, 548),
        fill=(255, 255, 255),
        outline=(150, 150, 150),
    )
    draw_severity_table(draw, metrics, (table_x, table_y), line_height=18)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(plot_path)
    image.save(pdf_sibling_path(plot_path), "PDF", resolution=100.0)


def replay_level(
    *,
    perturbation: str,
    level_index: int,
    value: float,
    iteration_dirs: list[Path],
    output_dir: Path,
    system_prompt: Path,
    question: str,
    choices: list[str],
    history_source: str,
    history_length: int,
    model: str,
    api_log: bool,
    rerun: bool,
    dry_run: bool,
    save_prompts: bool,
) -> list[dict[str, Any]]:
    vlm = None if dry_run else replay.make_vlm(system_prompt, model, api_log)
    logged_history: list[dict[str, Any]] = []
    replayed_history: list[dict[str, Any]] = []
    records = []

    for iteration_dir in iteration_dirs:
        cache_path = output_path_for(
            output_dir, perturbation, level_index, value, iteration_dir.name
        )
        if cache_path.is_file() and not rerun:
            record = replay.load_json(cache_path)
            logged_output = replay.logged_raw_output(iteration_dir)
            logged_history.append(make_history_entry(logged_output, iteration_dir))
            replayed_history.append(
                make_history_entry(record.get("raw_vlm_output", {}), iteration_dir)
            )
            records.append(record)
            continue

        if history_source == "logged":
            history = logged_history
        elif history_source == "replayed":
            history = replayed_history
        else:
            history = []

        prompt = replay.build_prompt(
            iteration_dir=iteration_dir,
            question=question,
            choices=choices,
            use_floorplan=True,
            history=history,
            history_length=history_length,
        )
        image_paths, image_ordering = replay.load_images(iteration_dir)

        if save_prompts:
            prompt_path = prompt_path_for(
                output_dir, perturbation, level_index, value, iteration_dir.name
            )
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")

        perturbed_images = perturb_images(image_paths, perturbation, value)
        if dry_run:
            perturbed_output: dict[str, Any] = {}
            success = False
        else:
            perturbed_output, success = vlm.answer(
                prompt, perturbed_images, image_ordering
            )

        logged_output = replay.logged_raw_output(iteration_dir)
        record = {
            "iteration": iteration_dir.name,
            "success": success,
            "perturbation": perturbation,
            "level_index": level_index,
            "value": value,
            "use_floorplan": True,
            "used_choices": bool(choices),
            "images": [
                {"path": path, "ordering": ordering}
                for path, ordering in zip(image_paths, image_ordering, strict=True)
            ],
            **output_summary_without_confidence(perturbed_output),
            "raw_vlm_output": perturbed_output,
            "logged": (
                output_summary_without_confidence(logged_output)
                if logged_output
                else None
            ),
        }
        if save_prompts:
            record["prompt_path"] = str(
                prompt_path_for(
                    output_dir, perturbation, level_index, value, iteration_dir.name
                )
            )

        replay.dump_json(cache_path, record)
        records.append(record)
        logged_history.append(make_history_entry(logged_output, iteration_dir))
        replayed_history.append(make_history_entry(perturbed_output, iteration_dir))

        print(
            f"[{perturbation} {level_index}:{value} {iteration_dir.name}] "
            f"{'dry-run' if dry_run else ('ok' if success else 'failed')}: "
            f"mode={record['mode']!r}, answer={record['answer']!r}"
        )

    return records


def compute_level_metrics(
    episode: str,
    log_dir: Path,
    perturbation: str,
    level_index: int,
    value: float,
    logged_outputs: list[dict[str, Any]],
    replay_records: list[dict[str, Any]],
    choices: list[str],
) -> dict[str, Any]:
    mode_pairs = [
        (record.get("raw_vlm_output", {}), logged_output)
        for record, logged_output in zip(replay_records, logged_outputs, strict=True)
        if logged_output.get("mode") is not None
    ]
    num_mode_matches = sum(
        replayed.get("mode") == logged.get("mode") for replayed, logged in mode_pairs
    )
    mode_consistency = num_mode_matches / max(len(mode_pairs), 1)

    replayed_final = select_final_answer(
        [record.get("raw_vlm_output", {}) for record in replay_records]
    )
    logged_final = select_final_answer(logged_outputs)
    answer_success = (
        1.0 if answer_matches(replayed_final, logged_final, choices) else 0.0
    )

    return {
        "episode": episode,
        "log_dir": str(log_dir),
        "perturbation": perturbation,
        "level_index": level_index,
        "value": value,
        "mode_consistency": mode_consistency,
        "answer_success": answer_success,
        "num_iterations": len(mode_pairs),
        "num_mode_matches": num_mode_matches,
        "reference_answer": logged_final.get("answer", ""),
        "perturbed_answer": replayed_final.get("answer", ""),
    }


def main() -> None:
    args = parse_args()
    system_prompt = args.system_prompt.expanduser().absolute()
    episode_specs = resolve_episode_specs(args)

    default_output_base = (
        args.log_dir.expanduser().absolute()
        if args.log_dir is not None
        else args.episodes_parent.expanduser().absolute()
    )
    output_dir = (
        args.output_dir.expanduser().absolute()
        if args.output_dir is not None
        else default_output_base / "image_robustness_outputs"
    )
    plot_path = (
        args.plot_path.expanduser().absolute()
        if args.plot_path is not None
        else output_dir / "eqa_image_robustness.png"
    )
    combined_plot_path = (
        args.combined_plot_path.expanduser().absolute()
        if args.combined_plot_path is not None
        else output_dir / "eqa_image_robustness_combined.png"
    )

    if not system_prompt.is_file():
        raise FileNotFoundError(f"System prompt file does not exist: {system_prompt}")

    perturbation_values = {
        "blur": args.blur_sigmas,
        "illumination": args.illumination_factors,
        "gamma": args.gamma_values,
    }

    all_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    episode_metrics: list[dict[str, Any]] = []

    for spec in episode_specs:
        episode = safe_name(spec["episode"])
        log_dir = Path(spec["log_dir"]).expanduser().absolute()
        question = spec["question"]
        choices = [] if args.no_choices else list(spec.get("choices", []))

        if not choices and not args.no_choices:
            print(f"[{episode}] No choices configured; prompt will omit choices.")
        if not log_dir.is_dir():
            raise FileNotFoundError(
                f"[{episode}] Log directory does not exist: {log_dir}"
            )

        iteration_dirs = replay.find_iteration_dirs(log_dir)
        if not iteration_dirs:
            raise RuntimeError(
                f"[{episode}] No replayable iterations found in {log_dir}"
            )

        logged_outputs = [replay.logged_raw_output(path) for path in iteration_dirs]
        episode_output_dir = output_dir / "episodes" / episode
        all_records[episode] = {}

        print(f"[{episode}] Replaying {len(iteration_dirs)} iterations from {log_dir}")
        for perturbation, values in perturbation_values.items():
            for level_index, value in enumerate(values):
                records = replay_level(
                    perturbation=perturbation,
                    level_index=level_index,
                    value=value,
                    iteration_dirs=iteration_dirs,
                    output_dir=episode_output_dir,
                    system_prompt=system_prompt,
                    question=question,
                    choices=choices,
                    history_source=args.history_source,
                    history_length=args.history_length,
                    model=args.model,
                    api_log=args.api_log,
                    rerun=args.rerun,
                    dry_run=args.dry_run,
                    save_prompts=args.save_prompts,
                )
                key = f"{perturbation}/{level_index:02d}_{safe_value(value)}"
                all_records[episode][key] = records
                episode_metrics.append(
                    compute_level_metrics(
                        episode,
                        log_dir,
                        perturbation,
                        level_index,
                        value,
                        logged_outputs,
                        records,
                        choices,
                    )
                )

    summary_metrics = summarize_episode_metrics(episode_metrics)

    aggregate = {
        "episodes": [
            {
                "episode": spec["episode"],
                "log_dir": str(spec["log_dir"]),
                "question": spec["question"],
                "choices": [] if args.no_choices else list(spec.get("choices", [])),
            }
            for spec in episode_specs
        ],
        "system_prompt": str(system_prompt),
        "used_choices": not args.no_choices,
        "use_floorplan": True,
        "model": args.model,
        "history_source": args.history_source,
        "perturbation_values": perturbation_values,
        "summary_metrics": summary_metrics,
        "episode_metrics": episode_metrics,
        "records": all_records,
    }
    replay.dump_json(output_dir / args.aggregate_name, aggregate)
    write_summary_metrics_csv(output_dir / args.csv_name, summary_metrics)
    write_episode_metrics_csv(output_dir / args.episode_csv_name, episode_metrics)

    if not args.no_plot:
        plot_metrics(summary_metrics, plot_path)
        plot_combined_metrics(summary_metrics, combined_plot_path)
        print(f"Wrote plot to: {plot_path}")
        print(f"Wrote plot PDF to: {pdf_sibling_path(plot_path)}")
        print(f"Wrote combined plot to: {combined_plot_path}")
        print(f"Wrote combined plot PDF to: {pdf_sibling_path(combined_plot_path)}")
    print(f"Wrote metrics to: {output_dir / args.csv_name}")
    print(f"Wrote per-episode metrics to: {output_dir / args.episode_csv_name}")
    print(f"Wrote aggregate results to: {output_dir / args.aggregate_name}")


if __name__ == "__main__":
    main()
