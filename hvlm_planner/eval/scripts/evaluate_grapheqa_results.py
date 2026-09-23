#!/usr/bin/env python3
"""Evaluate GraphEQA/OpenEQA result bundles.

Expected directory layout:

results_root/
  variation_a/
    openeqa/
      some_run/
        results.json
    exploreeqa/
      some_run/
        results.json
  variation_b/
    exploreeqa/
      another_run/
        output.json

Each JSON file is expected to contain a mapping from episode id to payload:
{
  "<episode_id>": {
    "Success": true,
    "metrics": {
      "overall_steps": 8,
      "traj_length": 3.83,
      "category": "object recognition",
      ...
    }
  }
}

Success rate is computed from the top-level ``Success`` field when present.
If ``Success`` is missing for a non-choice run, the script grades
``metrics.answer_output`` against the ground-truth answer.
For trajectory length, failed episodes with ``traj_length == 0`` are excluded
from the average path-length calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_NAMES: tuple[str, ...] = ("openeqa", "exploreeqa")

JUDGE_SYSTEM_PROMPT = (
    "You are grading embodied QA answers. "
    "Return ONLY strict JSON with keys: "
    "is_correct (boolean), confidence (float in [0,1]), reason (string). "
    "A prediction is correct when it matches the ground truth semantically. "
    "For multiple-choice tasks, if the prediction is close to a choice and that choice "
    "matches the ground truth, mark correct. "
    "Use conservative grading when uncertain."
)


@dataclass(frozen=True)
class EpisodeResult:
    """Single ExploreEQA/OpenEQA episode result."""

    variation: str
    dataset: str
    episode_id: str
    success: bool
    category: str
    question: str
    gt_answer: str
    answer_output: str
    uses_choices: bool
    choices: list[str]
    num_iters: int | None
    path_length: float | None
    source_json: Path

    @property
    def include_path_length(self) -> bool:
        """Whether this episode should contribute to path-length averages."""
        if self.path_length is None:
            return False
        return not (not self.success and self.path_length == 0.0)

    @property
    def has_answer(self) -> bool:
        return bool(self.answer_output.strip())


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for one variation/dataset group."""

    label: str
    variation: str
    dataset: str
    total_episodes: int
    successful_episodes: int
    avg_num_iters_all: float
    avg_num_iters_success: float
    avg_path_length_all: float
    avg_path_length_success: float
    success_rate: float


@dataclass
class JudgeResult:
    """LLM grading output for one episode."""

    is_correct: bool
    confidence: float
    reason: str
    from_cache: bool = False


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate ExploreEQA/OpenEQA result folders."
    )
    parser.add_argument(
        "--results-folder",
        type=Path,
        required=True,
        help="Root directory containing variation subdirectories.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_NAMES),
        choices=list(DATASET_NAMES),
        help="Dataset subdirectories to consider under each variation directory.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the aggregated metrics as JSON.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model for correctness grading when Success is missing.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Path to a JSON cache for LLM grading results.",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Disable LLM grading and use exact string match fallback.",
    )
    parser.add_argument(
        "--by-category",
        action="store_true",
        help="Also print metrics grouped by question category for each run.",
    )
    parser.add_argument(
        "--print-latex",
        action="store_true",
        help="Print LaTeX code for the summary table.",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=None,
        help="Optional path to save LaTeX table code.",
    )
    parser.add_argument(
        "--latex-caption",
        default="GraphEQA evaluation summary.",
        help="Caption used in the generated LaTeX table.",
    )
    parser.add_argument(
        "--latex-label",
        default="tab:grapheqa_eval_summary",
        help="Label used in the generated LaTeX table.",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    """Safe arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _looks_like_results_payload(payload: Any) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    first_value = next(iter(payload.values()))
    if not isinstance(first_value, dict):
        return False
    return "metrics" in first_value


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _judge_key(ep: EpisodeResult, model: str) -> str:
    payload = {
        "model": model,
        "question": ep.question,
        "choices": ep.choices,
        "gt_answer": ep.gt_answer,
        "answer_output": ep.answer_output,
        "uses_choices": ep.uses_choices,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def build_judge_prompt(ep: EpisodeResult) -> str:
    return (
        "Decide if the predicted answer is semantically correct.\n"
        "Return strict JSON only.\n\n"
        f"Question: {ep.question}\n"
        f"Question Category: {ep.category}\n"
        f"Choices: {json.dumps(ep.choices, ensure_ascii=False)}\n"
        f"Ground Truth Answers: {json.dumps([ep.gt_answer], ensure_ascii=False)}\n"
        f"Predicted Answer: {ep.answer_output}\n"
    )


def fallback_exact_match(ep: EpisodeResult) -> JudgeResult:
    pred = _normalize_text(ep.answer_output)
    gt = _normalize_text(ep.gt_answer)
    is_correct = bool(pred) and bool(gt) and pred == gt
    reason = "Exact normalized match" if is_correct else "No exact normalized match"
    return JudgeResult(
        is_correct=is_correct, confidence=1.0 if is_correct else 0.0, reason=reason
    )


def judge_episode(
    ep: EpisodeResult,
    *,
    client: Any | None,
    model: str,
    cache: dict[str, dict[str, Any]],
    disable_llm: bool,
) -> JudgeResult:
    if not ep.has_answer:
        return JudgeResult(
            is_correct=False, confidence=1.0, reason="No answer provided"
        )

    if disable_llm or client is None:
        return fallback_exact_match(ep)

    key = _judge_key(ep, model)
    cached = cache.get(key)
    if cached is not None:
        return JudgeResult(
            is_correct=bool(cached.get("is_correct", False)),
            confidence=float(cached.get("confidence", 0.0)),
            reason=str(cached.get("reason", "")),
            from_cache=True,
        )

    prompt = build_judge_prompt(ep)
    result, success = client.inference(prompt=prompt, images=None)
    if not success:
        return fallback_exact_match(ep)

    judged = JudgeResult(
        is_correct=bool(result.get("is_correct", False)),
        confidence=float(result.get("confidence", 0.0)),
        reason=str(result.get("reason", "")),
    )
    cache[key] = {
        "is_correct": judged.is_correct,
        "confidence": judged.confidence,
        "reason": judged.reason,
    }
    return judged


def build_client(model: str, disable_llm: bool) -> Any | None:
    if disable_llm:
        return None
    config_module = importlib.import_module("vlms_python.openai_client.config")
    client_module = importlib.import_module("vlms_python.openai_client.openai_client")
    OpenAIClientConfig = getattr(config_module, "OpenAIClientConfig")
    OpenAIClient = getattr(client_module, "OpenAIClient")
    cfg = OpenAIClientConfig(model=model, system_prompt=JUDGE_SYSTEM_PROMPT)
    return OpenAIClient(cfg)


def load_results_json(
    json_path: Path,
    *,
    variation: str,
    dataset: str,
    client: Any | None,
    model: str,
    cache: dict[str, dict[str, Any]],
    disable_llm: bool,
) -> list[EpisodeResult]:
    """Load a single results JSON file into episode records."""
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not _looks_like_results_payload(payload):
        return []

    episodes: list[EpisodeResult] = []
    for episode_id, episode_payload in payload.items():
        if not isinstance(episode_payload, dict):
            continue
        metrics = episode_payload.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        uses_choices = bool(metrics.get("uses_choices", False))
        choices = _string_list(metrics.get("choices"))
        gt_answer = str(metrics.get("answer", metrics.get("answer_id", "")))
        answer_output = str(metrics.get("answer_output", ""))
        success_value = episode_payload.get("Success")
        judged_success = False
        if success_value is None:
            if not uses_choices and answer_output.strip():
                judged = judge_episode(
                    EpisodeResult(
                        variation=variation,
                        dataset=dataset,
                        episode_id=str(episode_id),
                        success=False,
                        category=str(metrics.get("category", "unknown")),
                        question=str(metrics.get("question", "")),
                        gt_answer=gt_answer,
                        answer_output=answer_output,
                        uses_choices=uses_choices,
                        choices=choices,
                        num_iters=_int_or_none(metrics.get("overall_steps")),
                        path_length=_float_or_none(metrics.get("traj_length")),
                        source_json=json_path,
                    ),
                    client=client,
                    model=model,
                    cache=cache,
                    disable_llm=disable_llm,
                )
                judged_success = judged.is_correct
        else:
            judged_success = bool(success_value)
        episodes.append(
            EpisodeResult(
                variation=variation,
                dataset=dataset,
                episode_id=str(episode_id),
                success=judged_success,
                category=str(metrics.get("category", "unknown")),
                question=str(metrics.get("question", "")),
                gt_answer=gt_answer,
                answer_output=answer_output,
                uses_choices=uses_choices or success_value is not None,
                choices=choices,
                num_iters=_int_or_none(metrics.get("overall_steps")),
                path_length=_float_or_none(metrics.get("traj_length")),
                source_json=json_path,
            )
        )
    return episodes


def discover_results(
    results_root: Path,
    datasets: list[str],
    *,
    client: Any | None,
    model: str,
    cache: dict[str, dict[str, Any]],
    disable_llm: bool,
) -> dict[str, list[EpisodeResult]]:
    """Discover variation/dataset result groups below the provided root."""
    discovered: dict[str, list[EpisodeResult]] = {}

    for variation_dir in sorted(
        path for path in results_root.iterdir() if path.is_dir()
    ):
        for dataset in datasets:
            dataset_dir = variation_dir / dataset
            if not dataset_dir.is_dir():
                continue

            episodes: list[EpisodeResult] = []
            episodes.extend(
                load_results_json(
                    dataset_dir / "log.json",
                    variation=variation_dir.name,
                    dataset=dataset,
                    client=client,
                    model=model,
                    cache=cache,
                    disable_llm=disable_llm,
                )
            )

            if episodes:
                discovered[f"{variation_dir.name}/{dataset}"] = episodes

    return discovered


def aggregate_metrics(label: str, episodes: list[EpisodeResult]) -> AggregatedMetrics:
    """Aggregate metrics for one variation/dataset group."""
    if not episodes:
        raise ValueError(f"Cannot aggregate empty episode list for {label}")

    successful = [ep for ep in episodes if ep.success]

    all_num_iters = [float(ep.num_iters) for ep in episodes if ep.num_iters is not None]
    success_num_iters = [
        float(ep.num_iters) for ep in successful if ep.num_iters is not None
    ]
    all_path_lengths = [
        ep.path_length
        for ep in episodes
        if ep.include_path_length and ep.path_length is not None
    ]
    success_path_lengths = [
        ep.path_length
        for ep in successful
        if ep.include_path_length and ep.path_length is not None
    ]

    variation = episodes[0].variation
    dataset = episodes[0].dataset
    total_episodes = len(episodes)
    successful_episodes = len(successful)
    return AggregatedMetrics(
        label=label,
        variation=variation,
        dataset=dataset,
        total_episodes=total_episodes,
        successful_episodes=successful_episodes,
        avg_num_iters_all=mean(all_num_iters),
        avg_num_iters_success=mean(success_num_iters),
        avg_path_length_all=mean(all_path_lengths),
        avg_path_length_success=mean(success_path_lengths),
        success_rate=successful_episodes / max(total_episodes, 1),
    )


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def format_float(value: float) -> str:
    return f"{value:.3f}"


def render_table(rows: list[AggregatedMetrics]) -> str:
    """Render a fixed-width text table for terminal output."""
    headers = [
        "run",
        "variation",
        "dataset",
        "episodes",
        "successes",
        "avg_steps_all",
        "avg_steps_success",
        "avg_path_all",
        "avg_path_success",
        "success_rate",
    ]
    body = [
        [
            row.label,
            row.variation,
            row.dataset,
            str(row.total_episodes),
            str(row.successful_episodes),
            format_float(row.avg_num_iters_all),
            format_float(row.avg_num_iters_success),
            format_float(row.avg_path_length_all),
            format_float(row.avg_path_length_success),
            format_percent(row.success_rate),
        ]
        for row in rows
    ]

    widths = [len(header) for header in headers]
    for row in body:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * width for width in widths)
    lines = [fmt(headers), sep]
    lines.extend(fmt(row) for row in body)
    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def render_latex_table(
    rows: list[AggregatedMetrics], *, caption: str, label: str
) -> str:
    """Render a LaTeX summary table."""
    header = [
        "Run",
        "Variation",
        "Dataset",
        "N",
        "N$_{succ}$",
        "Steps$_{all}$",
        "Steps$_{succ}$",
        "Path$_{all}$",
        "Path$_{succ}$",
        "Succ.",
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(row.label),
                    _latex_escape(row.variation),
                    _latex_escape(row.dataset),
                    str(row.total_episodes),
                    str(row.successful_episodes),
                    format_float(row.avg_num_iters_all),
                    format_float(row.avg_num_iters_success),
                    format_float(row.avg_path_length_all),
                    format_float(row.avg_path_length_success),
                    format_percent(row.success_rate),
                ]
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()

    results_root = args.results_folder.expanduser().resolve()
    if not results_root.exists() or not results_root.is_dir():
        raise FileNotFoundError(f"Results folder not found: {results_root}")

    cache_path = (
        args.cache_path.expanduser().resolve()
        if args.cache_path is not None
        else results_root / "llm_eval_cache.json"
    )
    cache = load_cache(cache_path)
    client = build_client(model=args.model, disable_llm=args.disable_llm)

    discovered = discover_results(
        results_root,
        args.datasets,
        client=client,
        model=args.model,
        cache=cache,
        disable_llm=args.disable_llm,
    )

    rows = [
        aggregate_metrics(label, episodes) for label, episodes in discovered.items()
    ]
    rows.sort(key=lambda row: (row.dataset, row.variation, row.label))

    print("\n=== GraphEQA Evaluation Summary ===")
    if rows:
        print(render_table(rows))
    else:
        print("No valid GraphEQA/OpenEQA result JSON files were found.")

    latex_table = render_latex_table(
        rows,
        caption=args.latex_caption,
        label=args.latex_label,
    )

    if args.print_latex:
        print("\n=== LaTeX Table ===")
        print(latex_table)

    if args.latex_output is not None:
        latex_out = args.latex_output.expanduser().resolve()
        latex_out.parent.mkdir(parents=True, exist_ok=True)
        latex_out.write_text(latex_table + "\n", encoding="utf-8")
        print(f"Saved LaTeX table to: {latex_out}")

    per_category_payload: dict[str, list[dict[str, Any]]] = {}
    if args.by_category:
        for label in sorted(discovered):
            grouped: dict[str, list[EpisodeResult]] = {}
            for episode in discovered[label]:
                grouped.setdefault(episode.category or "unknown", []).append(episode)

            category_rows = [
                aggregate_metrics(f"{label}:{category}", category_episodes)
                for category, category_episodes in sorted(grouped.items())
            ]
            print(f"\n=== Per-category metrics: {label} ===")
            print(render_table(category_rows))
            per_category_payload[label] = [row.__dict__ for row in category_rows]

    if args.output_json is not None:
        payload = {
            "results_root": str(results_root),
            "model": args.model,
            "disable_llm": args.disable_llm,
            "runs": [row.__dict__ for row in rows],
            "per_category": per_category_payload,
        }
        out_path = args.output_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved JSON summary to: {out_path}")

    if not args.disable_llm:
        save_cache(cache_path, cache)
        print(f"Saved/updated LLM cache: {cache_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
