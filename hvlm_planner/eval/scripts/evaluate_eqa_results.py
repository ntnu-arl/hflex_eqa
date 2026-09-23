#!/usr/bin/env python3
"""Evaluate active embodied QA results with optional LLM-based answer grading.

This script scans a results root directory that contains subdirectories such as:
- hvlm_exploreeqa_choices_logs
- hvlm_exploreeqa_logs
- hvlm_open_eqa_choices_logs
- hvlm_open_eqa_logs

Each episode directory is expected to contain a ``log.json`` file.
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

DEFAULT_TARGET_DIRS: tuple[str, ...] = (
    "hvlm_exploreeqa_choices_logs",
    "hvlm_exploreeqa_logs",
    "hvlm_open_eqa_choices_logs",
    "hvlm_open_eqa_logs",
)


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
class EpisodeLog:
    """Single episode result loaded from one log.json."""

    source_dir: str
    episode_id: str
    question: str
    question_category: str
    choices: list[str]
    gt_answer: list[str]
    answer: str
    final_state: str
    num_iters: int | None
    path_length: float | None
    log_path: Path

    @property
    def is_finished(self) -> bool:
        return self.final_state.strip().lower() == "finished"

    @property
    def has_answer(self) -> bool:
        return bool(self.answer.strip())


@dataclass
class JudgeResult:
    """LLM grading output for one episode."""

    is_correct: bool
    confidence: float
    reason: str
    from_cache: bool = False


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a set of episodes."""

    label: str
    dataset: str
    uses_choices: bool
    total_episodes: int
    finished_episodes: int
    correct_episodes: int
    avg_num_iters_all: float
    avg_num_iters_finished: float
    avg_num_iters_correct: float
    avg_path_length_all: float
    avg_path_length_finished: float
    avg_path_length_correct: float
    success_rate: float
    success_rate_finished_answered: float
    llm_evaluated: int


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


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
        description="Evaluate embodied QA logs with OpenAI-based semantic grading."
    )
    parser.add_argument(
        "--results-folder",
        type=Path,
        required=True,
        help="Root folder containing the result subdirectories.",
    )
    parser.add_argument(
        "--target-dirs",
        nargs="+",
        default=list(DEFAULT_TARGET_DIRS),
        help="Subdirectory names to evaluate under results_folder.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model for correctness grading.",
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
        "--strict-finished",
        action="store_true",
        help="Only grade episodes with final_state='finished'.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save full metrics as JSON.",
    )
    parser.add_argument(
        "--by-question-category",
        action="store_true",
        help="Also report per-question-category metrics within each folder.",
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
        default="Embodied QA evaluation summary.",
        help="Caption used in the generated LaTeX table.",
    )
    parser.add_argument(
        "--latex-label",
        default="tab:eqa_eval_summary",
        help="Label used in the generated LaTeX table.",
    )
    return parser.parse_args()


def load_episode_log(log_path: Path, source_dir: str) -> EpisodeLog | None:
    """Load one episode log from JSON."""
    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    choices = payload.get("choices", []) or []
    if not isinstance(choices, list):
        choices = []
    choices = [str(c) for c in choices]

    gt_answer = payload.get("gt_answer", []) or []
    if isinstance(gt_answer, str):
        gt_answer = [gt_answer]
    if not isinstance(gt_answer, list):
        gt_answer = []
    gt_answer = [str(g) for g in gt_answer]

    return EpisodeLog(
        source_dir=source_dir,
        episode_id=log_path.parent.name,
        question=str(payload.get("question", "")),
        question_category=str(payload.get("question_category", "unknown")),
        choices=choices,
        gt_answer=gt_answer,
        answer=str(payload.get("answer", "")),
        final_state=str(payload.get("final_state", "")),
        num_iters=_int_or_none(payload.get("num_iters")),
        path_length=_float_or_none(payload.get("path_length")),
        log_path=log_path,
    )


def discover_logs(
    results_root: Path, target_dirs: list[str]
) -> dict[str, list[EpisodeLog]]:
    """Recursively discover and load episode logs for each requested directory."""
    out: dict[str, list[EpisodeLog]] = {}
    for dirname in target_dirs:
        run_dir = results_root / dirname
        episodes: list[EpisodeLog] = []
        if run_dir.exists() and run_dir.is_dir():
            for log_path in sorted(run_dir.rglob("log.json")):
                loaded = load_episode_log(log_path, source_dir=dirname)
                if loaded is not None:
                    episodes.append(loaded)
        out[dirname] = episodes
    return out


def infer_dataset_name(dirname: str) -> str:
    """Infer dataset label from run directory name."""
    lowered = dirname.lower()
    if "open_eqa" in lowered:
        return "open_eqa"
    if "exploreeqa" in lowered or "graph" in lowered:
        return "grapheqa"
    return "unknown"


def infer_uses_choices(dirname: str) -> bool:
    """Infer whether the run had multiple-choice options available."""
    return "choices" in dirname.lower()


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _judge_key(ep: EpisodeLog, model: str) -> str:
    """Stable hash key for cacheing LLM judgment for one episode."""
    payload = {
        "model": model,
        "question": ep.question,
        "choices": ep.choices,
        "gt_answer": ep.gt_answer,
        "answer": ep.answer,
        "final_state": ep.final_state,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Load LLM grading cache from disk."""
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
    """Persist LLM grading cache to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def build_judge_prompt(ep: EpisodeLog) -> str:
    """Build user prompt for semantic answer correctness grading."""
    return (
        "Decide if the predicted answer is semantically correct.\n"
        "Return strict JSON only.\n\n"
        f"Question: {ep.question}\n"
        f"Question Category: {ep.question_category}\n"
        f"Choices: {json.dumps(ep.choices, ensure_ascii=False)}\n"
        f"Ground Truth Answers: {json.dumps(ep.gt_answer, ensure_ascii=False)}\n"
        f"Predicted Answer: {ep.answer}\n"
        f"Final State: {ep.final_state}\n"
    )


def fallback_exact_match(ep: EpisodeLog) -> JudgeResult:
    """Fallback grading: normalized exact match against any gt answer."""
    pred = _normalize_text(ep.answer)
    gt_set = {_normalize_text(x) for x in ep.gt_answer if _normalize_text(x)}
    is_correct = bool(pred) and pred in gt_set
    reason = "Exact normalized match" if is_correct else "No exact normalized match"
    return JudgeResult(
        is_correct=is_correct, confidence=1.0 if is_correct else 0.0, reason=reason
    )


def judge_episode(
    ep: EpisodeLog,
    *,
    client: Any | None,
    model: str,
    cache: dict[str, dict[str, Any]],
    disable_llm: bool,
) -> JudgeResult:
    """Judge answer correctness for one episode."""
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


def mean(values: list[float]) -> float:
    """Safe arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate_metrics(
    label: str,
    episodes: list[EpisodeLog],
    *,
    client: Any | None,
    model: str,
    cache: dict[str, dict[str, Any]],
    disable_llm: bool,
    strict_finished: bool,
) -> AggregatedMetrics:
    """Compute aggregated metrics for one run directory."""
    finished = [ep for ep in episodes if ep.is_finished]
    finished_answered = [ep for ep in episodes if ep.is_finished and ep.has_answer]
    to_grade = episodes
    correct_episodes: list[EpisodeLog] = []
    correct_finished_answered = 0
    llm_evaluated = 0
    for ep in to_grade:
        judged = judge_episode(
            ep,
            client=client,
            model=model,
            cache=cache,
            disable_llm=disable_llm,
        )
        if ep.has_answer and not disable_llm and client is not None:
            llm_evaluated += 0 if judged.from_cache else 1
        if judged.is_correct:
            correct_episodes.append(ep)
            if ep.is_finished and ep.has_answer:
                correct_finished_answered += 1

    def _metrics_for(scope: list[EpisodeLog]) -> tuple[float, float, int]:
        num_iters_values = [
            float(ep.num_iters) for ep in scope if ep.num_iters is not None
        ]
        path_length_values = [
            ep.path_length for ep in scope if ep.path_length is not None
        ]
        return mean(num_iters_values), mean(path_length_values), len(scope)

    avg_num_iters_all, avg_path_length_all, total_episodes = _metrics_for(episodes)
    avg_num_iters_finished, avg_path_length_finished, finished_episodes = _metrics_for(
        finished
    )
    avg_num_iters_correct, avg_path_length_correct, correct_count = _metrics_for(
        correct_episodes
    )

    denom = max(len(episodes), 1)
    success_rate = len(correct_episodes) / denom
    success_rate_finished_answered = correct_finished_answered / max(
        len(finished_answered), 1
    )

    return AggregatedMetrics(
        label=label,
        dataset=infer_dataset_name(label),
        uses_choices=infer_uses_choices(label),
        total_episodes=total_episodes,
        finished_episodes=finished_episodes,
        correct_episodes=correct_count,
        avg_num_iters_all=avg_num_iters_all,
        avg_num_iters_finished=avg_num_iters_finished,
        avg_num_iters_correct=avg_num_iters_correct,
        avg_path_length_all=avg_path_length_all,
        avg_path_length_finished=avg_path_length_finished,
        avg_path_length_correct=avg_path_length_correct,
        success_rate=success_rate,
        success_rate_finished_answered=success_rate_finished_answered,
        llm_evaluated=llm_evaluated,
    )


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def format_float(value: float) -> str:
    return f"{value:.3f}"


def render_table(rows: list[AggregatedMetrics]) -> str:
    """Render a fixed-width text table for terminal output."""
    headers = [
        "run_dir",
        "dataset",
        "choices",
        "episodes_all",
        "episodes_finished",
        "episodes_correct",
        "avg_steps_all",
        "avg_steps_finished",
        "avg_steps_correct",
        "avg_path_all",
        "avg_path_finished",
        "avg_path_correct",
        "success_rate",
        "success_rate_finished_answered",
        "new_llm_calls",
    ]
    body = [
        [
            r.label,
            r.dataset,
            "yes" if r.uses_choices else "no",
            str(r.total_episodes),
            str(r.finished_episodes),
            str(r.correct_episodes),
            format_float(r.avg_num_iters_all),
            format_float(r.avg_num_iters_finished),
            format_float(r.avg_num_iters_correct),
            format_float(r.avg_path_length_all),
            format_float(r.avg_path_length_finished),
            format_float(r.avg_path_length_correct),
            format_percent(r.success_rate),
            format_percent(r.success_rate_finished_answered),
            str(r.llm_evaluated),
        ]
        for r in rows
    ]
    widths = [len(h) for h in headers]
    for row in body:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt(headers), sep]
    lines.extend(fmt(row) for row in body)
    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    """Escape plain text for safe LaTeX table rendering."""
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
    """Render a nice LaTeX table using booktabs and resizebox."""
    header = [
        "Run",
        "Dataset",
        "Choices",
        "N",
        "N$_{fin}$",
        "N$_{corr}$",
        "Steps$_{all}$",
        "Steps$_{fin}$",
        "Steps$_{corr}$",
        "Path$_{all}$",
        "Path$_{fin}$",
        "Path$_{corr}$",
        "Succ.",
        "Succ.$_{fin+ans}$",
        "New LLM",
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrcccccccccccc}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        row_values = [
            _latex_escape(row.label),
            _latex_escape(row.dataset),
            "yes" if row.uses_choices else "no",
            str(row.total_episodes),
            str(row.finished_episodes),
            str(row.correct_episodes),
            format_float(row.avg_num_iters_all),
            format_float(row.avg_num_iters_finished),
            format_float(row.avg_num_iters_correct),
            format_float(row.avg_path_length_all),
            format_float(row.avg_path_length_finished),
            format_float(row.avg_path_length_correct),
            format_percent(row.success_rate),
            format_percent(row.success_rate_finished_answered),
            str(row.llm_evaluated),
        ]
        lines.append(" & ".join(row_values) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_client(model: str, disable_llm: bool) -> Any | None:
    """Construct OpenAI client unless LLM grading is disabled."""
    if disable_llm:
        return None
    config_module = importlib.import_module("vlms_python.openai_client.config")
    client_module = importlib.import_module("vlms_python.openai_client.openai_client")
    OpenAIClientConfig = getattr(config_module, "OpenAIClientConfig")
    OpenAIClient = getattr(client_module, "OpenAIClient")
    cfg = OpenAIClientConfig(model=model, system_prompt=JUDGE_SYSTEM_PROMPT)
    return OpenAIClient(cfg)


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

    discovered = discover_logs(results_root, args.target_dirs)
    client = build_client(model=args.model, disable_llm=args.disable_llm)

    rows: list[AggregatedMetrics] = []
    per_category: dict[str, list[AggregatedMetrics]] = {}

    for dirname, episodes in discovered.items():
        run_metrics = aggregate_metrics(
            dirname,
            episodes,
            client=client,
            model=args.model,
            cache=cache,
            disable_llm=args.disable_llm,
            strict_finished=args.strict_finished,
        )
        rows.append(run_metrics)

        if args.by_question_category:
            groups: dict[str, list[EpisodeLog]] = {}
            for ep in episodes:
                groups.setdefault(ep.question_category or "unknown", []).append(ep)
            per_category[dirname] = []
            for category, cat_eps in sorted(groups.items()):
                per_category[dirname].append(
                    aggregate_metrics(
                        label=f"{dirname}:{category}",
                        episodes=cat_eps,
                        client=client,
                        model=args.model,
                        cache=cache,
                        disable_llm=args.disable_llm,
                        strict_finished=args.strict_finished,
                    )
                )

    rows.sort(key=lambda r: (r.dataset, not r.uses_choices, r.label))
    print("\n=== Embodied QA Evaluation Summary ===")
    print(render_table(rows))

    latex_table = render_latex_table(
        rows,
        caption=args.latex_caption,
        label=args.latex_label,
    )

    if args.print_latex:
        print("\n=== LaTeX Table (copy into your paper) ===")
        print(latex_table)

    if args.latex_output is not None:
        latex_out = args.latex_output.expanduser().resolve()
        latex_out.parent.mkdir(parents=True, exist_ok=True)
        latex_out.write_text(latex_table + "\n", encoding="utf-8")
        print(f"Saved LaTeX table to: {latex_out}")

    if args.by_question_category:
        for dirname in sorted(per_category):
            print(f"\n=== Per-category metrics: {dirname} ===")
            print(render_table(per_category[dirname]))

    output_payload: dict[str, Any] = {
        "results_root": str(results_root),
        "model": args.model,
        "disable_llm": args.disable_llm,
        "strict_finished": args.strict_finished,
        "runs": [r.__dict__ for r in rows],
        "per_category": {
            key: [item.__dict__ for item in value]
            for key, value in per_category.items()
        },
    }

    if args.output_json is not None:
        out_path = args.output_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved JSON summary to: {out_path}")

    if not args.disable_llm:
        save_cache(cache_path, cache)
        print(f"Saved/updated LLM cache: {cache_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
