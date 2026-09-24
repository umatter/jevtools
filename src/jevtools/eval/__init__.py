"""Evaluation harness and threshold tuning (spec §11): datasets, metrics, the harness, tuning, experiments.

``jevtools.eval`` measures a jevtools deployment on labelled cases and tunes its policy thresholds:

- :mod:`~jevtools.eval.dataset`: the JSONL case schema (§11.1) and :func:`load`;
- :mod:`~jevtools.eval.harness`: :func:`run` decides every case and attributes failures to their stage;
- :mod:`~jevtools.eval.metrics`: exact match, wrong-execution rate per tier, clarify usefulness, abstention
  precision, ECE/Brier per question family, recall@K, risk–coverage, flip rate (§11.2);
- :mod:`~jevtools.eval.tuning`: Clopper–Pearson thresholds, isotonic calibrators, the certification rule (§11.3,
  §11.4) and a versioned ``policy.toml``;
- :mod:`~jevtools.eval.experiments`: the live experiments E1–E10 (§11.2).

Numbers from the offline simulator or scripted backends test the harness; they are never evidence about Jev.
"""

from jevtools.eval.dataset import EvalCase, Gold, load
from jevtools.eval.experiments import EXPERIMENTS, ExperimentResult, run_experiment
from jevtools.eval.harness import RouterFactory, arun, router_factory_for, run, run_dataset, score_decision
from jevtools.eval.metrics import (
    Rate,
    abstention_precision,
    brier,
    clarify_rate,
    clarify_usefulness,
    ece,
    exact_match,
    family_calibration,
    flip_rate,
    hysteresis_width,
    recall_at_k,
    risk_coverage,
    risk_coverage_curves,
    summarize,
    wrong_execution_rate,
)
from jevtools.eval.report import EvalRecord, EvalReport, QuestionRecord
from jevtools.eval.stats import clopper_pearson_upper
from jevtools.eval.tuning import TuningResult, certify_critical, fit_calibrators, tune, tune_thresholds

__all__ = [
    "EXPERIMENTS",
    "EvalCase",
    "EvalRecord",
    "EvalReport",
    "ExperimentResult",
    "Gold",
    "QuestionRecord",
    "Rate",
    "RouterFactory",
    "TuningResult",
    "abstention_precision",
    "arun",
    "brier",
    "certify_critical",
    "clarify_rate",
    "clarify_usefulness",
    "clopper_pearson_upper",
    "ece",
    "exact_match",
    "family_calibration",
    "fit_calibrators",
    "flip_rate",
    "hysteresis_width",
    "load",
    "recall_at_k",
    "risk_coverage",
    "risk_coverage_curves",
    "router_factory_for",
    "run",
    "run_dataset",
    "run_experiment",
    "score_decision",
    "summarize",
    "tune",
    "tune_thresholds",
    "wrong_execution_rate",
]
