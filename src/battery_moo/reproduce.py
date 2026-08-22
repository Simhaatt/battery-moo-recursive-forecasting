"""Artifact-only reproduction, plotting, and manuscript contract validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from .integrity import validate_search_trials


def _actual(repo: Path, spec: dict) -> float:
    results = repo / "results"
    source, selector = spec["source"], spec.get("selector")
    if source == "optimizer_final_hv":
        df = pd.read_csv(results / "optimizer/hv_igd_by_run_and_evaluation.csv")
        return float(df.query("evaluations == 280 and method == @selector")["hypervolume"].mean())
    if source == "pooled_count":
        df = pd.read_csv(results / "audit/pooled_pareto_tidy.csv")
        return float(len(df) if selector == "all" else (df["method"] == selector).sum())
    paths = {
        "final_summary": "final/final_ten_seed_summary.csv",
        "paired_test": "final/final_paired_tests_holm.csv",
        "stress_summary": "stress/stress_ablation_summary.csv",
        "transfer_test": "transfer/transfer_cell_level_tests.csv",
        "transfer_summary": "transfer/transfer_summary.csv",
        "budget_summary": "audit/budget_to_parity_summary.csv",
    }
    keys = {
        "final_summary": "configuration", "paired_test": "comparison",
        "stress_summary": "variant", "transfer_test": "model_a",
        "transfer_summary": "model", "budget_summary": "method",
    }
    if source in paths:
        df = pd.read_csv(results / paths[source])
        return float(df.loc[df[keys[source]].eq(selector), spec["column"]].iloc[0])
    if source == "derived_parameter_reduction":
        df = pd.read_csv(results / "final/final_ten_seed_summary.csv").set_index("configuration")
        return float((1 - df.loc[selector, "parameters_mean"] / df.loc["Manual", "parameters_mean"]) * 100)
    raise KeyError(f"Unknown expected-value source: {source}")


def manuscript_check(repo: Path) -> list[dict]:
    contract = yaml.safe_load((repo / "configs/expected_values.yaml").read_text(encoding="utf-8"))
    default_tol = float(contract["tolerance"])
    rows = []
    for spec in contract["checks"]:
        actual = _actual(repo, spec)
        expected = float(spec["expected"])
        tol = float(spec.get("tolerance", default_tol))
        rows.append({
            "id": spec["id"], "metric": spec["label"], "label": spec["label"], "expected": expected,
            "recomputed": actual, "reproduced": actual, "difference": actual - expected,
            "absolute_error": abs(actual - expected),
            "tolerance": tol, "status": "PASS" if abs(actual - expected) <= tol else "FAIL",
        })
    return rows


def make_tables(repo: Path) -> None:
    out = repo / "results/generated"; out.mkdir(parents=True, exist_ok=True)
    hv = pd.read_csv(repo / "results/optimizer/hv_igd_by_run_and_evaluation.csv")
    hv.query("evaluations == 280").groupby("method").agg(
        hypervolume_mean=("hypervolume", "mean"), hypervolume_std=("hypervolume", "std"),
        igd_mean=("IGD", "mean"), igd_std=("IGD", "std"), runs=("run", "nunique")
    ).reset_index().to_csv(out / "table_front_statistics.csv", index=False)
    pd.read_csv(repo / "results/final/final_ten_seed_summary.csv").to_csv(out / "table_final_confirmation.csv", index=False)
    pd.read_csv(repo / "results/audit/budget_to_parity_summary.csv").to_csv(out / "table_budget_parity.csv", index=False)
    pd.read_csv(repo / "results/stress/stress_ablation_summary.csv").to_csv(out / "table_stress_ablation.csv", index=False)
    pd.read_csv(repo / "results/transfer/transfer_summary.csv").to_csv(out / "table_transfer.csv", index=False)
    pd.read_csv(repo / "results/grouped_cv/phase1_grouped_cv_metrics_raw.csv").to_csv(out / "table_grouped_cv.csv", index=False)


def make_figures(repo: Path) -> None:
    out = repo / "figures/generated"; out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    conv = pd.read_csv(repo / "results/optimizer/hv_igd_convergence_summary.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, g in conv.groupby("method", sort=False):
        ax.plot(g.evaluations, g.hv_mean, marker="o", ms=3, label=method)
        ax.fill_between(g.evaluations, g.hv_mean-g.hv_std, g.hv_mean+g.hv_std, alpha=.14)
    ax.set(xlabel="Objective evaluations", ylabel="Hypervolume")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "optimizer_convergence.png", dpi=300); plt.close(fig)

    pool = pd.read_csv(repo / "results/audit/pooled_pareto_tidy.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for method, g in pool.groupby("method", sort=False):
        ax.scatter(g.parameters, g.validation_macro_MAPE, label=method, alpha=.85)
    ax.set_xscale("log"); ax.set(xlabel="Trainable parameters", ylabel="Validation macro MAPE (%)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "pooled_pareto.png", dpi=300); plt.close(fig)

    budget = pd.read_csv(repo / "results/audit/budget_to_parity_summary.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.bar(budget.method, budget.mean_evaluations_reaching, yerr=budget.sd_evaluations_reaching, capsize=4)
    ax.set(ylabel="Evaluations to parity (reaching runs)")
    fig.tight_layout(); fig.savefig(out / "budget_to_parity.png", dpi=300); plt.close(fig)

    stress = pd.read_csv(repo / "results/stress/stress_ablation_summary.csv")
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.bar(["Base inputs", "+ stress"], stress.mean_macro_MAPE, yerr=stress.std_macro_MAPE, capsize=4)
    ax.set(ylabel="Test macro MAPE (%)"); fig.tight_layout(); fig.savefig(out / "stress_ablation.png", dpi=300); plt.close(fig)

    ablation = pd.read_csv(repo / "results/baselines/phase1e_current_model_ablation__phase1e_current_model_ablation_macro_summary.csv")
    ablation = ablation.query("protocol == 'autoregressive_rollout'").sort_values("MAPE").head(10)
    labels = ablation.variant.str.replace("Rct0", "Rpulse0", regex=False)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(labels[::-1], ablation.MAPE[::-1])
    ax.set(xlabel="Recursive macro MAPE (%)", ylabel="Input variant")
    fig.tight_layout(); fig.savefig(out / "feature_ablation.png", dpi=300); plt.close(fig)

    transfer = pd.read_csv(repo / "results/transfer/transfer_summary.csv").sort_values("mean_SOH_MAPE")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(transfer.model, transfer.mean_SOH_MAPE, xerr=transfer.std_SOH_MAPE.fillna(0), capsize=3)
    ax.set(xlabel="External SOH MAPE (%)"); fig.tight_layout(); fig.savefig(out / "transfer_baselines.png", dpi=300); plt.close(fig)


def run(repo: Path) -> int:
    make_tables(repo); make_figures(repo)
    trials = pd.read_csv(repo / "results/optimizer/all_optimizer_trials.csv")
    integrity = validate_search_trials(trials)
    checks = manuscript_check(repo)
    payload = {
        "overall_status": "PASS" if all(integrity.values()) and all(x["status"] == "PASS" for x in checks) else "FAIL",
        "manuscript_checks": checks,
        "search_integrity": integrity,
    }
    out = repo / "results/reproduction_check.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = ["# Reproduction check", "", f"Overall status: **{payload['overall_status']}**", "",
          "| Expected manuscript value | Reproduced value | Status |", "|---|---:|:---:|"]
    md += [f"| {x['label']} ({x['expected']:.6f}) | {x['reproduced']:.6f} | {x['status']} |" for x in checks]
    md += ["", "## Search-integrity checks", ""] + [f"- [{'x' if ok else ' '}] {name}" for name, ok in integrity.items()]
    (repo / "results/REPRODUCTION_CHECK.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print("Expected manuscript value\tReproduced value\tStatus")
    for x in checks: print(f"{x['label']} {x['expected']:.6f}\t{x['reproduced']:.6f}\t{x['status']}")
    print(f"OVERALL: {payload['overall_status']}")
    return 0 if payload["overall_status"] == "PASS" else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    raise SystemExit(run(args.repo.resolve()))


if __name__ == "__main__": main()
