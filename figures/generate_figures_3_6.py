"""Generate manuscript Figures 3--6 from authoritative raw result artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures" / "final"
TABLES = RESULTS / "figures_3_6"
OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {"NSGA-II": "#2166ac", "NSGA-III": "#d6604d", "Random": "#4d4d4d"}
MARKERS = {"NSGA-II": "o", "NSGA-III": "^", "Random": "s"}


def clean_axes(ax, horizontal_grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", width=0.8, length=3)
    if horizontal_grid:
        ax.grid(axis="y", color="#d9d9d9", lw=0.55, alpha=0.7)
        ax.set_axisbelow(True)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure3():
    raw = pd.read_csv(RESULTS / "optimizer" / "hv_igd_by_run_and_evaluation.csv")
    parity = pd.read_csv(RESULTS / "audit" / "budget_to_parity_by_run.csv")
    checkpoints = [28, 56, 84, 112, 140, 168, 196, 224, 252, 280]
    methods = ["NSGA-II", "NSGA-III", "Random"]
    assert set(raw.method) == set(methods)
    assert raw.groupby("method").run.nunique().eq(15).all()
    assert raw.groupby(["method", "run"]).evaluations.apply(lambda x: sorted(x) == checkpoints).all()
    summary = raw.groupby(["method", "evaluations"], sort=False).hypervolume.agg(
        mean_hv="mean", sd_hv="std", n_runs="count"
    ).reset_index()
    summary.to_csv(TABLES / "fig3_checkpoint_summary.csv", index=False)
    parity.rename(columns={"run": "run_id", "evaluations_to_parity": "first_passage_evaluations"})[
        ["method", "run_id", "reached", "first_passage_evaluations"]
    ].to_csv(TABLES / "fig3_first_passage_by_run.csv", index=False)

    final = summary.query("evaluations == 280").set_index("method")
    expected_hv = {"NSGA-II": (1.197236, 0.0185), "NSGA-III": (1.157911, 0.0244), "Random": (1.141019, 0.0164)}
    for method, (mean, sd) in expected_hv.items():
        assert abs(final.loc[method, "mean_hv"] - mean) < 1e-6
        assert abs(final.loc[method, "sd_hv"] - sd) < 6e-5

    target = 1.141019
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.7),
                                  gridspec_kw={"width_ratios": [1.72, 1]},
                                  constrained_layout=True)
    for method in methods:
        g = summary[summary.method.eq(method)]
        label = "Random search" if method == "Random" else method
        ax1.plot(g.evaluations, g.mean_hv, color=COLORS[method], marker=MARKERS[method],
                 ms=3.8, lw=1.45, label=label)
        ax1.fill_between(g.evaluations, g.mean_hv-g.sd_hv, g.mean_hv+g.sd_hv,
                         color=COLORS[method], alpha=0.13, lw=0)
    ax1.axhline(target, color="#555555", ls="--", lw=0.9)
    ax1.set_xlabel("Candidate evaluations", fontsize=11, fontweight="bold", labelpad=7)
    ax1.set_ylabel("Hypervolume (HV)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_xticks(checkpoints)
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(frameon=False, ncol=1, loc="lower right")
    ax1.text(0.01, 0.98, "(a)", transform=ax1.transAxes, va="top", fontweight="bold", fontsize=11)
    clean_axes(ax1)

    rng_offsets = np.linspace(-0.16, 0.16, 15)
    for xpos, method in enumerate(methods):
        g = parity[parity.method.eq(method)].sort_values("run")
        reached = g.reached.astype(bool).to_numpy()
        values = g.evaluations_to_parity.fillna(280).to_numpy(float)
        ax2.scatter(xpos+rng_offsets[reached], values[reached], s=18, marker="o",
                    color=COLORS[method], edgecolor="white", lw=0.35, zorder=3)
        ax2.scatter(xpos+rng_offsets[~reached], np.full((~reached).sum(), 280), s=24, marker="x",
                    color=COLORS[method], lw=1.0, zorder=3)
        vals = values[reached]
        mean = vals.mean()
        ax2.hlines(mean, xpos-0.22, xpos+0.22, color="black", lw=1.8, zorder=4)
        ax2.text(xpos, 8, f"{reached.sum()}/15\n{mean:.1f}", ha="center", va="bottom", fontsize=9.5)
    ax2.set_xticks(range(3), ["NSGA-II", "NSGA-III", "Random"])
    ax2.set_xlabel("Search method", fontsize=11, fontweight="bold", labelpad=7)
    ax2.set_ylabel("Evaluations to target", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_ylim(0, 300)
    ax2.text(0.01, 0.98, "(b)", transform=ax2.transAxes, va="top", fontweight="bold", fontsize=11)
    clean_axes(ax2)
    save(fig, "fig_optimizer_convergence_budget")

    print("Final HV check")
    for method in methods: print(method, f"{final.loc[method,'mean_hv']:.6f}", f"{final.loc[method,'sd_hv']:.6f}")
    print("Budget-parity check")
    for method in methods:
        g=parity[(parity.method.eq(method)) & parity.reached.astype(bool)].evaluations_to_parity
        print(method, len(g), f"{g.mean():.3f}", f"{g.std(ddof=1):.3f}")


def nondominated_mask(values):
    return np.array([not np.any(np.all(values <= row, axis=1) & np.any(values < row, axis=1)) for row in values])


def figure4():
    data = pd.read_csv(RESULTS / "optimizer" / "all_optimizer_trials.csv")
    assert len(data) == 12600 and data.groupby("method").size().eq(4200).all()
    assert data[["validation_macro_MAPE", "parameters", "latency_ms"]].notna().all().all()
    values = np.column_stack([data.validation_macro_MAPE, np.log10(data.parameters), np.log10(data.latency_ms)])
    gmin = np.array([0.7308796223, 4.5613160916, -0.4342352697])
    gmax = np.array([8.8913714513, 5.8939979806, 0.0246944637])
    normalized = (values-gmin)/(gmax-gmin)
    mask = nondominated_mask(normalized)
    front = data.loc[mask, ["method", "run", "evaluation", "candidate_id", "validation_macro_MAPE", "parameters", "latency_ms"]].copy()
    front[["normalised_mape", "normalised_log_params", "normalised_log_latency"]] = normalized[mask]
    front = front.rename(columns={"run": "run_id", "evaluation": "evaluation_id", "parameters": "parameter_count"})
    counts = front.groupby("method").size().to_dict()
    assert len(front) == 28 and counts == {"NSGA-II": 21, "NSGA-III": 2, "Random": 5}
    front.to_csv(TABLES / "pooled_nondominated_28.csv", index=False)

    panels = [
        ("parameters", "validation_macro_MAPE", "Trainable parameters", "Validation macro MAPE (%)", "fig4a_pareto_mape_parameters"),
        ("latency_ms", "validation_macro_MAPE", "Inference latency (ms)", "Validation macro MAPE (%)", "fig4b_pareto_mape_latency"),
    ]
    for x, y, xlabel, ylabel, stem in panels:
        fig, ax = plt.subplots(figsize=(4.6, 3.65), constrained_layout=True)
        ax.scatter(data[x], data[y], s=2.0, color="#bdbdbd", alpha=0.20, edgecolors="none", rasterized=True)
        for method in ["NSGA-II", "NSGA-III", "Random"]:
            g = data.loc[mask & data.method.eq(method)]
            label = "Random search" if method == "Random" else method
            ax.scatter(g[x], g[y], s=25, marker=MARKERS[method], color=COLORS[method],
                       edgecolor="#222222", lw=0.45, label=label, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold", labelpad=7)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", labelpad=8)
        clean_axes(ax)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3,
                  handletextpad=0.25, columnspacing=0.7, borderaxespad=0)
        save(fig, stem)
    print("Total candidate rows:", len(data))
    print("Pooled nondominated points:", len(front))
    for method in ["NSGA-II", "NSGA-III", "Random"]: print(method, "contribution:", counts[method])


def figure5():
    rows = [
        ("Full features", 1.168, 1.457),
        ("No temperature", 1.312, 2.004),
        ("No ageing index", 1.374, 1.847),
        (r"$k_{exp}+T$", 0.613, 1.938),
        (r"$k_{exp}+R_{e,0}$", 0.616, 1.284),
        (r"$k_{exp}+T+R_{e,0}$", 0.682, 2.090),
        (r"$k_{exp}+T+R_{e,0}+R_{pulse,0}+Q_0$", 0.806, 1.406),
    ]
    data = pd.DataFrame(rows, columns=["feature_set", "teacher_mape", "rollout_mape"])
    data["teacher_rank"] = data.teacher_mape.rank(method="min").astype(int)
    data["rollout_rank"] = data.rollout_mape.rank(method="min").astype(int)
    assert data.teacher_rank.tolist() == [5,6,7,1,2,3,4]
    assert data.rollout_rank.tolist() == [3,6,4,5,1,7,2]
    data.to_csv(TABLES / "fig5_teacher_rollout_values.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    highlight = {r"$k_{exp}+T$": "#b2182b", r"$k_{exp}+R_{e,0}$": "#2166ac"}
    left_offsets = {"Full features": .018, "No temperature": -.012, "No ageing index": .018,
                    r"$k_{exp}+T$": -.035, r"$k_{exp}+R_{e,0}$": .035,
                    r"$k_{exp}+T+R_{e,0}$": .01, r"$k_{exp}+T+R_{e,0}+R_{pulse,0}+Q_0$": 0}
    right_offsets = {1.457: .02, 1.406: -.02, 1.938: -.025, 2.004: .015, 2.090: 0, 1.847: 0, 1.284: 0}
    for row in data.itertuples(index=False):
        color = highlight.get(row.feature_set, "#8c8c8c")
        lw = 2.0 if row.feature_set in highlight else (1.35 if row.feature_set == "Full features" else 0.9)
        alpha = 1 if row.feature_set in highlight else 0.72
        ax.plot([0,1], [row.teacher_mape,row.rollout_mape], color=color, lw=lw, alpha=alpha)
        ax.scatter([0,1], [row.teacher_mape,row.rollout_mape], color=color, s=28 if row.feature_set in highlight else 18, zorder=3)
        ax.text(-0.055, row.teacher_mape+left_offsets[row.feature_set], row.feature_set,
                ha="right", va="center", fontsize=9.5, color=color)
        ax.text(1.055, row.rollout_mape+right_offsets[row.rollout_mape], f"{row.rollout_mape:.3f}",
                ha="left", va="center", fontsize=9.5, color=color)
    t = data.loc[data.feature_set.eq(r"$k_{exp}+T$")].iloc[0]
    r = data.loc[data.feature_set.eq(r"$k_{exp}+R_{e,0}$")].iloc[0]
    ax.text(.56, 1.72, "Rank 1 → 5", color=highlight[t.feature_set], fontsize=9.5)
    ax.text(.57, 1.03, "Rank 2 → 1", color=highlight[r.feature_set], fontsize=9.5)
    ax.set_xlim(-.75, 1.28); ax.set_ylim(.5, 2.2)
    ax.set_xticks([0,1], ["Teacher forced", "Recursive rollout"])
    ax.tick_params(axis="x", pad=17)
    for label in ax.get_xticklabels(): label.set_fontweight("bold")
    ax.set_ylabel("Macro MAPE (%)", fontsize=11, fontweight="bold", labelpad=8)
    clean_axes(ax)
    fig.subplots_adjust(left=.12, right=.88, top=.97, bottom=.22)
    save(fig, "fig_teacher_rollout_ablation")
    print(data.to_string(index=False))


def figure6():
    raw = pd.read_csv(RESULTS / "transfer" / "transfer_cell_metrics.csv")
    models = ["source_pretrained_head_adapted", "target_only_LSTM_from_scratch"]
    paired = raw[raw.model.isin(models)].groupby(["cell_key", "model"]).Q_MAPE.mean().unstack()
    paired = paired.rename(columns={models[0]: "transfer_mape", models[1]: "target_only_mape"}).dropna()
    paired["delta_pp"] = paired.transfer_mape-paired.target_only_mape
    delta_for_statistics = paired.delta_pp.to_numpy()
    paired = paired.sort_values("delta_pp").reset_index().rename(columns={"cell_key": "cell_id"})
    delta = paired.delta_pp.to_numpy()
    rng = np.random.default_rng(20260820)
    boot = np.array([rng.choice(delta_for_statistics, len(delta_for_statistics), replace=True).mean() for _ in range(10000)])
    ci = np.quantile(boot, [.025,.975])
    statistic, pvalue = wilcoxon(delta, alternative="two-sided", method="auto")
    expected = pd.read_csv(RESULTS / "transfer" / "transfer_cell_level_tests.csv").iloc[0]
    assert len(delta)==40
    assert abs(delta.mean()-expected.mean_cell_MAPE_difference_a_minus_b_pp)<1e-12
    assert abs(np.median(delta)-expected.median_cell_MAPE_difference_a_minus_b_pp)<1e-12
    assert np.allclose(ci,[expected.bootstrap_95ci_low,expected.bootstrap_95ci_high],atol=1e-12)
    assert statistic==expected.wilcoxon_W and abs(pvalue-expected.wilcoxon_p_two_sided)<1e-18
    paired.to_csv(TABLES / "fig6_transfer_cell_effects.csv", index=False)
    summary = {"n": len(delta), "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
               "bootstrap_ci_low": float(ci[0]), "bootstrap_ci_high": float(ci[1]),
               "wilcoxon_W": float(statistic), "wilcoxon_p": float(pvalue), "bootstrap_seed": 20260820,
               "bootstrap_resamples": 10000}
    (TABLES / "fig6_transfer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, (ax, ax_mean) = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True,
                                     gridspec_kw={"width_ratios": [8.5, 1.2]},
                                     constrained_layout=True)
    x=np.arange(1,41)
    colors=np.where(delta>=0,"#b2182b","#2166ac")
    ax.scatter(x,delta,c=colors,s=23,edgecolor="white",lw=.35,zorder=3)
    ax.axhline(0,color="#555555",lw=.9)
    ax_mean.axhline(0,color="#555555",lw=.9)
    ax_mean.errorbar(0,delta.mean(),yerr=[[delta.mean()-ci[0]],[ci[1]-delta.mean()]],fmt="D",
                color="black",mfc="white",mec="black",ms=6,capsize=4,lw=1.2,zorder=4)
    ax_mean.text(0,ci[1]+.65,f"{delta.mean():+.2f}",ha="center",fontsize=10,fontweight="bold")
    ax.set_xlim(0,41); ax.set_xticks([1,10,20,30,40])
    ax.set_xlabel("External cells ordered by paired effect", fontsize=11, fontweight="bold", labelpad=7)
    ax.set_ylabel("Paired MAPE difference (pp)", fontsize=11, fontweight="bold", labelpad=8)
    ax_mean.set_xlim(-.8,.8); ax_mean.set_xticks([0],["Mean"])
    ax_mean.tick_params(axis="x",pad=8)
    for label in ax_mean.get_xticklabels(): label.set_fontweight("bold")
    ax_mean.set_title("95% CI",fontsize=10,fontweight="bold",pad=7)
    clean_axes(ax)
    clean_axes(ax_mean)
    ax_mean.grid(False)
    legend_handles = [
        Line2D([0],[0],marker="o",linestyle="none",markerfacecolor="#b2182b",markeredgecolor="white",label="Negative transfer"),
        Line2D([0],[0],marker="o",linestyle="none",markerfacecolor="#2166ac",markeredgecolor="white",label="Beneficial transfer"),
        Line2D([0],[0],marker="D",linestyle="none",markerfacecolor="white",markeredgecolor="black",label="Mean and 95% CI"),
    ]
    fig.legend(handles=legend_handles,loc="upper center",bbox_to_anchor=(0.5,1.08),ncol=3,
               frameon=False,columnspacing=1.4,handletextpad=.4)
    save(fig, "fig_external_transfer")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    figure3(); figure4(); figure5(); figure6()
