"""Differential-evolution optimization of the rollout LSTM.

Search variables: sequence length L, hidden width, LSTM layers, differentiable
rollout horizon H, and learning rate.  Fitness is validation rollout macro MAPE;
the fixed test trajectories are used only after the search is complete.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from run_round2_experiments import (
    DEVICE,
    FULL9,
    LSTMMulti,
    OUTPUT,
    SEEDS,
    TARGETS,
    TrainConfig,
    build_sequences,
    load_phase1_split,
    metrics_2d,
    rollout_predict,
    set_seed,
)


L_VALUES = np.array([10, 15, 20, 25])
HIDDEN_VALUES = np.array([64, 96, 128, 160, 192])
LAYER_VALUES = np.array([1, 2, 3])
HORIZON_VALUES = np.array([3, 5, 8, 10, 15])
LR_LOG10_BOUNDS = (-4.0, math.log10(2e-3))
MANUAL = {"L": 20, "hidden": 192, "layers": 2, "H": 10, "lr": 1e-3}


@dataclass(frozen=True)
class Candidate:
    L: int
    hidden: int
    layers: int
    H: int
    lr: float

    @property
    def key(self):
        return (self.L, self.hidden, self.layers, self.H, round(self.lr, 8))


def decode(v: np.ndarray) -> Candidate:
    return Candidate(
        L=int(L_VALUES[np.argmin(np.abs(L_VALUES - v[0]))]),
        hidden=int(HIDDEN_VALUES[np.argmin(np.abs(HIDDEN_VALUES - v[1]))]),
        layers=int(LAYER_VALUES[np.argmin(np.abs(LAYER_VALUES - v[2]))]),
        H=int(HORIZON_VALUES[np.argmin(np.abs(HORIZON_VALUES - v[3]))]),
        lr=float(10 ** np.clip(v[4], *LR_LOG10_BOUNDS)),
    )


BOUNDS = np.array(
    [
        [L_VALUES.min(), L_VALUES.max()],
        [HIDDEN_VALUES.min(), HIDDEN_VALUES.max()],
        [LAYER_VALUES.min(), LAYER_VALUES.max()],
        [HORIZON_VALUES.min(), HORIZON_VALUES.max()],
        [LR_LOG10_BOUNDS[0], LR_LOG10_BOUNDS[1]],
    ],
    dtype=float,
)


def scaled_rollout_segments(df, features, L, H, sc_x, sc_y, stride=2):
    x0s, exogs, ys = [], [], []
    nf = len(features)
    x_mean, x_scale = sc_x.mean_, sc_x.scale_
    for _, g0 in df.groupby("cell_id", sort=False):
        g = g0.sort_values("k_exp").reset_index(drop=True)
        full = g[features + TARGETS].to_numpy(np.float32)
        exog = g[features].to_numpy(np.float32)
        targ = g[TARGETS].to_numpy(np.float32)
        if len(g) < L + H:
            continue
        full_s = ((full - x_mean) / x_scale).astype(np.float32)
        exog_s = ((exog - x_mean[:nf]) / x_scale[:nf]).astype(np.float32)
        targ_s = sc_y.transform(targ).astype(np.float32)
        for i in range(L, len(g) - H + 1, stride):
            x0s.append(full_s[i - L : i])
            exogs.append(exog_s[i : i + H])
            ys.append(targ_s[i : i + H])
    return np.asarray(x0s), np.asarray(exogs), np.asarray(ys)


def teacher_pretrain(train, val, candidate: Candidate, seed: int, epochs: int, patience: int):
    set_seed(seed)
    xtr, ytr = build_sequences(train, FULL9, candidate.L)
    xva, yva = build_sequences(val, FULL9, candidate.L)
    nf = xtr.shape[-1]
    sc_x, sc_y = StandardScaler(), StandardScaler()
    xtr = sc_x.fit_transform(xtr.reshape(-1, nf)).reshape(xtr.shape).astype(np.float32)
    xva = sc_x.transform(xva.reshape(-1, nf)).reshape(xva.shape).astype(np.float32)
    ytr = sc_y.fit_transform(ytr).astype(np.float32)
    yva = sc_y.transform(yva).astype(np.float32)
    model = LSTMMulti(nf, candidate.hidden, candidate.layers).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=candidate.lr, weight_decay=1e-5)
    loader = DataLoader(
        TensorDataset(torch.tensor(xtr), torch.tensor(ytr)),
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    xv = torch.tensor(xva, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(yva, dtype=torch.float32, device=DEVICE)
    best, best_metric, bad, history = None, np.inf, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.smooth_l1_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            yvp = sc_y.inverse_transform(model(xv).cpu().numpy())
        val_metric = metrics_2d(sc_y.inverse_transform(yva), yvp)["macro_MAPE"]
        history.append({"stage": "teacher", "epoch": epoch, "train_loss": np.mean(losses), "val_direct_macro_MAPE": val_metric})
        if val_metric < best_metric - 1e-7:
            best_metric, bad = val_metric, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best)
    return model, sc_x, sc_y, opt, history


def rollout_finetune(
    model,
    train,
    val,
    candidate: Candidate,
    sc_x,
    sc_y,
    seed,
    epochs,
    patience,
):
    x0, exog, targ = scaled_rollout_segments(
        train, FULL9, candidate.L, candidate.H, sc_x, sc_y, stride=2
    )
    if not len(x0):
        return [], np.inf
    loader = DataLoader(
        TensorDataset(torch.tensor(x0), torch.tensor(exog), torch.tensor(targ)),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=candidate.lr * 0.30, weight_decay=1e-5)
    yt0, yp0, _, _ = rollout_predict(model, sc_x, sc_y, val, FULL9, candidate.L)
    best_metric = metrics_2d(yt0, yp0)["macro_MAPE"]
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bad, history = 0, [{"stage": "rollout", "epoch": 0, "val_macro_MAPE": best_metric}]
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for win, fx, yy in loader:
            win, fx, yy = win.to(DEVICE), fx.to(DEVICE), yy.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            step_losses = []
            for step in range(candidate.H):
                yp = model(win)
                step_losses.append(nn.functional.mse_loss(yp, yy[:, step, :]))
                nxt = torch.cat([fx[:, step, :], yp], dim=1).unsqueeze(1)
                win = torch.cat([win[:, 1:, :], nxt], dim=1)
            loss = torch.stack(step_losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        yt, yp, _, _ = rollout_predict(model, sc_x, sc_y, val, FULL9, candidate.L)
        metric = metrics_2d(yt, yp)["macro_MAPE"]
        history.append(
            {"stage": "rollout", "epoch": epoch, "train_loss": np.mean(losses), "val_macro_MAPE": metric}
        )
        if metric < best_metric - 1e-5:
            best_metric, bad = metric, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if best is not None:
        model.load_state_dict(best)
    return history, best_metric


def train_candidate(
    train,
    val,
    candidate,
    seed=2026,
    teacher_epochs=32,
    teacher_patience=6,
    rollout_epochs=10,
    rollout_patience=3,
):
    t0 = time.perf_counter()
    model, sc_x, sc_y, _, h1 = teacher_pretrain(
        train, val, candidate, seed, teacher_epochs, teacher_patience
    )
    h2, best_metric = rollout_finetune(
        model,
        train,
        val,
        candidate,
        sc_x,
        sc_y,
        seed,
        rollout_epochs,
        rollout_patience,
    )
    seconds = time.perf_counter() - t0
    if not np.isfinite(best_metric):
        yt, yp, _, _ = rollout_predict(model, sc_x, sc_y, val, FULL9, candidate.L)
        best_metric = metrics_2d(yt, yp)["macro_MAPE"]
    return model, sc_x, sc_y, pd.DataFrame(h1 + h2), float(best_metric), seconds


def run_de(pop_size=8, generations=3, F=0.7, CR=0.8):
    train, val, test = load_phase1_split()
    rng = np.random.default_rng(20260819)
    pop = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(pop_size, len(BOUNDS)))
    pop[0] = np.array([MANUAL["L"], MANUAL["hidden"], MANUAL["layers"], MANUAL["H"], math.log10(MANUAL["lr"])])
    cache, trials, convergence = {}, [], []
    total_t0 = time.perf_counter()

    def evaluate(vector, generation, slot):
        cand = decode(vector)
        if cand.key in cache:
            metric, seconds = cache[cand.key]
            cached = True
        else:
            _, _, _, _, metric, seconds = train_candidate(
                train, val, cand, teacher_epochs=80, teacher_patience=18,
                rollout_epochs=15, rollout_patience=5,
            )
            cache[cand.key] = (metric, seconds)
            cached = False
        trials.append(
            {
                "generation": generation,
                "slot": slot,
                **asdict(cand),
                "validation_macro_MAPE": metric,
                "seconds": seconds,
                "cached": cached,
            }
        )
        print(f"g={generation} slot={slot} {cand} val={metric:.4f}% time={seconds:.1f}s cached={cached}")
        return metric

    fitness = np.array([evaluate(pop[i], 0, i) for i in range(pop_size)])
    convergence.append({"generation": 0, "best_validation_macro_MAPE": fitness.min(), "evaluations": len(cache)})
    for generation in range(1, generations + 1):
        for i in range(pop_size):
            choices = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(choices, size=3, replace=False)
            mutant = np.clip(pop[a] + F * (pop[b] - pop[c]), BOUNDS[:, 0], BOUNDS[:, 1])
            mask = rng.random(len(BOUNDS)) < CR
            mask[rng.integers(0, len(BOUNDS))] = True
            trial = np.where(mask, mutant, pop[i])
            trial_fit = evaluate(trial, generation, i)
            if trial_fit <= fitness[i]:
                pop[i], fitness[i] = trial, trial_fit
        convergence.append(
            {"generation": generation, "best_validation_macro_MAPE": fitness.min(), "evaluations": len(cache)}
        )

    trial_df = pd.DataFrame(trials)
    conv_df = pd.DataFrame(convergence)
    trial_df.to_csv(OUTPUT / "de_search_trials.csv", index=False)
    conv_df.to_csv(OUTPUT / "de_search_convergence.csv", index=False)
    best_row = trial_df.loc[trial_df["validation_macro_MAPE"].idxmin()]
    optimized = Candidate(
        int(best_row.L), int(best_row.hidden), int(best_row.layers), int(best_row.H), float(best_row.lr)
    )

    final_rows, final_histories = [], []
    for label, candidate in [("manual", Candidate(**MANUAL)), ("DE_optimized", optimized)]:
        for seed in SEEDS:
            model, sc_x, sc_y, hist, val_metric, seconds = train_candidate(
                train,
                val,
                candidate,
                seed=seed,
                teacher_epochs=220,
                teacher_patience=40,
                rollout_epochs=40,
                rollout_patience=15,
            )
            yt, yp, meta, infer_seconds = rollout_predict(model, sc_x, sc_y, test, FULL9, candidate.L)
            row = {
                "configuration": label,
                "seed": seed,
                **asdict(candidate),
                "validation_macro_MAPE": val_metric,
                **metrics_2d(yt, yp),
                "train_seconds": seconds,
                "inference_ms_per_trajectory": infer_seconds / meta["cell_id"].nunique() * 1000,
            }
            final_rows.append(row)
            hist["configuration"], hist["seed"] = label, seed
            final_histories.append(hist)
            print(label, seed, f"test={row['macro_MAPE']:.4f}%")
    final = pd.DataFrame(final_rows)
    final.to_csv(OUTPUT / "de_final_manual_vs_optimized_seed_metrics.csv", index=False)
    pd.concat(final_histories, ignore_index=True).to_csv(OUTPUT / "de_final_training_history.csv", index=False)
    summary = final.groupby("configuration", as_index=False).agg(
        mean_macro_MAPE=("macro_MAPE", "mean"),
        std_macro_MAPE=("macro_MAPE", "std"),
        mean_Q_MAPE=("Q_MAPE", "mean"),
        std_Q_MAPE=("Q_MAPE", "std"),
        mean_Re_MAPE=("Re_MAPE", "mean"),
        std_Re_MAPE=("Re_MAPE", "std"),
        mean_train_seconds=("train_seconds", "mean"),
        mean_inference_ms_per_trajectory=("inference_ms_per_trajectory", "mean"),
    )
    summary.to_csv(OUTPUT / "de_final_manual_vs_optimized_summary.csv", index=False)

    search_cost = {
        "algorithm": "differential evolution",
        "population_size": pop_size,
        "generations": generations,
        "requested_trial_evaluations": int(pop_size * (generations + 1)),
        "unique_trained_configurations": len(cache),
        "wall_seconds_search_and_final": time.perf_counter() - total_t0,
        "search_training_seconds_sum": float(sum(v[1] for v in cache.values())),
        "manual_configuration": MANUAL,
        "optimized_configuration": asdict(optimized),
        "best_validation_macro_MAPE": float(best_row.validation_macro_MAPE),
    }
    (OUTPUT / "de_search_cost.json").write_text(json.dumps(search_cost, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(conv_df["generation"], conv_df["best_validation_macro_MAPE"], marker="o", color="#235789")
    ax.set_xlabel("DE generation")
    ax.set_ylabel("Best validation rollout macro MAPE (%)")
    ax.set_title("Differential-evolution convergence")
    ax.set_xticks(conv_df["generation"])
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "figure_de_convergence.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure_de_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(search_cost, indent=2))


def run_stochastic_rescreen(top_k=5, extra_seeds=(42, 52)):
    """Re-evaluate the best DE candidates to reduce single-seed selection noise."""
    train, val, test = load_phase1_split()
    trials = pd.read_csv(OUTPUT / "de_search_trials.csv").sort_values("validation_macro_MAPE")
    trials["key"] = trials.apply(
        lambda r: f"{int(r.L)}-{int(r.hidden)}-{int(r.layers)}-{int(r.H)}-{float(r.lr):.8g}", axis=1
    )
    top = trials.drop_duplicates("key").head(top_k)
    rows = []
    for _, r in top.iterrows():
        cand = Candidate(int(r.L), int(r.hidden), int(r.layers), int(r.H), float(r.lr))
        rows.append({**asdict(cand), "seed": 2026, "validation_macro_MAPE": float(r.validation_macro_MAPE), "source": "DE_search"})
        for seed in extra_seeds:
            _, _, _, _, metric, seconds = train_candidate(
                train, val, cand, seed=seed, teacher_epochs=80, teacher_patience=18,
                rollout_epochs=15, rollout_patience=5,
            )
            rows.append({**asdict(cand), "seed": seed, "validation_macro_MAPE": metric, "seconds": seconds, "source": "rescreen"})
            print("rescreen", cand, seed, f"val={metric:.4f}%")
    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT / "de_top5_stochastic_rescreen.csv", index=False)
    summary = raw.groupby(["L", "hidden", "layers", "H", "lr"], as_index=False).agg(
        mean_validation_macro_MAPE=("validation_macro_MAPE", "mean"),
        std_validation_macro_MAPE=("validation_macro_MAPE", "std"),
        seeds=("seed", "nunique"),
    ).sort_values("mean_validation_macro_MAPE")
    summary.to_csv(OUTPUT / "de_top5_stochastic_rescreen_summary.csv", index=False)
    best = summary.iloc[0]
    selected = Candidate(int(best.L), int(best.hidden), int(best.layers), int(best.H), float(best.lr))
    payload = {
        "selection_rule": f"lowest mean validation rollout macro MAPE across seed 2026 plus {list(extra_seeds)}",
        "selected_configuration": asdict(selected),
        "mean_validation_macro_MAPE": float(best.mean_validation_macro_MAPE),
        "std_validation_macro_MAPE": float(best.std_validation_macro_MAPE),
    }
    (OUTPUT / "de_rescreen_selected_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    # If the robust selection is new, perform the same five-seed held-out run.
    existing = pd.read_csv(OUTPUT / "de_final_manual_vs_optimized_seed_metrics.csv")
    known = {
        Candidate(**MANUAL).key: "manual",
        Candidate(**json.loads((OUTPUT / "de_search_cost.json").read_text())["optimized_configuration"]).key: "DE_optimized",
    }
    if selected.key in known:
        chosen = existing[existing["configuration"].eq(known[selected.key])].copy()
    else:
        final_rows = []
        for seed in SEEDS:
            model, sc_x, sc_y, _, val_metric, seconds = train_candidate(
                train, val, selected, seed=seed, teacher_epochs=220, teacher_patience=40,
                rollout_epochs=40, rollout_patience=15,
            )
            yt, yp, meta, infer_seconds = rollout_predict(model, sc_x, sc_y, test, FULL9, selected.L)
            final_rows.append({
                "configuration": "DE_rescreened", "seed": seed, **asdict(selected),
                "validation_macro_MAPE": val_metric, **metrics_2d(yt, yp),
                "train_seconds": seconds,
                "inference_ms_per_trajectory": infer_seconds / meta["cell_id"].nunique() * 1000,
            })
            print("DE_rescreened", seed, f"test={final_rows[-1]['macro_MAPE']:.4f}%")
        chosen = pd.DataFrame(final_rows)
    chosen.to_csv(OUTPUT / "de_rescreened_final_seed_metrics.csv", index=False)
    chosen.groupby("configuration", as_index=False).agg(
        mean_macro_MAPE=("macro_MAPE", "mean"), std_macro_MAPE=("macro_MAPE", "std"),
        mean_Q_MAPE=("Q_MAPE", "mean"), std_Q_MAPE=("Q_MAPE", "std"),
        mean_Re_MAPE=("Re_MAPE", "mean"), std_Re_MAPE=("Re_MAPE", "std"),
    ).to_csv(OUTPUT / "de_rescreened_final_summary.csv", index=False)


if __name__ == "__main__":
    run_de()
