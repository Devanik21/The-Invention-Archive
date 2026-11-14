"""Run a reproducible small-data comparison of ALIEN-DL and MLP baselines."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from alien_dl import AlienConfig, AlienNet, MLP, parameter_matched_width


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataset(name: str):
    if name == "wine":
        data = load_wine()
    elif name == "breast_cancer":
        data = load_breast_cancer()
    elif name == "digits":
        data = load_digits()
    else:
        raise ValueError(name)
    return data.data.astype(np.float32), data.target.astype(np.int64), data.target_names


def make_split(name: str, seed: int):
    X, y, _ = get_dataset(name)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed + 1000, stratify=y_temp
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    n_classes = len(np.unique(y))
    return (
        torch.tensor(X_train), torch.tensor(y_train),
        torch.tensor(X_val), torch.tensor(y_val),
        torch.tensor(X_test), torch.tensor(y_test),
        X.shape[1], n_classes,
    )


def batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool):
    if shuffle:
        idx = torch.randperm(len(X))
    else:
        idx = torch.arange(len(X))
    for start in range(0, len(X), batch_size):
        take = idx[start:start + batch_size]
        yield X[take], y[take]


def predict(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            outputs.append(model(X[start:start + 256]).argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def train_model(model: nn.Module, kind: str, Xtr: torch.Tensor, ytr: torch.Tensor,
                Xv: torch.Tensor, yv: torch.Tensor, seed: int,
                max_epochs: int = 100, batch_size: int = 32):
    # Keep the run small and deterministic on CPU.
    criterion = nn.CrossEntropyLoss()
    if kind == "alien":
        assert isinstance(model, AlienNet)
        # The ordinary optimizer excludes theta; theta receives its explicit
        # allocation-gated SGD update. All other trainable parameters use Adam.
        other_params = [p for n, p in model.named_parameters()
                        if n not in ("theta", "alloc_logits") and p.requires_grad]
        optimizer = torch.optim.AdamW(other_params, lr=0.012, weight_decay=1e-4)
        best_state = None
        best_val = -1.0
        best_epoch = 0
        bad_epochs = 0
        start = time.perf_counter()
        for epoch in range(max_epochs):
            model.train()
            epoch_utility = torch.zeros_like(model.alloc_logits)
            n_utility_batches = 0
            for xb, yb in batches(Xtr, ytr, batch_size, shuffle=True):
                optimizer.zero_grad(set_to_none=True)
                if model.theta.grad is not None:
                    model.theta.grad = None
                logits = model(xb)
                loss = criterion(logits, yb)
                # A per-synapse first-order contribution signal drives allocation
                # even when a particular coupling pathway is ablated.
                grad_q = torch.autograd.grad(loss, model._last_q, retain_graph=True)[0]
                utility = -(model._last_q.detach() * grad_q.detach()).mean(dim=0)
                epoch_utility.add_(utility)
                n_utility_batches += 1
                loss.backward()

                grad_theta = model.theta.grad.detach().clone()
                model.synaptic_parameter_step(grad_theta)
                optimizer.step()
                model.alloc_logits.grad = None

            if n_utility_batches:
                model.allocation_step(epoch_utility / n_utility_batches)
            model.structural_epoch_step()
            val_pred = predict(model, Xv)
            val_acc = accuracy_score(yv.numpy(), val_pred)
            if val_acc > best_val + 1e-6:
                best_val = val_acc
                best_epoch = epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= 18:
                break
        elapsed = time.perf_counter() - start
        if best_state is not None:
            model.load_state_dict(best_state)
        return best_epoch, elapsed

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012, weight_decay=1e-4)
    best_state = None
    best_val = -1.0
    best_epoch = 0
    bad_epochs = 0
    start = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in batches(Xtr, ytr, batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_pred = predict(model, Xv)
        val_acc = accuracy_score(yv.numpy(), val_pred)
        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= 18:
            break
    elapsed = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, elapsed


def build_model(kind: str, n_features: int, n_classes: int):
    if kind == "mlp_fixed":
        return MLP(n_features, n_classes, width=32)
    if kind == "mlp_matched":
        # Use the full ALIEN parameter count as the matching target.
        probe = AlienNet(n_features, n_classes)
        width = parameter_matched_width(n_features, n_classes,
                                        sum(p.numel() for p in probe.parameters() if p.requires_grad))
        return MLP(n_features, n_classes, width=width)
    if kind == "alien":
        return AlienNet(n_features, n_classes, AlienConfig())
    raise ValueError(kind)


def run_main(datasets, seeds, outdir: Path, epochs: int):
    records = []
    for dataset in datasets:
        for seed in seeds:
            set_seed(seed)
            Xtr, ytr, Xv, yv, Xte, yte, nf, nc = make_split(dataset, seed)
            for kind in ("mlp_fixed", "mlp_matched", "alien"):
                set_seed(seed + {"mlp_fixed": 11, "mlp_matched": 23, "alien": 37}[kind])
                model = build_model(kind, nf, nc)
                nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
                best_epoch, elapsed = train_model(model, kind, Xtr, ytr, Xv, yv,
                                                   seed, max_epochs=epochs)
                pred = predict(model, Xte)
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "model": kind,
                    "accuracy": accuracy_score(yte.numpy(), pred),
                    "macro_f1": f1_score(yte.numpy(), pred, average="macro"),
                    "parameters": nparams,
                    "best_epoch": best_epoch,
                    "train_seconds": elapsed,
                }
                if kind == "alien":
                    row.update(model.diagnostics())
                records.append(row)
                print(f"{dataset:14s} seed={seed:3d} {kind:12s} "
                      f"acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f} "
                      f"params={nparams:5d} time={elapsed:.2f}s", flush=True)
    df = pd.DataFrame(records)
    df.to_csv(outdir / "benchmark_results.csv", index=False)
    summary = (df.groupby(["dataset", "model"])
                 .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
                      macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                      parameters_mean=("parameters", "mean"), seconds_mean=("train_seconds", "mean"))
                 .reset_index())
    summary.to_csv(outdir / "benchmark_summary.csv", index=False)
    return df, summary


def run_factorial(dataset: str, seeds, outdir: Path, epochs: int):
    records = []
    configs = []
    for r in (0, 1):
        for p in (0, 1):
            for a in (0, 1):
                configs.append((r, p, a))
    for seed in seeds:
        Xtr, ytr, Xv, yv, Xte, yte, nf, nc = make_split(dataset, seed)
        for r, p, a in configs:
            set_seed(seed + 701)
            cfg = AlienConfig(r_coupling=bool(r), p_coupling=bool(p), a_coupling=bool(a))
            model = AlienNet(nf, nc, cfg)
            nparams = sum(x.numel() for x in model.parameters() if x.requires_grad)
            best_epoch, elapsed = train_model(model, "alien", Xtr, ytr, Xv, yv,
                                               seed, max_epochs=epochs)
            pred = predict(model, Xte)
            records.append({
                "dataset": dataset, "seed": seed, "R": r, "P": p, "A": a,
                "accuracy": accuracy_score(yte.numpy(), pred),
                "macro_f1": f1_score(yte.numpy(), pred, average="macro"),
                "parameters": nparams, "best_epoch": best_epoch,
                "train_seconds": elapsed,
                **model.diagnostics(),
            })
            print(f"factorial {dataset:10s} seed={seed} RPA={r}{p}{a} "
                  f"acc={records[-1]['accuracy']:.4f}", flush=True)
    df = pd.DataFrame(records)
    df.to_csv(outdir / "factorial_results.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["wine", "breast_cancer", "digits"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--factorial-seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--skip-factorial", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    outdir = Path(__file__).resolve().parent / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    df, summary = run_main(args.datasets, args.seeds, outdir, args.epochs)
    print("\nMAIN SUMMARY\n", summary.to_string(index=False), flush=True)
    if not args.skip_factorial:
        factorial = run_factorial("wine", args.factorial_seeds, outdir, args.epochs)
        fsummary = factorial.groupby(["R", "P", "A"]).agg(
            accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std")
        ).reset_index()
        fsummary.to_csv(outdir / "factorial_summary.csv", index=False)
        print("\nFACTORIAL SUMMARY\n", fsummary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
