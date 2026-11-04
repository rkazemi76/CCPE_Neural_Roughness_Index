#!/usr/bin/env python3
"""Validate packaged expanding-window out-of-sample forecasts and recompute metrics.

Run from the reproducibility package root:
  python code/ccpe_validate_extended_oos.py

Only pandas and NumPy are required. This script does not redownload data or refit models;
 it checks the saved forecast outputs and recomputes RMSE, MAE, correlation, and QLIKE.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

RUNS = [
    ("initial_train_30", 30, Path("results/v2_23_garch"), "v223_paired_oos_predictions.csv", "v223_oos_metrics.csv"),
    ("initial_train_25", 25, Path("results/v2_25_oos_initial_train_25"), "v223_paired_oos_predictions.csv", "v223_oos_metrics.csv"),
    ("initial_train_20", 20, Path("results/v2_25_oos_initial_train_20"), "v223_paired_oos_predictions.csv", "v223_oos_metrics.csv"),
]
MODELS = ["Naive_Current_RV", "HAR_RV", "GARCH11_RV", "HAR_RV_plus_Neural_H", "HAR_RV_plus_Variogram_H"]

def qlike(actual, forecast):
    y = np.maximum(np.asarray(actual, dtype=float), 1e-18)
    p = np.maximum(np.asarray(forecast, dtype=float), 1e-18)
    r = y / p
    return float(np.mean(r - np.log(r) - 1.0))

def main():
    root = Path.cwd()
    all_rows, reports = [], []
    ts_sets = {}
    for name, initial, rel_dir, pred_file, metrics_file in RUNS:
        run_dir = root / rel_dir
        pred_path = run_dir / pred_file
        metrics_path = run_dir / metrics_file
        if not pred_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"Missing forecast or metrics file for {name}: {pred_path}")
        df = pd.read_csv(pred_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        timestamps = sorted(df["timestamp"].unique())
        ts_sets[name] = set(timestamps)
        asset_counts = df.groupby("timestamp")["symbol"].nunique()
        n_oos = len(timestamps)
        n_rows = len(df)
        dupes = int(df.duplicated(["timestamp", "symbol"]).sum())
        expected_symbols = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"}
        symbol_set = set(df["symbol"].unique())
        pred_log_cols = [f"predicted_logRV__{m}" for m in MODELS]
        pred_rv_cols = [f"predicted_rv_qlike__{m}" for m in MODELS]
        missing_cols = [c for c in pred_log_cols + pred_rv_cols + ["actual_logRV", "actual_rv"] if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {pred_path}: {missing_cols}")
        missing_values = int(df[pred_log_cols + pred_rv_cols + ["actual_logRV", "actual_rv"]].isna().sum().sum())
        if dupes != 0 or missing_values != 0 or not (asset_counts.eq(5).all()) or symbol_set != expected_symbols:
            raise ValueError(f"Forecast integrity failed for {name}: dupes={dupes}, missing={missing_values}, asset_counts={asset_counts.to_dict()}, symbols={symbol_set}")
        actual_log = df["actual_logRV"].to_numpy(float)
        actual_rv = df["actual_rv"].to_numpy(float)
        for model in MODELS:
            pred_log = df[f"predicted_logRV__{model}"].to_numpy(float)
            rmse = float(np.sqrt(np.mean((actual_log - pred_log) ** 2)))
            mae = float(np.mean(np.abs(actual_log - pred_log)))
            corr = float(np.corrcoef(actual_log, pred_log)[0, 1])
            q = qlike(actual_rv, df[f"predicted_rv_qlike__{model}"].to_numpy(float))
            all_rows.append({
                "configuration": name,
                "initial_train_timestamps": initial,
                "oos_timestamps": n_oos,
                "n_asset_date_forecasts": n_rows,
                "first_oos_timestamp_utc": pd.Timestamp(timestamps[0]).isoformat(),
                "last_oos_timestamp_utc": pd.Timestamp(timestamps[-1]).isoformat(),
                "model": model,
                "RMSE_logRV_recomputed": rmse,
                "MAE_logRV_recomputed": mae,
                "QLIKE_RV_recomputed": q,
                "Corr_logRV_recomputed": corr,
            })
        saved = pd.read_csv(metrics_path)
        metrics_diffs = {}
        for model in MODELS:
            row = saved.loc[saved["model"].eq(model)]
            if len(row) != 1:
                raise ValueError(f"Expected exactly one saved metric row for {model} in {metrics_path}")
            calc = next(r for r in all_rows if r["configuration"] == name and r["model"] == model)
            diff = {
                "RMSE_abs_diff": abs(calc["RMSE_logRV_recomputed"] - float(row.iloc[0]["RMSE_logRV"])),
                "MAE_abs_diff": abs(calc["MAE_logRV_recomputed"] - float(row.iloc[0]["MAE_logRV"])),
                "QLIKE_abs_diff": abs(calc["QLIKE_RV_recomputed"] - float(row.iloc[0]["QLIKE_RV"])),
                "Corr_abs_diff": abs(calc["Corr_logRV_recomputed"] - float(row.iloc[0]["Corr_logRV"])),
            }
            metrics_diffs[model] = diff
            if max(diff.values()) > 2e-5:
                raise ValueError(f"Metric reproduction mismatch for {name}/{model}: {diff}")
        reports.append({
            "configuration": name,
            "initial_train_timestamps": initial,
            "oos_timestamps": n_oos,
            "n_asset_date_forecasts": n_rows,
            "first_oos_timestamp_utc": pd.Timestamp(timestamps[0]).isoformat(),
            "last_oos_timestamp_utc": pd.Timestamp(timestamps[-1]).isoformat(),
            "unique_assets_per_timestamp": sorted(asset_counts.unique().tolist()),
            "duplicate_timestamp_symbol_rows": dupes,
            "missing_prediction_or_target_values": missing_values,
            "saved_metrics_recomputed_within_tolerance": True,
            "max_abs_metric_reproduction_error": max(v for vv in metrics_diffs.values() for v in vv.values()),
            "garch_fit_status_counts": pd.read_csv(run_dir / "v223_garch_fit_status.csv").to_dict("records"),
        })
    # Confirm test windows are nested, and do not call them independent replicates.
    sets = [ts_sets["initial_train_30"], ts_sets["initial_train_25"], ts_sets["initial_train_20"]]
    nested = sets[0].issubset(sets[1]) and sets[1].issubset(sets[2])
    if not nested:
        raise ValueError("Expected OOS timestamp windows to be nested")
    report = {
        "validation": "PASS",
        "method": "recomputed metrics from saved aligned predictions; no refitting and no raw-data download",
        "windows_are_nested_not_independent": nested,
        "all_windows_share_last_timestamp": len({r["last_oos_timestamp_utc"] for r in reports}) == 1,
        "runs": reports,
    }
    outdir = root / "results/ccpe_extended_oos_validation"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(outdir / "forecast_metrics_recomputed.csv", index=False)
    (outdir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {outdir}")

if __name__ == "__main__":
    main()
