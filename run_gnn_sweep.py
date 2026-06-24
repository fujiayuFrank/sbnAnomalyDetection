from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ============================================================
# Default Paths
# ============================================================

PROJECT_DIR = Path("/exp/sbnd/app/users/jiayufu/sbnAnomalyDetection")
DEFAULT_CONFIG_DIR = PROJECT_DIR / "tunning_configs"

# Store trained models under the usual checkpoint area.
DEFAULT_MODEL_ROOT = PROJECT_DIR / "checkpoints" / "gnn"

# Store each sweep run/model under checkpoints/gnn/<run_name>/
DEFAULT_RUNS_ROOT = DEFAULT_MODEL_ROOT

# Store database/export summaries directly in sbnAnomalyDetection path.
DEFAULT_DB_PATH = PROJECT_DIR / "gnn_sweep.sqlite3"

# ============================================================
# Good / bad classification
# Same as C++ ROOT plotter
# ============================================================

GOOD_RUNS = {18445, 19724, 20141, 20142, 20144}
BAD_RUNS = {19627, 19946, 20104}

# ============================================================
# Small helpers
# ============================================================

def now_str() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def timestamp_for_name() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML config did not load as a dict: {path}")
    return cfg


def save_yaml(path: Path, cfg: dict[str, Any]) -> None:
    with path.open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(flatten_dict(v, key))
        else:
            flat[key] = v
    return flat


def jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def value_to_str(v: Any) -> str:
    try:
        return json.dumps(v, default=jsonable)
    except TypeError:
        return str(v)


# ============================================================
# Database
# ============================================================

def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT UNIQUE,
            config_name TEXT,
            original_config_path TEXT,
            run_config_path TEXT,
            run_dir TEXT,
            checkpoint_dir TEXT,
            final_model_path TEXT,
            score_npz_path TEXT,
            status TEXT,
            start_time TEXT,
            end_time TEXT,
            duration_sec REAL,
            train_cmd TEXT,
            infer_cmd TEXT,
            train_returncode INTEGER,
            infer_returncode INTEGER,
            train_log_path TEXT,
            infer_log_path TEXT,
            error TEXT,
            config_json TEXT,
            metrics_json TEXT
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS params (
            experiment_id INTEGER,
            key TEXT,
            value TEXT,
            PRIMARY KEY (experiment_id, key),
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            experiment_id INTEGER,
            key TEXT,
            value REAL,
            value_text TEXT,
            PRIMARY KEY (experiment_id, key),
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );
        """
    )

    conn.commit()
    return conn


def insert_experiment(
    conn: sqlite3.Connection,
    *,
    run_name: str,
    config_name: str,
    original_config_path: Path,
    run_config_path: Path,
    run_dir: Path,
    checkpoint_dir: Path,
    final_model_path: Path,
    train_cmd: list[str],
    infer_cmd: list[str],
    train_log_path: Path | None,
    infer_log_path: Path | None,
    patched_cfg: dict[str, Any],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO experiments (
            run_name,
            config_name,
            original_config_path,
            run_config_path,
            run_dir,
            checkpoint_dir,
            final_model_path,
            status,
            start_time,
            train_cmd,
            infer_cmd,
            train_log_path,
            infer_log_path,
            config_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_name,
            config_name,
            str(original_config_path),
            str(run_config_path),
            str(run_dir),
            str(checkpoint_dir),
            str(final_model_path),
            "running",
            now_str(),
            " ".join(shlex.quote(x) for x in train_cmd),
            " ".join(shlex.quote(x) for x in infer_cmd),
            str(train_log_path) if train_log_path else None,
            str(infer_log_path) if infer_log_path else None,
            json.dumps(patched_cfg, default=jsonable),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_params(
    conn: sqlite3.Connection,
    experiment_id: int,
    cfg: dict[str, Any],
) -> None:
    flat = flatten_dict(cfg)
    rows = [(experiment_id, k, value_to_str(v)) for k, v in sorted(flat.items())]
    conn.executemany(
        """
        INSERT OR REPLACE INTO params (experiment_id, key, value)
        VALUES (?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def insert_metrics(
    conn: sqlite3.Connection,
    experiment_id: int,
    metrics: dict[str, Any],
) -> None:
    rows = []
    for k, v in sorted(metrics.items()):
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)):
            rows.append((experiment_id, k, float(v), None))
        else:
            rows.append((experiment_id, k, None, value_to_str(v)))

    conn.executemany(
        """
        INSERT OR REPLACE INTO metrics (experiment_id, key, value, value_text)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def update_experiment_done(
    conn: sqlite3.Connection,
    experiment_id: int,
    *,
    status: str,
    duration_sec: float,
    train_returncode: int | None,
    infer_returncode: int | None,
    score_npz_path: Path | None,
    error: str | None,
    metrics: dict[str, Any],
) -> None:
    conn.execute(
        """
        UPDATE experiments
        SET
            status = ?,
            end_time = ?,
            duration_sec = ?,
            train_returncode = ?,
            infer_returncode = ?,
            score_npz_path = ?,
            error = ?,
            metrics_json = ?
        WHERE id = ?
        """,
        (
            status,
            now_str(),
            duration_sec,
            train_returncode,
            infer_returncode,
            str(score_npz_path) if score_npz_path else None,
            error,
            json.dumps(metrics, default=jsonable),
            experiment_id,
        ),
    )
    conn.commit()


# ============================================================
# Config patching
# ============================================================

def prepare_run_config(
    original_cfg: dict[str, Any],
    run_dir: Path,
) -> tuple[dict[str, Any], Path, Path, Path]:
    """
    Make a per-run config so different YAML jobs do not overwrite each other.

    Returns
    -------
    patched_cfg, checkpoint_dir, final_model_path, expected_score_npz_path
    """

    cfg = copy.deepcopy(original_cfg)

    training = cfg.setdefault("training", {})
    inference = cfg.setdefault("inference", {})

    checkpoint_dir = run_dir
    final_model_path = checkpoint_dir / "gnn_final.pt"
    expected_score_npz_path = checkpoint_dir / "scores.npz"

    training["checkpoint_dir"] = str(checkpoint_dir)
    training["output_path"] = str(final_model_path)

    # Keep inference input_path / threshold / max_windows from the YAML.
    # Patch checkpoint path so inference evaluates the model just trained.
    inference["checkpoint_path"] = str(final_model_path)

    # This assumes your sbn-infer reads inference.output_path.
    # If your infer CLI uses another key, add that key here too.
    inference.setdefault("output_path", str(expected_score_npz_path))

    return cfg, checkpoint_dir, final_model_path, expected_score_npz_path


# ============================================================
# Subprocess execution
# ============================================================

def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int | None = None,
) -> int:
    """Run command directly in terminal without capturing output."""
    header = (
        f"$ {' '.join(shlex.quote(x) for x in cmd)}\n"
        f"cwd={cwd}\n"
        + "=" * 80
        + "\n"
    )
    print(header, end="", flush=True)

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            timeout=timeout,
        )
        returncode = completed.returncode

    except subprocess.TimeoutExpired:
        print(f"\nCommand timed out after {timeout} seconds.")
        raise

    footer = "\n" + "=" * 80 + "\n" + f"returncode={returncode}\n"
    print(footer, end="", flush=True)

    return int(returncode)


# ============================================================
# Result parsing
# ============================================================

def read_last_training_history_row(history_csv: Path) -> dict[str, Any]:
    if not history_csv.exists():
        return {}

    with history_csv.open("r", newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        return {}

    last = rows[-1]
    metrics: dict[str, Any] = {}

    for key, value in last.items():
        if value is None or value == "":
            continue
        try:
            metrics[f"train_history.final_{key}"] = float(value)
        except ValueError:
            metrics[f"train_history.final_{key}"] = value

    # Useful best values too
    for wanted in ["loss", "val_loss", "score_p95", "score_p99"]:
        vals = []
        for row in rows:
            if wanted in row and row[wanted] not in ("", None):
                try:
                    vals.append(float(row[wanted]))
                except ValueError:
                    pass
        if vals:
            metrics[f"train_history.best_{wanted}"] = min(vals)

    return metrics


def summarize_array(prefix: str, arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return {
            f"{prefix}.n": 0,
        }

    return {
        f"{prefix}.n": int(arr.size),
        f"{prefix}.mean": float(np.mean(arr)),
        f"{prefix}.std": float(np.std(arr)),
        f"{prefix}.min": float(np.min(arr)),
        f"{prefix}.median": float(np.median(arr)),
        f"{prefix}.p90": float(np.percentile(arr, 90)),
        f"{prefix}.p95": float(np.percentile(arr, 95)),
        f"{prefix}.p99": float(np.percentile(arr, 99)),
        f"{prefix}.max": float(np.max(arr)),
    }


def summarize_scores_npz(score_npz_path: Path) -> dict[str, Any]:
    if not score_npz_path.exists():
        return {}

    metrics: dict[str, Any] = {}
    data = np.load(score_npz_path, allow_pickle=True)

    metrics["score_npz.exists"] = 1
    metrics["score_npz.path"] = str(score_npz_path)

    if "scores" in data:
        scores = np.asarray(data["scores"])
        metrics.update(summarize_array("scores", scores))

    if "scores_max" in data:
        scores_max = np.asarray(data["scores_max"])
        metrics.update(summarize_array("scores_max", scores_max))

    if "first_run" in data:
        first_run = np.asarray(data["first_run"])
        finite_run_mask = np.isfinite(first_run)
        runs = first_run[finite_run_mask].astype(int)

        unique_runs = sorted(np.unique(runs))
        metrics["runs.n_unique"] = int(len(unique_runs))
        metrics["runs.list"] = ",".join(str(r) for r in unique_runs)

        if "scores" in data:
            scores = np.asarray(data["scores"], dtype=np.float64)
            if scores.shape == first_run.shape:
                for group_name, group_runs in [
                    ("good", GOOD_RUNS),
                    ("bad", BAD_RUNS),
                ]:
                    mask = np.isin(runs, list(group_runs))
                    scores_valid = scores[finite_run_mask][mask]
                    metrics.update(summarize_array(f"scores.{group_name}", scores_valid))

                good_mask = np.isin(runs, list(GOOD_RUNS))
                bad_mask = np.isin(runs, list(BAD_RUNS))
                good_scores = scores[finite_run_mask][good_mask]
                bad_scores = scores[finite_run_mask][bad_mask]
                if good_scores.size > 0 and bad_scores.size > 0:
                    metrics["scores.bad_minus_good.mean"] = (
                        float(np.mean(bad_scores)) - float(np.mean(good_scores))
                    )
                    metrics["scores.bad_minus_good.p95"] = (
                        float(np.percentile(bad_scores, 95))
                        - float(np.percentile(good_scores, 95))
                    )

        if "scores_max" in data:
            scores_max = np.asarray(data["scores_max"], dtype=np.float64)
            if scores_max.shape == first_run.shape:
                for group_name, group_runs in [
                    ("good", GOOD_RUNS),
                    ("bad", BAD_RUNS),
                ]:
                    mask = np.isin(runs, list(group_runs))
                    scores_valid = scores_max[finite_run_mask][mask]
                    metrics.update(summarize_array(f"scores_max.{group_name}", scores_valid))

                good_mask = np.isin(runs, list(GOOD_RUNS))
                bad_mask = np.isin(runs, list(BAD_RUNS))
                good_scores = scores_max[finite_run_mask][good_mask]
                bad_scores = scores_max[finite_run_mask][bad_mask]
                if good_scores.size > 0 and bad_scores.size > 0:
                    metrics["scores_max.bad_minus_good.mean"] = (
                        float(np.mean(bad_scores)) - float(np.mean(good_scores))
                    )
                    metrics["scores_max.bad_minus_good.p95"] = (
                        float(np.percentile(bad_scores, 95))
                        - float(np.percentile(good_scores, 95))
                    )

    return metrics


def find_score_npz(
    expected_score_npz_path: Path,
    run_dir: Path,
) -> Path | None:
    if expected_score_npz_path.exists():
        return expected_score_npz_path

    candidates = list(run_dir.rglob("*scores*.npz"))
    if not candidates:
        candidates = list(run_dir.rglob("*.npz"))

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def collect_run_metrics(
    checkpoint_dir: Path,
    expected_score_npz_path: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], Path | None]:
    metrics: dict[str, Any] = {}

    history_csv = checkpoint_dir / "training_history.csv"
    metrics.update(read_last_training_history_row(history_csv))

    score_npz_path = find_score_npz(expected_score_npz_path, run_dir)
    if score_npz_path is not None:
        metrics.update(summarize_scores_npz(score_npz_path))
    else:
        metrics["score_npz.exists"] = 0

    return metrics, score_npz_path


# ============================================================
# Export
# ============================================================

def export_summary(conn: sqlite3.Connection, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "gnn_sweep_summary.csv"
    xlsx_path = output_dir / "gnn_sweep_summary.xlsx"

    query = """
    SELECT
        e.id,
        e.run_name,
        e.config_name,
        e.status,
        e.start_time,
        e.end_time,
        e.duration_sec,
        e.run_dir,
        e.final_model_path,
        e.score_npz_path,
        e.error,
        m.key AS metric_key,
        COALESCE(CAST(m.value AS TEXT), m.value_text) AS metric_value
    FROM experiments e
    LEFT JOIN metrics m ON e.id = m.experiment_id
    ORDER BY e.id, m.key
    """

    rows = conn.execute(query).fetchall()
    cols = [desc[0] for desc in conn.execute(query).description]

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        writer.writerows(rows)

    try:
        import pandas as pd

        experiments = pd.read_sql_query("SELECT * FROM experiments ORDER BY id", conn)
        params = pd.read_sql_query("SELECT * FROM params ORDER BY experiment_id, key", conn)
        metrics = pd.read_sql_query("SELECT * FROM metrics ORDER BY experiment_id, key", conn)

        with pd.ExcelWriter(xlsx_path) as writer:
            experiments.to_excel(writer, sheet_name="experiments", index=False)
            params.to_excel(writer, sheet_name="params", index=False)
            metrics.to_excel(writer, sheet_name="metrics", index=False)

        print(f"Exported Excel summary: {xlsx_path}")

    except Exception as exc:
        print(f"WARNING: could not export Excel summary: {exc}")

    print(f"Exported CSV summary: {csv_path}")


# ============================================================
# Main sweep
# ============================================================

def run_sweep(args: argparse.Namespace) -> int:
    config_dir = Path(args.config_dir)
    runs_root = Path(args.runs_root)
    db_path = Path(args.db_path)

    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory does not exist: {config_dir}")

    config_paths = sorted(config_dir.glob(args.pattern))
    if not config_paths:
        raise FileNotFoundError(f"No configs matching {args.pattern!r} found in {config_dir}")

    runs_root.mkdir(parents=True, exist_ok=True)

    train_base_cmd = shlex.split(args.train_cmd)
    infer_base_cmd = shlex.split(args.infer_cmd)

    conn = init_db(db_path)

    print("=" * 80)
    print(f"Config directory: {config_dir}")
    print(f"Number of configs: {len(config_paths)}")
    print(f"Runs root: {runs_root}")
    print(f"Database: {db_path}")
    print("=" * 80)

    for idx, config_path in enumerate(config_paths, start=1):
        t0 = time.perf_counter()

        config_stem = safe_name(config_path.stem)
        run_name = f"{config_stem}"
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        run_config_path = run_dir / "config_run.yaml"
        original_copy_path = run_dir / "config_original.yaml"
        # train_log_path = run_dir / "train.log"
        # infer_log_path = run_dir / "infer.log"

        print("\n" + "=" * 80)
        print(f"[{idx}/{len(config_paths)}] {config_path}")
        print(f"Run directory: {run_dir}")
        print("=" * 80)

        train_returncode: int | None = None
        infer_returncode: int | None = None
        experiment_id: int | None = None
        score_npz_path: Path | None = None
        metrics: dict[str, Any] = {}
        error: str | None = None
        status = "failed"

        try:
            original_cfg = load_yaml(config_path)
            patched_cfg, checkpoint_dir, final_model_path, expected_score_npz_path = prepare_run_config(
                original_cfg,
                run_dir,
            )

            # save_yaml(original_copy_path, original_cfg)
            save_yaml(run_config_path, patched_cfg)

            train_cmd = train_base_cmd + ["--config", str(run_config_path)]
            infer_cmd = infer_base_cmd + ["--config", str(run_config_path)]

            experiment_id = insert_experiment(
                conn,
                run_name=run_name,
                config_name=config_path.name,
                original_config_path=config_path,
                run_config_path=run_config_path,
                run_dir=run_dir,
                checkpoint_dir=checkpoint_dir,
                final_model_path=final_model_path,
                train_cmd=train_cmd,
                infer_cmd=infer_cmd,
                train_log_path=None,
                infer_log_path=None,
                patched_cfg=patched_cfg,
            )
            insert_params(conn, experiment_id, patched_cfg)

            print("Training...")
            train_returncode = run_command(
                train_cmd,
                cwd=PROJECT_DIR,
                timeout=args.timeout,
            )
            if train_returncode != 0:
                raise RuntimeError(f"Training failed with return code {train_returncode}")

            if args.skip_infer:
                print("Skipping inference because --skip-infer was set.")
                status = "trained_no_infer"
            else:
                print("Inference...")
                infer_returncode = run_command(
                    infer_cmd,
                    cwd=PROJECT_DIR,
                    timeout=args.timeout,
                )
                if infer_returncode != 0:
                    raise RuntimeError(f"Inference failed with return code {infer_returncode}")

                status = "success"

            metrics, score_npz_path = collect_run_metrics(
                checkpoint_dir=checkpoint_dir,
                expected_score_npz_path=expected_score_npz_path,
                run_dir=run_dir,
            )

            insert_metrics(conn, experiment_id, metrics)

        except Exception as exc:
            error = repr(exc)
            print(f"ERROR: {error}")

        finally:
            duration_sec = time.perf_counter() - t0

            if experiment_id is not None:
                update_experiment_done(
                    conn,
                    experiment_id,
                    status=status,
                    duration_sec=duration_sec,
                    train_returncode=train_returncode,
                    infer_returncode=infer_returncode,
                    score_npz_path=score_npz_path,
                    error=error,
                    metrics=metrics,
                )

            print(f"Status: {status}")
            print(f"Duration: {duration_sec:.1f} s")

            if error and args.stop_on_error:
                print("Stopping because --stop-on-error was set.")
                break

    export_summary(conn, db_path.parent)
    conn.close()

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a GNN YAML sweep: train, infer, and record results in SQLite."
    )

    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory containing YAML configs.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Directory where per-run outputs will be written.",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--pattern",
        default="*.yaml",
        help="Glob pattern for config files.",
    )
    parser.add_argument(
        "--train-cmd",
        default="sbn-train",
        help='Training command, e.g. "sbn-train" or "python -m sbn_anomaly.train.cli".',
    )
    parser.add_argument(
        "--infer-cmd",
        default="sbn-infer",
        help='Inference command, e.g. "sbn-infer" or "python -m sbn_anomaly.infer.cli".',
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds for each train/infer subprocess.",
    )
    parser.add_argument(
        "--skip-infer",
        action="store_true",
        help="Only train models; do not run inference.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the sweep after the first failed config.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return run_sweep(args)


if __name__ == "__main__":
    sys.exit(main())