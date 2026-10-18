#!/usr/bin/env python3
"""SAME: train on the full official TRAIN, then evaluate official TEST once.

Uses dataset-specific recipes from configs/uea18.json.
Input: official TRAIN/TEST arrays in one NPZ file per dataset.
Early stop on TRAIN cross-entropy; evaluate the last trained model on TEST.
No repartitioning, test-based selection, automatic plotting, retry, or overwrite.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROTOCOL = "official_train_ce_earlystop_final_test"
EARLY_STOP_MONITOR = "sample_weighted_online_train_ce"
CHECKPOINT_STRATEGY = "last trained epoch; do not restore an earlier checkpoint"
SEEDS = (42, 123, 777, 2024, 3407)
TRAIN_DEFAULTS = {
    "protocol": PROTOCOL, "seed": 42,
    "epochs": 100, "patience": 20, "early_stop_min_delta": 0.0,
    "batch_size": 32, "learning_rate": 0.001, "weight_decay": 0.0001,
    "num_workers": 0,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


class TrainingCEEarlyStopper:
    """Stop after patience epochs without a significant TRAIN-CE decrease.

    The monitor is the sample-weighted CE from training-mode batch forwards,
    not total loss, validation loss, or any TEST metric. Does not restore weights.
    """

    def __init__(self, patience, min_delta=0.0):
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
            raise ValueError("patience must be a positive integer")
        if not isinstance(min_delta, (int, float)) or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("early_stop_min_delta must be finite and nonnegative")
        self.patience, self.min_delta = patience, float(min_delta)
        self.reference_ce, self.reference_epoch = math.inf, 0
        self.bad_epochs, self.last_epoch, self.triggered = 0, 0, False

    def update(self, train_ce, epoch):
        if self.triggered:
            raise RuntimeError("Early stopping already triggered")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= self.last_epoch:
            raise ValueError("Early-stop epochs must be positive and strictly increasing")
        train_ce = float(train_ce)
        if not math.isfinite(train_ce) or train_ce < 0:
            raise FloatingPointError("TRAIN cross-entropy must be finite and nonnegative")
        improved = train_ce < self.reference_ce - self.min_delta
        if improved:
            self.reference_ce, self.reference_epoch, self.bad_epochs = train_ce, epoch, 0
        else:
            self.bad_epochs += 1
        self.last_epoch = epoch
        self.triggered = self.bad_epochs >= self.patience
        return improved, self.triggered

    def state(self):
        return {"monitor": EARLY_STOP_MONITOR, "patience": self.patience,
                "min_delta": self.min_delta, "reference_train_ce": self.reference_ce,
                "reference_epoch": self.reference_epoch, "non_improving_epochs": self.bad_epochs,
                "last_epoch": self.last_epoch, "triggered": self.triggered}


def official_metadata(train_labels, test_labels):
    """Audit the existing partitions; derive class coordinates from TRAIN only."""
    train_labels, test_labels = np.asarray(train_labels), np.asarray(test_labels)
    for name, labels in (("TRAIN", train_labels), ("TEST", test_labels)):
        if labels.ndim != 1 or not len(labels):
            raise ValueError(f"{name} labels must be nonempty and one-dimensional")
        if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
            raise ValueError(f"Non-finite {name} labels")
    classes = np.unique(train_labels)
    if len(classes) < 2:
        raise ValueError("Classification requires at least two TRAIN classes")
    if not np.isin(test_labels, classes).all():
        raise ValueError("TEST class is absent from official TRAIN")
    return {
        "protocol": PROTOCOL, "official_split_preserved": True,
        "train_samples": len(train_labels), "test_samples": len(test_labels),
        "validation_samples": 0, "sample_order": "original NPZ order within each partition",
        "class_counts": [{"class": label.item(), "train": int(np.sum(train_labels == label)),
                          "test": int(np.sum(test_labels == label))} for label in classes],
    }


def find_npz(data_root, dataset):
    candidates = (data_root / f"{dataset}.npz",
                  data_root / dataset / f"{dataset}.npz",
                  data_root / dataset / "data.npz")
    matches = [path.resolve() for path in candidates if path.is_file()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous data for {dataset}: {matches}")
    if not matches:
        raise FileNotFoundError(f"No official TRAIN/TEST NPZ for {dataset}; tried {candidates}")
    return matches[0]


def resolve_recipe(recipes, dataset, seed, args):
    defaults = recipes.get("defaults", {})
    profile = recipes["datasets"][dataset]
    model = {**defaults.get("model", {}), **profile.get("model", {})}
    training = {**TRAIN_DEFAULTS, **defaults.get("training", {}),
                **profile.get("training", {})}
    unknown = set(training) - set(TRAIN_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown training keys for {dataset}: {sorted(unknown)}")
    for name in ("epochs", "patience", "early_stop_min_delta", "num_workers"):
        value = getattr(args, name, None)
        if value is not None:
            training[name] = value
    if training["protocol"] != PROTOCOL:
        raise ValueError("Only full TRAIN with final-only TEST evaluation is supported")
    training.update(seed=seed)
    for name in ("epochs", "patience", "batch_size"):
        if not isinstance(training[name], int) or isinstance(training[name], bool) or training[name] < 1:
            raise ValueError(f"{dataset}: {name} must be a positive integer")
    if not isinstance(training["num_workers"], int) or training["num_workers"] < 0:
        raise ValueError(f"{dataset}: num_workers must be a nonnegative integer")
    for name in ("learning_rate", "weight_decay"):
        value = training[name]
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{dataset}: {name} must be finite")
        if value < 0 or (name == "learning_rate" and value == 0):
            raise ValueError(f"{dataset}: invalid {name}")
    TrainingCEEarlyStopper(training["patience"], training["early_stop_min_delta"])
    if model.get("architecture_version") != "ordered_multiscale_v2":
        raise ValueError(f"{dataset}: expected ordered_multiscale_v2 config")
    return model, training


def load_arrays(path):
    """Keep official TRAIN/TEST separate, preserving sample order and masks."""
    with np.load(path, allow_pickle=False) as data:
        missing = {"train_x", "train_y", "test_x", "test_y"} - set(data.files)
        if missing:
            raise ValueError(f"Missing NPZ arrays: {sorted(missing)}")
        arrays = {}
        for prefix in ("train", "test"):
            x = np.array(data[f"{prefix}_x"], dtype=np.float32, copy=True)
            y = np.asarray(data[f"{prefix}_y"])
            if (x.ndim != 3 or y.ndim != 1 or len(x) != len(y) or not len(y)
                    or x.shape[1] < 1 or x.shape[2] < 1):
                raise ValueError(f"Invalid {prefix} arrays: require x[N,T,C] and y[N]")
            if y.dtype.kind in "fc" and not np.isfinite(y).all():
                raise ValueError(f"Non-finite {prefix} labels")
            key = f"{prefix}_mask"
            mask = np.asarray(data[key], dtype=bool) if key in data else np.ones(x.shape[:2], bool)
            if mask.shape != x.shape[:2] or not mask.any(axis=1).all():
                raise ValueError(f"Invalid {prefix} masks")
            if not np.isfinite(x[mask]).all():
                raise ValueError(f"Non-finite valid {prefix} observations")
            x[~mask] = 0
            arrays[prefix] = (x, y, mask)
    train_x, train_labels, train_mask = arrays["train"]
    test_x, test_labels, test_mask = arrays["test"]
    if train_x.shape[-1] != test_x.shape[-1]:
        raise ValueError("TRAIN and TEST channel counts differ")
    official_metadata(train_labels, test_labels)
    classes, train_y = np.unique(train_labels, return_inverse=True)
    label_ids = {label.item(): index for index, label in enumerate(classes)}
    test_y = np.asarray([label_ids[label.item()] for label in test_labels], dtype=np.int64)
    return train_x, train_y.astype(np.int64), train_mask, test_x, test_y, test_mask, classes


def classification_metrics(logits, labels, class_count):
    logits, labels = np.asarray(logits), np.asarray(labels)
    if (class_count < 2 or logits.ndim != 2 or logits.shape[1] != class_count
            or labels.ndim != 1 or not len(labels) or len(logits) != len(labels)
            or labels.dtype.kind not in "iu" or (labels < 0).any()
            or (labels >= class_count).any() or not np.isfinite(logits).all()):
        raise ValueError("Invalid evaluation logits")
    predictions = logits.argmax(axis=1)
    confusion = np.bincount(class_count * labels + predictions,
                            minlength=class_count ** 2).reshape(class_count, class_count)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2.0 * confusion.diagonal(), denominator,
                   out=np.zeros(class_count, dtype=np.float64), where=denominator > 0)
    return {"accuracy": float(np.mean(predictions == labels)),
            "macro_f1": float(f1.mean())}, confusion


def evaluate(model, data_loader, device, class_count):
    import torch
    model.eval()
    logits_parts, label_parts = [], []
    with torch.no_grad():
        for x, y, mask in data_loader:
            logits_parts.append(model(x.to(device), mask.to(device)).cpu().numpy())
            label_parts.append(y.numpy())
    logits, labels = np.concatenate(logits_parts), np.concatenate(label_parts)
    metrics, confusion = classification_metrics(logits, labels, class_count)
    return metrics, logits, labels, confusion


def log_record(folder, value):
    line = json.dumps(value, ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    with (folder / "train.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def train_one(dataset, seed, data_path, folder, model_recipe,
              training, device_name, data_sha256, config_sha256):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from same import SAMEConfig, SAME

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for a smoke test")
    train_x, train_y, train_mask, test_x, test_y, test_mask, classes = load_arrays(data_path)
    configuration = SAMEConfig(**{**model_recipe, "channels": train_x.shape[-1],
                                "classes": len(classes)})
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "resolved_config.json", {
        "dataset": dataset, "model": asdict(configuration), "training": training,
        "data_source": str(data_path), "data_sha256": data_sha256,
        "config_sha256": config_sha256, "seed": seed,
        "protocol": PROTOCOL,
        "checkpoint_strategy": CHECKPOINT_STRATEGY,
        "early_stopping": True, "early_stop_monitor": EARLY_STOP_MONITOR,
        "validation_used": False,
        "normalization": "sample-wise masked normalization if normalize_input=true; no fitted TEST statistics",
    })

    def make_loader(x, y, mask, shuffle):
        tensors = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask))
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(tensors, batch_size=training["batch_size"], shuffle=shuffle,
                          num_workers=training["num_workers"], pin_memory=device.type == "cuda",
                          generator=generator)

    train_loader = make_loader(train_x, train_y, train_mask, True)
    model = SAME(configuration).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"])
    stopper = TrainingCEEarlyStopper(training["patience"], training["early_stop_min_delta"])
    history = []
    for epoch in range(1, training["epochs"] + 1):
        model.train()
        model.set_training_progress((epoch - 1) / max(1, training["epochs"] - 1))
        loss_sum, sample_count, correct, components = 0.0, 0, 0, {}
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = model(x, mask, return_aux=True)
            losses = model.loss(logits, y, auxiliary)
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{dataset} seed={seed}: non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            correct += int((logits.detach().argmax(-1) == y).sum())
            loss_sum += float(loss.detach()) * len(x)
            for key, value in losses.items():
                components[key] = components.get(key, 0.0) + float(value.detach()) * len(x)
            sample_count += len(x)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": loss_sum / sample_count,
                  "train_accuracy": correct / sample_count, "learning_rate": epoch_learning_rate,
                  **{f"train_{key}": value / sample_count for key, value in components.items()}}
        improved, should_stop = stopper.update(record["train_ce"], epoch)
        record.update(train_ce_improved=improved,
                      non_improving_epochs=stopper.bad_epochs,
                      early_stop_triggered=should_stop)
        history.append(record)
        log_record(folder, record)
        write_json(folder / "history.json", history)
        if should_stop:
            log_record(folder, {"early_stop": stopper.state()})
            break
    final_epoch = history[-1]["epoch"]
    stop_reason = "train_ce_patience" if stopper.triggered else "max_epochs"
    # Save the final model BEFORE seeing any TEST prediction or metric.
    torch.save({"state_dict": model.state_dict(), "config": asdict(configuration),
                "epoch": final_epoch, "classes": classes.tolist(), "protocol": PROTOCOL,
                "seed": seed, "checkpoint_strategy": CHECKPOINT_STRATEGY,
                "max_epochs": training["epochs"], "stop_reason": stop_reason,
                "early_stop_state": stopper.state()},
               folder / "final.pt")
    test_loader = make_loader(test_x, test_y, test_mask, False)
    test, test_logits, test_labels, confusion = evaluate(model, test_loader, device, len(classes))
    np.savez_compressed(folder / "test_outputs.npz", logits=test_logits, labels=test_labels,
                        predictions=test_logits.argmax(axis=1), confusion=confusion,
                        sample_indices=np.arange(len(test_y)), classes=classes)
    result = {
        "dataset": dataset, "protocol": PROTOCOL, "seed": seed,
        "final_epoch": final_epoch, "max_epochs": training["epochs"],
        "epochs_run": len(history), "test": test,
        "classes": classes.tolist(), "train_samples": len(train_y),
        "validation_samples": 0, "test_samples": len(test_y),
        "official_split_preserved": True, "early_stopping": True,
        "stopped_early": final_epoch < training["epochs"], "stop_reason": stop_reason,
        "early_stop_state": stopper.state(),
        "test_evaluations": 1, "test_used_for_selection": False,
        "checkpoint_strategy": CHECKPOINT_STRATEGY,
    }
    write_json(folder / "metrics.json", result)
    log_record(folder, {"final": result})
    del model, optimizer, scheduler, train_loader, test_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_summaries(output, results):
    rows = []
    for result in results:
        rows.append({
            "dataset": result["dataset"], "protocol": PROTOCOL, "seed": result["seed"],
            "final_epoch": result["final_epoch"], "epochs_run": result["epochs_run"],
            "max_epochs": result["max_epochs"], "stop_reason": result["stop_reason"],
            "patience": result["early_stop_state"]["patience"],
            "train_samples": result["train_samples"], "validation_samples": result["validation_samples"],
            "test_samples": result["test_samples"],
            "test_accuracy": result["test"]["accuracy"], "test_macro_f1": result["test"]["macro_f1"],
        })
    if not rows:
        return
    with (output / "seed_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        selected = [row for row in rows if row["dataset"] == dataset]
        summary = {"dataset": dataset, "seed_count": len(selected)}
        for key, prefix in (("test_accuracy", "ACC"), ("test_macro_f1", "macro_F1")):
            values = np.array([row[key] for row in selected]) * 100
            summary[prefix + "_mean_pct"] = float(values.mean())
            summary[prefix + "_sample_sd_pct"] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary_rows.append(summary)
    with (output / "dataset_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(output / "summary.json", {
        "protocol": PROTOCOL, "official_split_preserved": True,
        "early_stop_monitor": EARLY_STOP_MONITOR,
        "completed_runs": len(results), "datasets": summary_rows,
        "dataset_average_ACC_pct": float(np.mean([row["ACC_mean_pct"] for row in summary_rows])),
        "dataset_average_macro_F1_pct": float(np.mean([row["macro_F1_mean_pct"] for row in summary_rows])),
        "sd_convention": "sample SD (ddof=1); null for a single seed",
    })


def parser():
    root = Path(__file__).resolve().parent
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--config", type=Path, default=root / "configs/uea18.json")
    argument_parser.add_argument("--data_root", "--data-root", type=Path, default=root / "data")
    argument_parser.add_argument("--output", type=Path, help="New, nonexisting output directory")
    argument_parser.add_argument("--datasets", nargs="+", help="Default: all datasets in the config")
    argument_parser.add_argument("--seeds", nargs="+", type=nonnegative, default=list(SEEDS))
    argument_parser.add_argument("--device", default="auto")
    argument_parser.add_argument("--epochs", type=int, help="Maximum epoch budget; optional override")
    argument_parser.add_argument("--patience", type=int, help="Override TRAIN-CE early-stopping patience")
    argument_parser.add_argument("--early_stop_min_delta", "--early-stop-min-delta", type=float,
                                 help="Required absolute CE decrease; default 0.0")
    argument_parser.add_argument("--num_workers", type=nonnegative)
    argument_parser.add_argument("--dry_run", "--dry-run", action="store_true",
                                 help="Preflight data and recipes without Torch or output writes")
    return argument_parser


def main(argv=None):
    args = parser().parse_args(argv)
    args.config, args.data_root = args.config.resolve(), args.data_root.resolve()
    recipes = read_json(args.config)
    profiles = recipes.get("datasets", {})
    datasets = args.datasets if args.datasets is not None else list(profiles)
    if not datasets or len(datasets) != len(set(datasets)) or set(datasets) - set(profiles):
        raise ValueError("Choose unique dataset names present in the configuration")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Choose unique model seeds")
    if any(seed >= 2 ** 32 for seed in args.seeds):
        raise ValueError("Model seeds must lie in [0, 2**32)")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = (args.output if args.output is not None else
              Path(__file__).resolve().parent / "result" / f"uea18_train_ce_earlystop_{timestamp}").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output directory: {output}")
    if args.data_root == output or args.data_root.is_relative_to(output) or args.config.is_relative_to(output):
        raise ValueError("Output must not contain the source data/configuration")
    plan = []
    for dataset in datasets:
        data_path = find_npz(args.data_root, dataset)
        with np.load(data_path, allow_pickle=False) as data:
            missing = {"train_x", "train_y", "test_x", "test_y"} - set(data.files)
            if missing:
                raise ValueError(f"{dataset}: missing {sorted(missing)}")
            train_labels, test_labels = np.asarray(data["train_y"]), np.asarray(data["test_y"])
        metadata = official_metadata(train_labels, test_labels)
        runs = []
        for seed in args.seeds:
            model, training = resolve_recipe(recipes, dataset, seed, args)
            runs.append({"seed": seed, "model": model, "training": training})
        plan.append({"dataset": dataset, "data_path": data_path, "partitions": metadata, "runs": runs})
        print(json.dumps({"preflight": dataset, "source": str(data_path), "partitions": metadata,
                          "model_seeds": args.seeds}, ensure_ascii=False), flush=True)
    if args.dry_run:
        print(f"DRY RUN OK: {len(datasets)} datasets x {len(args.seeds)} seeds; no files written.")
        return 0
    # Import training dependencies before creating any output directory.
    try:
        import torch  # noqa: F401
        from same import SAMEConfig
    except ImportError as error:
        raise RuntimeError("Install the requirements in a Python environment with PyTorch") from error
    for item in plan:
        for run in item["runs"]:
            SAMEConfig(**{**run["model"], "channels": 1,
                        "classes": len(item["partitions"]["class_counts"])})
    config_sha256 = sha256_file(args.config)
    output.mkdir(parents=True, exist_ok=False)
    print(f"NEW OUTPUT DIRECTORY: {output}", flush=True)
    write_json(output / "config_source.json", recipes)
    write_json(output / "experiment_manifest.json", {
        "created_utc": timestamp, "config_source": str(args.config), "config_sha256": config_sha256,
        "data_root": str(args.data_root), "datasets": datasets, "model_seeds": args.seeds,
        "protocol": PROTOCOL, "official_split_preserved": True,
        "partition_rule": "use all official TRAIN for training and all official TEST for final evaluation",
        "checkpoint_strategy": CHECKPOINT_STRATEGY,
        "early_stopping": True, "early_stop_monitor": EARLY_STOP_MONITOR,
        "validation_used": False,
        "diagnostics": "no automatic explanation export or plotting",
        "script_sha256": sha256_file(Path(__file__)),
        "model_sources_sha256": {
            path.relative_to(Path(__file__).resolve().parent).as_posix(): sha256_file(path)
            for path in sorted((Path(__file__).resolve().parent / "same").glob("*.py"))
        },
    })
    results, active_run = [], None
    try:
        for item in plan:
            dataset = item["dataset"]
            data_folder = output / "data_manifests"
            data_folder.mkdir(parents=True, exist_ok=True)
            data_sha256 = sha256_file(item["data_path"])
            write_json(data_folder / f"{dataset}.json", {
                **item["partitions"], "data_source": str(item["data_path"]), "data_sha256": data_sha256,
            })
            for run in item["runs"]:
                active_run = {"dataset": dataset, "seed": run["seed"]}
                print(f"TRAIN {dataset} seed={run['seed']} max_epochs={run['training']['epochs']} "
                      f"patience={run['training']['patience']} monitor=TRAIN_CE (no validation)", flush=True)
                result = train_one(dataset, run["seed"], item["data_path"],
                                   output / dataset / f"seed_{run['seed']}", run["model"],
                                   run["training"], args.device, data_sha256, config_sha256)
                results.append(result)
                save_summaries(output, results)
                write_json(output / "progress.json", {"status": "running", "completed_runs": len(results),
                                                      "planned_runs": len(datasets) * len(args.seeds)})
    except BaseException as error:
        write_json(output / "failure.json", {"active_run": active_run, "error_type": type(error).__name__,
                                              "error": str(error), "completed_runs": len(results),
                                              "automatic_retry": False})
        print(f"Run stopped. Completed and partial outputs retained at: {output}", file=sys.stderr)
        raise
    write_json(output / "progress.json", {"status": "completed", "completed_runs": len(results),
                                          "planned_runs": len(datasets) * len(args.seeds)})
    print(f"COMPLETED. Results: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
