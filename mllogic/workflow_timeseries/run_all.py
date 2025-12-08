import argparse
import json
from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping

from .config import Config
from .data_prep import prepare_data
from .models import build_lstm, build_gru
from .metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Run full workflow (LSTM + GRU).")
    p.add_argument("--gare", required=True, help="Gare name to train on.")
    p.add_argument("--model", choices=["lstm", "gru", "both"], default="lstm")
    return p.parse_args()


def train_and_eval(model_name: str, build_fn, data, cfg: Config, gare: str):
    es = EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience,
        mode="min",
        restore_best_weights=True,
    )

    model = build_fn(input_shape=data["X_train_seq"].shape[1:], cfg=cfg)
    model.fit(
        data["X_train_seq"],
        data["y_train_seq"],
        validation_split=cfg.validation_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=False,
        callbacks=[es],
        verbose=0,
    )

    y_pred_scaled = model.predict(data["X_test_seq"])
    y_pred = data["y_scaler"].inverse_transform(y_pred_scaled)
    y_true = data["y_scaler"].inverse_transform(data["y_test_seq"])

    metrics = compute_metrics(y_true.flatten(), y_pred.flatten())

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    model_path = cfg.models_dir / f"{model_name}_{gare}.keras"
    model.save(model_path)

    metrics_path = cfg.outputs_dir / f"metrics_{model_name}_{gare}.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    import pandas as pd

    aligned_index = data["df_test"].index[cfg.n_steps :]
    comparison = pd.DataFrame(
        {"y_true": y_true.flatten(), "y_pred": y_pred.flatten()},
        index=aligned_index,
    )
    comparison_path = cfg.outputs_dir / f"comparison_{model_name}_{gare}.csv"
    comparison.to_csv(comparison_path)

    print(f"[{model_name}] Saved model to {model_path}")
    print(f"[{model_name}] Saved metrics to {metrics_path}")
    print(f"[{model_name}] Saved comparison to {comparison_path}")
    print(f"[{model_name}] Metrics: {metrics}")


def main():
    args = parse_args()
    cfg = Config()
    data = prepare_data(cfg, gare=args.gare)

    if args.model in ("lstm", "both"):
        train_and_eval("lstm", build_lstm, data, cfg, args.gare)
    if args.model in ("gru", "both"):
        train_and_eval("gru", build_gru, data, cfg, args.gare)


if __name__ == "__main__":
    main()
