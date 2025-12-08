import argparse
import json
from pathlib import Path
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

from .config import Config
from .data_prep import prepare_data
from .models import build_lstm
from .metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Train LSTM model for gare pricing.")
    p.add_argument("--gare", required=True, help="Gare name to train on.")
    p.add_argument(
        "--config",
        default=None,
        help="Optional path to override config (not implemented yet).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    data = prepare_data(cfg, gare=args.gare)

    model = build_lstm(input_shape=data["X_train_seq"].shape[1:], cfg=cfg)

    es = EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience,
        mode="min",
        restore_best_weights=True,
    )

    history = model.fit(
        data["X_train_seq"],
        data["y_train_seq"],
        validation_split=cfg.validation_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=False,
        callbacks=[es],
        verbose=0,
    )

    # Evaluate
    y_pred_scaled = model.predict(data["X_test_seq"])
    y_pred = data["y_scaler"].inverse_transform(y_pred_scaled)
    y_true = data["y_scaler"].inverse_transform(data["y_test_seq"])

    metrics = compute_metrics(y_true.flatten(), y_pred.flatten())

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    model_path = cfg.models_dir / f"lstm_{args.gare}.keras"
    model.save(model_path)

    metrics_path = cfg.outputs_dir / f"metrics_lstm_{args.gare}.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    # Comparison table (aligned with test indices after n_steps)
    aligned_index = data["df_test"].index[cfg.n_steps :]


    comparison = pd.DataFrame(
        {"y_true": y_true.flatten(), "y_pred": y_pred.flatten()},
        index=aligned_index,
    )
    comparison_path = cfg.outputs_dir / f"comparison_lstm_{args.gare}.csv"
    comparison.to_csv(comparison_path)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved comparison to {comparison_path}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
