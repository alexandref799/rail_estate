from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class Config:
    # Data
    data_path: Path = Path(
        "/Users/matthieulataste/code/alexandref799/rail_estate/mllogic/df_new_gare.csv"
    )
    colonne_date: str = "date"
    colonne_gare: str = "nom_gare"
    target: str = "prix_m2"

    colonnes_numeriques: List[str] = field(
        default_factory=lambda: [
            "Surface reelle bati",
            "prix_m2",
            "Nombre pieces principales",
            "distance_gare_km",
            "relative_signature",
            "relative_opening",
            "taux",
        ]
    )
    colonnes_categorielles: List[str] = field(
        default_factory=lambda: [
            "Nature mutation",
            "Type local",
        ]
    )

    # Time setup
    forecast_horizon: int = 24
    n_steps: int = 12

    # Model hyperparameters
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    gru_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2

    # Training
    epochs: int = 200
    batch_size: int = 32
    validation_split: float = 0.2
    patience: int = 15
    learning_rate: float = 0.001

    # Outputs
    models_dir: Path = Path("mllogic/models")
    outputs_dir: Path = Path("mllogic/outputs")
