
import json
from pathlib import Path

def save_dml_results(results: dict, filepath: str) -> None:
    """
    Sauvegarde les résultats DML dans un fichier JSON.

    Paramètres
    ----------
    results : dict
        Dictionnaire retourné par run_dml_sklearn_linear.
    filepath : str
        Chemin du fichier de sortie (ex. 'outputs/dml_gp_1km_results.json').
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Résultats DML sauvegardés dans : {path.resolve()}")
