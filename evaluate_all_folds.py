import sys
import os
from src.manual_evaluation import evaluate_single_fold
from src.utils import load_config

# add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)


def main():
    config = load_config()

    # select ensemble config
    ensemble_key = "ensemble1"
    if ensemble_key not in config:
        print(f"Error: {ensemble_key} not found in config.")
        return

    ensemble_models = config[ensemble_key]["models"]
    k_folds = config["hyperparameters"].get("k_folds", 5)

    print(
        f"Evaluating all {len(ensemble_models) * k_folds} models from {ensemble_key} on Hold-out Test Set...\n"
    )

    results = []

    for model_cfg in ensemble_models:
        name = model_cfg["name"]
        norm = model_cfg["normalisation"]

        for fold in range(k_folds):
            try:
                acc = evaluate_single_fold(name, fold, norm)
                if acc is not None:
                    results.append(
                        {"Model": name, "Fold": fold, "Norm": norm, "Accuracy": acc}
                    )
            except Exception as e:
                print(f"Failed to evaluate {name} Fold {fold}: {e}")

    # print Summary Table
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"{'Model':<20} {'Fold':<5} {'Norm':<10} {'Accuracy':<10}")
    print("-" * 50)

    for r in results:
        print(f"{r['Model']:<20} {r['Fold']:<5} {r['Norm']:<10} {r['Accuracy']:.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    main()
