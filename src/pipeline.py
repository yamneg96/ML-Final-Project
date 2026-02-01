# src/pipeline.py
from data_preprocessing import load_data, preprocess_data, split_and_save
from train import train_logistic, train_random_forest, save_model
from evaluate import evaluate_model
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    # Load & preprocess
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_save(X, y)

    # Train models
    print("Training Logistic Regression...")
    lr_model = train_logistic(X_train, y_train)
    save_model(lr_model, "logistic_model.pkl")

    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, "random_forest_model.pkl")

    # Evaluate
    print("\nEvaluating models on test set:")
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    # Results table
    results = pd.DataFrame([lr_metrics, rf_metrics], index=["Logistic Regression","Random Forest"])
    print(results)

    # Select best model based on ROC-AUC (or accuracy as tiebreaker)
    if rf_metrics['roc_auc'] > lr_metrics['roc_auc']:
        best_model = rf_model
        best_name = "Random Forest"
    elif lr_metrics['roc_auc'] > rf_metrics['roc_auc']:
        best_model = lr_model
        best_name = "Logistic Regression"
    else:
        # Tiebreaker: use accuracy
        if rf_metrics['accuracy'] > lr_metrics['accuracy']:
            best_model = rf_model
            best_name = "Random Forest"
        else:
            best_model = lr_model
            best_name = "Logistic Regression"
    
    # Save best model
    save_model(best_model, "best_model.pkl")
    print(f"\nBest model: {best_name} (saved as best_model.pkl)")

    # Save results
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "model_comparison.csv"
    results.to_csv(out_path, index=True)
    print(f"Pipeline complete. Results saved to {out_path}")
