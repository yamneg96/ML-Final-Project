# src/pipeline.py
from data_preprocessing import load_data, preprocess_data, split_and_save
from train import train_logistic, train_random_forest, save_model
from evaluate import evaluate_model
import pandas as pd

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

    # Save results
    results.to_csv("../results/model_comparison.csv")
    print("Pipeline complete. Results saved to ../results/model_comparison.csv")
