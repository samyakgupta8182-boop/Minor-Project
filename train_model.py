import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def generate_synthetic_data(num_samples=10000):
    """
    Generates a statistically realistic banking customer churn dataset.
    Features and behavioral correlations match real-world banking churn benchmarks.
    """
    np.random.seed(42)
    
    # 1. Base customer identifiers
    customer_ids = np.arange(15634000, 15634000 + num_samples)
    surnames = ["Smith", "Martin", "Brown", "Taylor", "Wilson", "Davis", "White", "Clark", "Hall", "Thomas",
                "Garcia", "Martinez", "Robinson", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young"]
    surname_col = np.random.choice(surnames, size=num_samples)
    
    # 2. Demographic & financial attributes
    credit_scores = np.clip(np.random.normal(650, 96, num_samples).astype(int), 350, 850)
    geographies = np.random.choice(["France", "Spain", "Germany"], size=num_samples, p=[0.50, 0.25, 0.25])
    genders = np.random.choice(["Male", "Female"], size=num_samples, p=[0.54, 0.46])
    
    # Age distribution
    ages = np.clip(np.random.normal(39, 10, num_samples).astype(int), 18, 92)
    tenures = np.random.randint(0, 11, num_samples)
    
    # Balances: 36% zero balance, rest active deposits
    has_zero_balance = np.random.choice([True, False], size=num_samples, p=[0.36, 0.64])
    balances = np.random.normal(120000, 30000, num_samples)
    balances[has_zero_balance] = 0.0
    
    # Number of products, card, active status, salary
    num_products = np.random.choice([1, 2, 3, 4], size=num_samples, p=[0.508, 0.459, 0.027, 0.006])
    has_cr_card = np.random.choice([0, 1], size=num_samples, p=[0.29, 0.71])
    is_active_member = np.random.choice([0, 1], size=num_samples, p=[0.48, 0.52])
    salaries = np.random.uniform(10000, 200000, num_samples)
    
    # 3. Behavioral Risk Modeling
    # In banking data (e.g. Kaggle Churn Modeling benchmark):
    # - Inactivity is a critical precursor to exit
    # - Product saturation (3 or 4 products) causes high dissatisfaction/complication
    # - 2 products is the 'loyalty sweet spot'
    # - Customers aged 42+ show significantly higher churn
    # - German market shows higher propensity to switch
    # - Low credit score (<540) or large idle balance with zero activity adds churn pressure
    
    risk_points = np.zeros(num_samples)
    risk_points += (ages > 42) * 2.5
    risk_points += (ages > 52) * 1.5
    risk_points += (is_active_member == 0) * 2.5
    risk_points += (num_products >= 3) * 4.0
    risk_points += (num_products == 1) * 1.0
    risk_points -= (num_products == 2) * 2.5
    risk_points += (geographies == "Germany") * 1.8
    risk_points += (credit_scores < 540) * 1.5
    risk_points += ((balances > 90000) & (is_active_member == 0)) * 1.5
    risk_points += (genders == "Female") * 0.6
    
    # Realistic decision boundary with Gaussian boundary variance
    threshold = 4.8
    noisy_score = risk_points + np.random.normal(0, 0.35, num_samples)
    exited = (noisy_score > threshold).astype(int)
    
    # Subtle 1% edge-case flipping for real-world stochasticity
    flip_idx = np.random.choice(num_samples, size=int(0.01 * num_samples), replace=False)
    exited[flip_idx] = 1 - exited[flip_idx]
    
    df = pd.DataFrame({
        "CustomerId": customer_ids,
        "Surname": surname_col,
        "CreditScore": credit_scores,
        "Geography": geographies,
        "Gender": genders,
        "Age": ages,
        "Tenure": tenures,
        "Balance": balances,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": salaries,
        "Exited": exited
    })
    
    return df

def train_and_save_pipeline(df, save_dir):
    """
    Trains an optimized Random Forest Classifier using a scikit-learn Pipeline
    and saves the model artifacts and training metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Define features and target
    X = df.drop(columns=["CustomerId", "Surname", "Exited"])
    y = df["Exited"]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define column transformer preprocessing (including binary features!)
    numeric_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    categorical_features = ["Geography", "Gender"]
    binary_features = ["HasCrCard", "IsActiveMember"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("bin", "passthrough", binary_features)
        ]
    )
    
    # Define Pipeline with tuned Random Forest Classifier
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    step = max(1, len(fpr) // 100)
    metrics["roc_curve"] = {
        "fpr": fpr[::step].tolist(),
        "tpr": tpr[::step].tolist(),
        "auc": float(auc(fpr, tpr))
    }
    
    # Extract Feature Importances
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names + binary_features
    
    importances = pipeline.named_steps["classifier"].feature_importances_
    
    feature_importances = []
    for name, imp in zip(all_feature_names, importances):
        feature_importances.append({"feature": name, "importance": float(imp)})
    
    feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)
    metrics["feature_importances"] = feature_importances
    
    # Save model pipeline
    model_path = os.path.join(save_dir, "churn_model.joblib")
    joblib.dump(pipeline, model_path)
    
    # Save training dataset preview for demo data load
    data_path = os.path.join(save_dir, "demo_customers.csv")
    df.head(200).to_csv(data_path, index=False)
    
    # Save training dataset summary info
    metrics["dataset_summary"] = {
        "total_samples": len(df),
        "total_churn": int(df["Exited"].sum()),
        "churn_rate": float(df["Exited"].mean()),
        "active_rate": float(df["IsActiveMember"].mean())
    }
    
    # Save metrics JSON
    metrics_path = os.path.join(save_dir, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Demo data saved to {data_path}")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {metrics['f1_score']*100:.2f}%")
    print(f"ROC AUC:   {metrics['roc_curve']['auc']:.4f}")

if __name__ == "__main__":
    print("Generating improved realistic banking dataset...")
    df = generate_synthetic_data(10000)
    
    print("Training optimized model pipeline...")
    train_and_save_pipeline(df, os.path.dirname(os.path.abspath(__file__)))
