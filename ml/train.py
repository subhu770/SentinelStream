"""Machine learning training pipeline for SentinelStream Fraud Detection.

Generates realistic synthetic financial transactions with extreme 0.2% class imbalance,
non-linear feature interactions, Gaussian noise, and realistic legitimate/fraud overlap.
Trains an XGBoost classifier with scale_pos_weight, evaluates PR-AUC/ROC-AUC, and exports artifacts.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Add repository root to path for module imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.models.schemas import MerchantCategory
from app.services.feature_engineering import ALL_FEATURE_NAMES, extract_features_dataframe


def generate_synthetic_transactions(
    n_samples: int = 100_000,
    fraud_rate: float = 0.002,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates realistic synthetic transactions with probabilistic fraud and non-linear feature overlap.

    Key characteristics:
    - 0.2% extreme class imbalance (approx 200 frauds in 100,000 transactions).
    - Multi-modal legitimate behaviors: holiday spenders, travelers, errands velocity.
    - Heterogeneous fraud personas: Account Takeover (ATO), Card Testing Swarms, High-ticket bustout.
    - Latent probabilistic risk score with Gaussian noise to avoid artificial linear separability.

    Args:
        n_samples: Total number of transactions (default: 100,000).
        fraud_rate: Target fraction of fraudulent transactions (default: 0.002 -> 0.2%).
        seed: Random seed for deterministic reproducibility.

    Returns:
        pd.DataFrame containing realistic transactions and binary 'is_fraud' ground truth.
    """
    rng = np.random.default_rng(seed)
    target_fraud_count = int(n_samples * fraud_rate)

    print(
        f"Generating {n_samples:,} synthetic transactions with realistic non-linear overlap "
        f"({target_fraud_count:,} target frauds - {fraud_rate * 100:.2f}%)..."
    )

    now = datetime.now(timezone.utc)
    seconds_30d = 30 * 24 * 3600
    all_categories = [c.value for c in MerchantCategory]

    # 1. Establish User Population Baseline (~10,000 historical profiles)
    n_users = 10_000
    user_ids = [f"usr_{i:05d}" for i in range(n_users)]
    user_mean_amounts = rng.lognormal(mean=3.8, sigma=0.6, size=n_users)  # ~$45 median spend
    user_std_amounts = user_mean_amounts * rng.uniform(0.2, 0.45, size=n_users)

    # Assign transactions to users
    assigned_user_idxs = rng.integers(0, n_users, size=n_samples)
    tx_users = [user_ids[idx] for idx in assigned_user_idxs]
    tx_user_avg = np.array([user_mean_amounts[idx] for idx in assigned_user_idxs])
    tx_user_std = np.array([user_std_amounts[idx] for idx in assigned_user_idxs])

    # 2. Transaction Amounts: Lognormal variations + legitimate outlier spikes + Gaussian noise
    amount_multipliers = rng.lognormal(mean=0.0, sigma=0.55, size=n_samples)
    # Legitimate occasional spikes (e.g. appliances, gifts, weekend splurges)
    legit_spikes = (rng.uniform(0, 1, size=n_samples) < 0.02) * rng.uniform(3.0, 7.0, size=n_samples)
    amounts = np.maximum(3.0, tx_user_avg * (amount_multipliers + legit_spikes))
    # Gaussian measurement noise
    amounts = np.maximum(1.0, amounts + rng.normal(0, 2.5, size=n_samples))

    # 3. Location Delta: Base daily proximity + legitimate travel/vacation spikes + noise
    base_loc = rng.exponential(scale=4.0, size=n_samples)
    # 3.5% of legitimate transactions involve business trips or interstate travel (50 - 600km)
    travel_loc = (rng.uniform(0, 1, size=n_samples) < 0.035) * rng.gamma(shape=2.5, scale=120.0, size=n_samples)
    location_delta = np.maximum(0.0, base_loc + travel_loc + rng.normal(0, 1.0, size=n_samples))

    # 4. Transaction Velocity: Poisson distribution + noise
    base_velocity = rng.poisson(lam=1.3, size=n_samples) + 1
    velocity_noise = rng.choice([-1, 0, 1], size=n_samples, p=[0.1, 0.8, 0.1])
    velocity = np.maximum(1, base_velocity + velocity_noise)

    # 5. Merchant Categories Distribution
    cat_weights = [0.22, 0.10, 0.08, 0.18, 0.22, 0.02, 0.02, 0.08, 0.05, 0.03]
    cat_weights = np.array(cat_weights) / sum(cat_weights)
    categories = rng.choice(all_categories, size=n_samples, p=cat_weights)

    # 6. Timestamps & Diurnal Patterns
    timestamps = [now - timedelta(seconds=float(rng.uniform(0, seconds_30d))) for _ in range(n_samples)]
    hours = np.array([ts.hour for ts in timestamps])

    # 7. Non-linear Latent Fraud Risk Function (Probabilistic Combination)
    z_amount = (amounts - tx_user_avg) / (tx_user_std + 1e-5)
    log_loc = np.log1p(location_delta)

    cat_risk_map = {
        "crypto": 1.4,
        "gambling": 1.2,
        "electronics": 0.8,
        "travel": 0.5,
        "retail": 0.0,
        "dining": -0.2,
        "groceries": -0.4,
        "utilities": -0.5,
        "entertainment": 0.2,
        "services": 0.1,
    }
    cat_risk = np.array([cat_risk_map[c] for c in categories])

    # Interaction A: Elevated spend AND high-risk merchant AND abnormal geographic displacement
    interact_high_spend = np.clip(z_amount - 1.5, 0, 8) * (log_loc > 3.0) * np.maximum(0, cat_risk + 0.2)

    # Interaction B: Elevated velocity AND nocturnal hours (1 AM - 5 AM) AND displacement
    night_mask = ((hours >= 1) & (hours <= 5)).astype(float)
    interact_velocity = np.clip(velocity - 2, 0, 10) * (0.8 + 0.8 * night_mask) * (log_loc > 2.0)

    # Interaction C: Card testing micro-bursts (low amounts, rapid successive attempts)
    interact_card_testing = (amounts < 25.0).astype(float) * np.clip(velocity - 3, 0, 8) * 1.5

    # Interaction D: Unobserved real-world Gaussian variance / latent noise
    latent_noise = rng.normal(0, 1.30, size=n_samples)

    # Non-linear composite logit
    latent_score = (
        0.52 * z_amount
        + 0.48 * log_loc
        + 0.58 * velocity
        + 0.72 * cat_risk
        + 1.00 * interact_high_spend
        + 0.95 * interact_velocity
        + 0.80 * interact_card_testing
        + latent_noise
    )

    # Softmax sampling across probability spectrum creates realistic overlap
    exp_scores = np.exp((latent_score - np.max(latent_score)) / 1.95)
    sampling_probs = exp_scores / np.sum(exp_scores)

    fraud_indices = rng.choice(n_samples, size=target_fraud_count, replace=False, p=sampling_probs)
    is_fraud = np.zeros(n_samples, dtype=int)
    is_fraud[fraud_indices] = 1

    actual_fraud_count = int(np.sum(is_fraud))
    print(f"Generated {actual_fraud_count:,} fraud events ({actual_fraud_count / n_samples * 100:.3f}%).")

    df = pd.DataFrame({
        "transaction_id": [f"tx_{i:06d}" for i in range(n_samples)],
        "user_id": tx_users,
        "amount": amounts,
        "timestamp": timestamps,
        "merchant_category": categories,
        "location_delta": location_delta,
        "user_avg_amount_30d": tx_user_avg,
        "user_std_amount_30d": tx_user_std,
        "user_tx_count_1h": velocity,
        "is_fraud": is_fraud,
    })

    return df


def train_fraud_model(
    df: pd.DataFrame,
    artifacts_dir: Path,
    model_filename: str = "fraud_model.joblib",
) -> Path:
    """Transforms features, trains XGBoost with scale_pos_weight, and exports model artifact."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    target_path = artifacts_dir / model_filename

    # Extract aligned streaming features
    print("Computing feature pipeline transformations...")
    X = extract_features_dataframe(df)
    y = df["is_fraud"].values

    print(f"Features matrix shape: {X.shape}")

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = float(n_neg / n_pos)

    print(f"Training set: {len(y_train):,} samples (Pos: {n_pos}, Neg: {n_neg})")
    print(f"Calculated scale_pos_weight for imbalance: {scale_pos_weight:.2f}")

    # Initialize XGBoost Classifier with scale_pos_weight
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric=["aucpr", "auc", "logloss"],
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    print("Fitting XGBoost Classifier...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    # Evaluation on Held-Out Test Set
    print("\n--- Model Evaluation on Test Set ---")
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

    # Find optimal F1 threshold from precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * (precisions * recalls) / (precisions + recalls + 1e-10),
        0.0,
    )
    best_idx = np.argmax(f1_scores[:-1])
    opt_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    print(f"Optimal F1 Threshold: {opt_threshold:.4f} (Max F1: {best_f1:.4f})")

    # Classification report at 0.50 default and at optimal threshold
    y_pred_default = (y_pred_proba >= 0.50).astype(int)
    print("\nClassification Report (threshold = 0.50):")
    print(classification_report(y_test, y_pred_default, digits=4))

    print("Confusion Matrix (threshold = 0.50):")
    cm = confusion_matrix(y_test, y_pred_default)
    print(f"TN: {cm[0, 0]:,}, FP: {cm[0, 1]:,}")
    print(f"FN: {cm[1, 0]:,}, TP: {cm[1, 1]:,}")

    y_pred_opt = (y_pred_proba >= opt_threshold).astype(int)
    print(f"\nClassification Report at Optimal Threshold ({opt_threshold:.4f}):")
    print(classification_report(y_test, y_pred_opt, digits=4))

    # Package production artifact
    artifact_payload = {
        "model": model,
        "metadata": {
            "version": "xgboost_v1.0",
            "features": ALL_FEATURE_NAMES,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "fraud_rate": float(np.mean(y)),
            "scale_pos_weight": scale_pos_weight,
            "metrics": {
                "roc_auc": round(float(roc_auc), 5),
                "pr_auc": round(float(pr_auc), 5),
                "optimal_threshold": round(opt_threshold, 4),
                "f1_at_optimal": round(best_f1, 4),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    joblib.dump(artifact_payload, target_path, compress=3)
    file_size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"\nArtifact successfully exported to: {target_path} ({file_size_mb:.2f} MB)")

    return target_path


def main() -> None:
    """Execution entry point for training pipeline."""
    artifacts_dir = BASE_DIR / "ml" / "artifacts"
    df = generate_synthetic_transactions(n_samples=100_000, fraud_rate=0.002, seed=42)
    train_fraud_model(df, artifacts_dir)


if __name__ == "__main__":
    main()
