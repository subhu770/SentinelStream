"""Feature engineering and transformation pipeline for real-time streaming inference.

Designed for sub-millisecond execution using optimized NumPy scalar operations
rather than heavyweight DataFrame abstractions.
"""

import math
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from app.models.schemas import MerchantCategory, TransactionPayload

# Fixed vocabulary for category one-hot encoding
CATEGORY_ORDER: List[str] = [
    MerchantCategory.RETAIL.value,
    MerchantCategory.ELECTRONICS.value,
    MerchantCategory.TRAVEL.value,
    MerchantCategory.DINING.value,
    MerchantCategory.GROCERIES.value,
    MerchantCategory.GAMBLING.value,
    MerchantCategory.CRYPTO.value,
    MerchantCategory.UTILITIES.value,
    MerchantCategory.ENTERTAINMENT.value,
    MerchantCategory.SERVICES.value,
]

CATEGORY_TO_IDX: Dict[str, int] = {cat: idx for idx, cat in enumerate(CATEGORY_ORDER)}

NUMERICAL_FEATURE_NAMES: List[str] = [
    "amount",
    "log_amount",
    "location_delta",
    "log_location_delta",
    "user_tx_count_1h",
    "amount_zscore",
    "amount_to_avg_ratio",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

CATEGORY_FEATURE_NAMES: List[str] = [f"cat_{cat}" for cat in CATEGORY_ORDER]

ALL_FEATURE_NAMES: List[str] = NUMERICAL_FEATURE_NAMES + CATEGORY_FEATURE_NAMES
TOTAL_FEATURES_COUNT: int = len(ALL_FEATURE_NAMES)


def extract_features_single(payload: TransactionPayload) -> np.ndarray:
    """Extract features from a single TransactionPayload directly into a NumPy vector.

    Execution time: < 0.05ms (direct CPU cache friendly vector creation).

    Returns:
        np.ndarray: 2D array of shape (1, TOTAL_FEATURES_COUNT) with float32 dtype.
    """
    # 1. Base values & defaults
    amount = float(payload.amount)
    loc_delta = float(payload.location_delta)
    tx_count_1h = float(payload.user_tx_count_1h if payload.user_tx_count_1h is not None else 1)
    
    avg_amt = float(payload.user_avg_amount_30d if payload.user_avg_amount_30d is not None else amount)
    std_amt = float(payload.user_std_amount_30d if (payload.user_std_amount_30d is not None and payload.user_std_amount_30d > 0.0) else 1.0)

    # 2. Mathematical streaming features
    log_amt = math.log1p(max(amount, 0.0))
    log_loc = math.log1p(max(loc_delta, 0.0))
    z_score = (amount - avg_amt) / (std_amt + 1e-6)
    ratio = amount / (avg_amt + 1e-6)

    # 3. Temporal features
    ts = payload.timestamp
    hour = float(ts.hour)
    weekday = float(ts.weekday())
    is_weekend = 1.0 if weekday in (5, 6) else 0.0

    # 4. Construct feature vector directly
    feat_vector = np.zeros((1, TOTAL_FEATURES_COUNT), dtype=np.float32)
    feat_vector[0, 0] = amount
    feat_vector[0, 1] = log_amt
    feat_vector[0, 2] = loc_delta
    feat_vector[0, 3] = log_loc
    feat_vector[0, 4] = tx_count_1h
    feat_vector[0, 5] = z_score
    feat_vector[0, 6] = ratio
    feat_vector[0, 7] = hour
    feat_vector[0, 8] = weekday
    feat_vector[0, 9] = is_weekend

    # 5. One-hot category encoding
    cat_str = payload.merchant_category
    cat_idx = CATEGORY_TO_IDX.get(cat_str)
    if cat_idx is not None:
        feat_vector[0, len(NUMERICAL_FEATURE_NAMES) + cat_idx] = 1.0

    return feat_vector


def extract_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from a Pandas DataFrame (used for training / evaluation).

    Produces columns in identical order to extract_features_single.
    """
    df_out = pd.DataFrame(index=df.index)

    # Base values
    amount = df["amount"].astype(float)
    loc_delta = df["location_delta"].astype(float)
    tx_count_1h = df["user_tx_count_1h"].fillna(1).astype(float)
    avg_amt = df["user_avg_amount_30d"].fillna(amount).astype(float)
    std_amt = df["user_std_amount_30d"].fillna(1.0).replace(0.0, 1.0).astype(float)

    # Streaming features
    df_out["amount"] = amount
    df_out["log_amount"] = np.log1p(amount.clip(lower=0))
    df_out["location_delta"] = loc_delta
    df_out["log_location_delta"] = np.log1p(loc_delta.clip(lower=0))
    df_out["user_tx_count_1h"] = tx_count_1h
    df_out["amount_zscore"] = (amount - avg_amt) / (std_amt + 1e-6)
    df_out["amount_to_avg_ratio"] = amount / (avg_amt + 1e-6)

    # Temporal features
    timestamps = pd.to_datetime(df["timestamp"])
    df_out["hour_of_day"] = timestamps.dt.hour.astype(float)
    df_out["day_of_week"] = timestamps.dt.dayofweek.astype(float)
    df_out["is_weekend"] = df_out["day_of_week"].isin([5, 6]).astype(float)

    # One-hot encoded categories
    for cat in CATEGORY_ORDER:
        col_name = f"cat_{cat}"
        df_out[col_name] = (df["merchant_category"] == cat).astype(float)

    # Ensure column ordering matches ALL_FEATURE_NAMES
    return df_out[ALL_FEATURE_NAMES].astype(np.float32)
