"""Unit tests for Pydantic v2 schemas and validation rules."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError
from app.models.schemas import (
    Decision,
    MerchantCategory,
    PredictionResponse,
    RiskLevel,
    TransactionPayload,
)


def test_valid_transaction_payload():
    """Tests successful validation of a complete transaction payload."""
    payload = TransactionPayload(
        transaction_id="tx_test_101",
        user_id="usr_555",
        amount=150.00,
        timestamp=datetime.now(timezone.utc),
        merchant_category=MerchantCategory.ELECTRONICS,
        location_delta=10.5,
        user_avg_amount_30d=120.0,
        user_std_amount_30d=25.0,
        user_tx_count_1h=2,
    )
    assert payload.transaction_id == "tx_test_101"
    assert payload.amount == 150.00
    assert payload.merchant_category == "electronics"
    assert payload.location_delta == 10.5


def test_transaction_payload_timestamp_parsing():
    """Tests ISO string and unix epoch timestamp parsing in Pydantic validator."""
    # Test ISO format string
    payload_iso = TransactionPayload(
        transaction_id="tx_iso",
        user_id="usr_1",
        amount=10.0,
        timestamp="2026-11-02T12:00:00Z",
        merchant_category=MerchantCategory.RETAIL,
        location_delta=1.0,
    )
    assert payload_iso.timestamp.year == 2026
    assert payload_iso.timestamp.month == 11

    # Test epoch float timestamp
    payload_epoch = TransactionPayload(
        transaction_id="tx_epoch",
        user_id="usr_2",
        amount=25.0,
        timestamp=1730550000.0,
        merchant_category=MerchantCategory.DINING,
        location_delta=0.0,
    )
    assert isinstance(payload_epoch.timestamp, datetime)


def test_invalid_negative_amount_fails_validation():
    """Tests that negative or zero transaction amounts raise ValidationError."""
    with pytest.raises(ValidationError) as exc:
        TransactionPayload(
            transaction_id="tx_invalid_amt",
            user_id="usr_1",
            amount=-50.0,  # Invalid: gt=0.0 required
            timestamp=datetime.now(timezone.utc),
            merchant_category=MerchantCategory.GROCERIES,
            location_delta=0.0,
        )
    assert "amount" in str(exc.value)


def test_invalid_location_delta_fails_validation():
    """Tests that negative location delta raises ValidationError."""
    with pytest.raises(ValidationError) as exc:
        TransactionPayload(
            transaction_id="tx_invalid_loc",
            user_id="usr_1",
            amount=50.0,
            timestamp=datetime.now(timezone.utc),
            merchant_category=MerchantCategory.GROCERIES,
            location_delta=-5.0,  # Invalid: ge=0.0 required
        )
    assert "location_delta" in str(exc.value)


def test_invalid_merchant_category_fails_validation():
    """Tests that unregistered category strings are rejected."""
    with pytest.raises(ValidationError) as exc:
        TransactionPayload(
            transaction_id="tx_invalid_cat",
            user_id="usr_1",
            amount=50.0,
            timestamp=datetime.now(timezone.utc),
            merchant_category="illegal_category",  # type: ignore[arg-type]
            location_delta=0.0,
        )
    assert "merchant_category" in str(exc.value)


def test_prediction_response_schema():
    """Tests serialization and constraints on PredictionResponse."""
    resp = PredictionResponse(
        transaction_id="tx_resp_01",
        is_fraud=True,
        fraud_probability=0.924,
        risk_level=RiskLevel.HIGH,
        decision=Decision.DECLINE,
        inference_time_ms=1.45,
        threshold_used=0.50,
        model_version="xgboost_v1.0",
    )
    assert resp.is_fraud is True
    assert resp.risk_level == "HIGH"
    assert resp.decision == "DECLINE"
    assert resp.inference_time_ms == 1.45
