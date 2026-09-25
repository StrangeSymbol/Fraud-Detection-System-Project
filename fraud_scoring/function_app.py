"""
Azure Function: fraud scoring HTTP endpoint.

Accepts a POST request with a JSON body containing the 22 engineered
features for one transaction (the same feature set produced by
feature_engineering.py in the main project), and returns a fraud
probability plus a flagged decision at the cost-optimal threshold
established in step 6 of the project (0.91).

Example request body:
{
    "log_amount": 4.5,
    "log_amount_to_avg_ratio": 0.69,
    "is_low_history": 0,
    "log_seconds_since_prev_txn": 10.2,
    "origin_prior_txn_count": 12,
    "is_new_destination": 1,
    "txns_prior_1h": 0,
    "txns_prior_24h": 1,
    "txns_prior_7d": 3,
    "amount_prior_24h": 150.0,
    "hour_sin": 0.5,
    "hour_cos": -0.86,
    "dow_sin": 0.0,
    "dow_cos": 1.0,
    "txn_CASH_IN": 0, "txn_CASH_OUT": 0, "txn_DEBIT": 0, "txn_PAYMENT": 0, "txn_TRANSFER": 1,
    "acct_agent": 0, "acct_customer": 1, "acct_merchant": 0
}

The model and scaler are loaded once at cold start (module scope) and
reused across warm invocations -- this is the standard pattern for
avoiding a multi-second model-load penalty on every single request.
"""

import json
import logging
import os

import azure.functions as func
import numpy as np

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

COST_OPTIMAL_THRESHOLD = 0.91
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.pt")

# --- Lazy model loading ---
# IMPORTANT: we deliberately do NOT load torch/the model at module import
# time. Azure's Python worker "indexes" this file (imports it) during cold
# start to discover routes -- if that import is slow (torch + a model load
# can take 10-30+ seconds), indexing can time out silently, leaving zero
# functions registered even though deployment reports success. Instead,
# the model loads on the FIRST actual request, and stays cached in memory
# for subsequent warm invocations on the same worker instance.
_model = None
_feature_cols = None
_scale_cols = None
_scale_idx = None
_scaler_mean = None
_scaler_scale = None


def _ensure_model_loaded():
    global _model, _feature_cols, _scale_cols, _scale_idx, _scaler_mean, _scaler_scale
    if _model is not None:
        return  # already loaded on this warm instance

    import torch
    from model_def import FraudNet

    logging.info("Loading fraud model (first request on this instance)...")
    checkpoint = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
    _feature_cols = checkpoint["feature_cols"]
    _scale_cols = checkpoint["scale_cols"]
    _scale_idx = [_feature_cols.index(c) for c in _scale_cols]
    _scaler_mean = checkpoint["scaler_mean"]
    _scaler_scale = checkpoint["scaler_scale"]

    _model = FraudNet(n_features=len(_feature_cols))
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()
    logging.info(f"Model loaded. Expecting {len(_feature_cols)} features.")


def _score_transaction(payload: dict) -> dict:
    """Core scoring logic, separated from the HTTP wrapper so it can be
    unit-tested directly without spinning up the Azure Functions runtime."""
    import torch  # local import -- torch is only needed once we actually score

    _ensure_model_loaded()

    missing = [c for c in _feature_cols if c not in payload]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    x = np.array([[float(payload[c]) for c in _feature_cols]], dtype=np.float32)
    x[:, _scale_idx] = (x[:, _scale_idx] - _scaler_mean) / _scaler_scale

    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        prob = torch.sigmoid(_model(x_t)).item()

    return {
        "fraud_probability": round(prob, 6),
        "flagged": bool(prob >= COST_OPTIMAL_THRESHOLD),
        "threshold_used": COST_OPTIMAL_THRESHOLD,
    }


@app.route(route="score", methods=["POST"])
def score(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Fraud scoring request received.")

    try:
        payload = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Request body must be valid JSON"}),
            status_code=400,
            mimetype="application/json",
        )

    try:
        result = _score_transaction(payload)
    except ValueError as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json",
        )
    except Exception as e:
        logging.exception("Unexpected scoring error")
        return func.HttpResponse(
            json.dumps({"error": "Internal scoring error", "detail": str(e)}),
            status_code=500,
            mimetype="application/json",
        )

    return func.HttpResponse(
        json.dumps(result),
        status_code=200,
        mimetype="application/json",
    )
