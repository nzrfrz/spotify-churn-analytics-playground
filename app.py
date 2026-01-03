from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import joblib
import pandas as pd
import numpy as np
import traceback

app = Flask(__name__, static_folder=None)
CORS(app)

# helper loader: prefer non-FE artifacts, fallback to FE

numeric_label = [
  'listening_time',
  'ads_listened_per_week',
  'skip_rate',
  'songs_played_per_day',
  'age'
]

skip_edges = [0.0, 0.3, 0.6, 1.0]
skip_labels = ['low_skip', 'medium_skip', 'high_skip']

age_bins = [-1, 20, 30, 45, 200]
age_labels = ['teen', 'young_adult', 'adult', 'senior']


def try_load_candidate(names_files):
  for f in names_files:
    try:
      if os.path.exists(f):
        return joblib.load(f)
    except Exception:
      continue
  return None


def load_preproc_and_model():
  preproc = try_load_candidate([
      "./preprocess_rfc_feature_construction.joblib",
  ])
  model = try_load_candidate([
      "./model_rfc_feature_construction.joblib",
  ])
  return preproc, model

# Small rule-based recommender that maps top contributing features -> suggestions


def build_recommendations(contributors, raw_payload):
  _recommendations = []
  _reasons = []

  # look at top positive contributors
  positive_contributors = [c for c in contributors if c['contribution'] > 0]
  positive_contributors_sorted = sorted(
      positive_contributors, key=lambda x: x['contribution'], reverse=True)

  # helper checks
  def has_feature_keywords(keywords):
    for c in positive_contributors_sorted:
      for keyword in keywords:
        if keyword in c['feature'].lower():
          return c
    return None

  # skip_rate
  if has_feature_keywords(['skip_rate']):
    _reasons.append('high skip rate')
    _recommendations.append(
        "Personalize playlists and recommend similar artists (reduce skipping).")

  # ads
  if has_feature_keywords(['ads_listened', 'ads_listened_per_week', 'ads']):
    _reasons.append("many ads listened")
    sub_type = (raw_payload.get("subscription_type") or "").lower()
    if sub_type == 'free':
      _recommendations.append(
          'Offer short Premium trial (7 days) or a one-time ad-free day.')
    else:
      _recommendations.append(
          'Consider promo or explain Premium benefits (less ads).')

  # listening time low (we check raw payload)
  _listening_time = raw_payload.get(
      'listening_time_minutes') or raw_payload.get('listening_time')
  if _listening_time is not None:
    try:
      if float(_listening_time) < 100:
        _reasons.append('low listening time')
        _recommendations.append(
            'Send re-engagement message + curated "Welcome Back" playlist.')
    except Exception:
      pass

  # songs per day low
  _songs_played_per_day = raw_payload.get('songs_played_per_day')
  if _songs_played_per_day is not None:
    try:
      if int(_songs_played_per_day) < 20:
        _reasons.append('low songs played per day')
        _recommendations.append('Promote trending playlists / local charts.')
    except Exception:
      pass

  # subscription free vs premium
  if raw_payload.get('subscription_type', '').lower() == 'free':
    _recommendations.append(
        'Target with conversion offer for Free users (trial or discount).')

  # If nothing flagged from coefficients, fallback to heuristic based on raw payload
  if not _recommendations:
    # e.g. if user free and skip_rate high or ads many, give conversion offer
    if ((raw_payload.get('subscription_type') or '').lower() == 'free'
        or float(raw_payload.get('skip_rate') or 0) > 0.5
            or int(raw_payload.get('ads_listened_per_week') or 0) > 10):
      _recommendations.append(
          'Offer trial / promo for Free users to reduce churn risk.')
    else:
      _recommendations.append(
          'Send light re-engagement (in-app recommendation) and monitor.')

  # deduplicate while preserving order
  seen = set()
  clean_recs = []
  for r in _recommendations:
    if r not in seen:
      clean_recs.append(r)
      seen.add(r)

  seen_reasons = set()
  clean_reasons = []
  for r in _reasons:
    if r not in seen_reasons:
      clean_reasons.append(r)
      seen_reasons.add(r)

  return clean_reasons, clean_recs


def apply_feature_engineering(df_in, medians, ads_median_val, listening_q1_val):
  df_fe = df_in.copy()

  # numeric impute
  for c in numeric_label:
    if c in df_fe.columns:
      df_fe[c] = pd.to_numeric(
          df_fe[c], errors="coerce").fillna(medians.get(c, 0))

  listening = df_fe.get("listening_time", 0)
  songs = df_fe.get("songs_played_per_day", 0)
  skip = df_fe.get("skip_rate", 0)

  df_fe["engagement_score"] = listening + songs + (1 - skip) * 10

  ads = df_fe.get("ads_listened_per_week", 0)
  df_fe["is_ad_heavy"] = (ads > ads_median_val).astype(int)

  df_fe["low_activity"] = (listening < listening_q1_val).astype(int)

  df_fe["skip_rate_bucket"] = pd.cut(
      skip,
      bins=skip_edges,
      labels=skip_labels,
      include_lowest=True
  ).astype(str)

  subs = df_fe.get("subscription_type", "")
  df_fe["is_premium_like"] = subs.isin(
      ["Premium", "Family", "Student"]).astype(int)

  age = df_fe.get("age", -1)
  df_fe["age_group"] = pd.cut(
      age,
      bins=age_bins,
      labels=age_labels,
      include_lowest=True
  ).astype(str)

  offline = df_fe.get("offline_listening", 0)
  df_fe["engagement_offline_interaction"] = df_fe["engagement_score"] * offline

  return df_fe


@app.route("/predict", methods=["POST"])
def predict():
  if not request.is_json:
    return jsonify({"error": "Expecting application/json"}), 400
  payload = request.get_json()
  app.logger.info("Received payload: %s", payload)

  preproc, model = load_preproc_and_model()
  if preproc is None or model is None:
    return jsonify({"error": "Model or preprocessor not found on server."}), 500

  try:
    # Build DataFrame keys must match training preprocessor names
    df = pd.DataFrame([{
        'gender': payload.get('gender') or '',
        'age': int(payload.get('age') or 0),
        'country': payload.get('country') or '',
        'subscription_type': payload.get('subscription_type') or '',
        'listening_time': float(payload.get('listening_time_minutes') or payload.get('listening_time') or 0.0),
        'songs_played_per_day': float(payload.get('songs_played_per_day') or 0.0),
        'skip_rate': float(payload.get('skip_rate') or 0.0),
        'device_type': payload.get('device_type') or '',
        'ads_listened_per_week': float(payload.get('ads_listened_per_week') or 0.0),
        'offline_listening': int(payload.get('offline_listening') or 0),
    }])

    df_fe = apply_feature_engineering(
      df,
      medians=preproc.named_steps["feature_construction"].medians
      if hasattr(preproc, "named_steps") else {},
      ads_median_val=preproc.named_steps["feature_construction"].ads_median
      if hasattr(preproc, "named_steps") else 0,
      listening_q1_val=preproc.named_steps["feature_construction"].listening_q1
      if hasattr(preproc, "named_steps") else 0
    )
    # transform
    X_t = preproc.transform(df_fe)  # numpy array
    proba = float(model.predict_proba(X_t)[0][1])
    pred = int(model.predict(X_t)[0])

    contributions = []
    # if linear model (has coef_), compute coef * x for each transformed feature
    if hasattr(model, 'coef_'):
      coefs = model.coef_.flatten()
      # try to get feature names
      try:
        num_names = preproc.transformers_[0][2]
        try:
          cat_names = preproc.named_transformers_[
              'cat']['onehot'].get_feature_names_out(preproc.transformers_[1][2])
          cat_names = list(cat_names)
        except Exception:
          cat_names = []
        feat_names = list(num_names) + list(cat_names)
      except Exception:
        feat_names = [f"f{i}" for i in range(X_t.shape[1])]

      x_vals = X_t.flatten().tolist()
      for name, coef, xval in zip(feat_names, coefs, x_vals):
        contributions.append({
            "feature": str(name),
            "coef": float(coef),
            "value": float(xval),
            "contribution": float(coef * xval)
        })
      # sort by contribution descending
      contributions_sorted = sorted(
          contributions, key=lambda x: x['contribution'], reverse=True)
    else:
      contributions_sorted = []

    # build recommendations from top contributions & raw payload
    reasons, recs = build_recommendations(contributions_sorted[:10], payload)
    # print(contributions_sorted)

    resp = {
        "ok": True,
        "prediction": int(pred),
        "probability": proba,
        "top_contributions": contributions_sorted[:10],
        "reasons": reasons,
        "recommendations": recs,
        "received": payload
    }
    return jsonify(resp), 200
  except Exception as e:
    app.logger.error("Error during predict: %s", traceback.format_exc())
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500


if __name__ == "__main__":
  # Run dev server
  app.run(host="127.0.0.1", port=5000, debug=True)
