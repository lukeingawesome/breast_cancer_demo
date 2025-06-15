"""
Train a simple medical‑tabular model on the
Breast‑Cancer Wisconsin dataset (binary classification).

⚠️  Research/education only – NOT a clinical diagnostic tool.
"""
import argparse, json, pathlib, datetime
import joblib, yaml
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from .model_utils import seed_everything

def main(cfg):
    seed_everything(cfg["seed"])

    # 1) Load data ----------------------------------------------------------------
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["seed"]
    )

    # 2) Train --------------------------------------------------------------------
    model = LogisticRegression(max_iter=cfg["max_iter"], n_jobs=-1)
    model.fit(X_train, y_train)

    # 3) Evaluate -----------------------------------------------------------------
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # 4) Persist model + metrics ---------------------------------------------------
    ts   = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out  = pathlib.Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out / f"model_{ts}.joblib")
    metrics = {"accuracy": acc, "roc_auc": auc, "timestamp_utc": ts}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 5) Emit metrics for GitHub Actions summary
    print(f"::notice file=metrics.json::accuracy={acc:.3f}, roc_auc={auc:.3f}")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yml"))
    main(cfg)
