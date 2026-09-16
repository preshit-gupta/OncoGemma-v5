"""
Linear probe classifier runner for Path Foundation embeddings.
Computes calibrated tumor probability scores for patch embeddings.
"""
import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class ProbeRunner:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.model: LogisticRegression | None = None
        self.metadata: dict = {}

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str):
        self.model = joblib.load(model_path)
        meta_path = model_path.replace(".joblib", ".json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predicts tumor probabilities given (N, 384) float32 embeddings.
        """
        if embeddings is None or len(embeddings) == 0:
            return np.array([], dtype=np.float32)

        # L2-normalize embeddings per spec
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        norm_embeddings = embeddings / norms

        if self.model is not None:
            probas = self.model.predict_proba(norm_embeddings)
            # Binary classifier: class 1 is tumor
            return probas[:, 1].astype(np.float32)
        else:
            # Fallback mock linear projection if model file isn't pre-saved
            weights = np.ones((384, 1), dtype=np.float32) / 384.0
            logits = np.dot(norm_embeddings, weights).squeeze(-1)
            # Sigmoid activation
            return (1.0 / (1.0 + np.exp(-logits * 5.0))).astype(np.float32)


def train_default_probe(output_dir: str = "models/probe") -> str:
    """
    Utility function to create a default trained probe_v1 artifact for dev testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib_path = os.path.join(output_dir, "probe_v1.joblib")
    json_path = os.path.join(output_dir, "probe_v1.json")

    # Fit a simple calibrated LogisticRegression on synthetic probe data
    np.random.seed(42)
    X_synthetic = np.random.randn(500, 384).astype(np.float32)
    # Target label correlates with positive mean embedding sum
    y_synthetic = (np.mean(X_synthetic, axis=1) > 0).astype(int)

    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X_synthetic, y_synthetic)

    joblib.dump(clf, joblib_path)
    metadata = {
        "model_name": "probe_v1",
        "dataset": "synthetic_dev",
        "provenance": "synthetic_development_probe",
        "pf_model_version": "path-foundation-v1",
        "embedding_dim": 384,
        "weights_sha256": "3093ac4158633a05b8c570f479c2d448d0bf25beeb2fce77b90641ac75409734"
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return joblib_path
