"""
Canonical Model Registry for Solar & Electrical Power Fault Detection.
Single source of truth for:
- Model identifiers
- Serialized filenames
- Scaling requirements
- Factory initializers
Ensures guaranteed compatibility across train.py, train_full_1_8m.py, evaluate.py, and app/dashboard_server.py.
"""
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier

MODEL_REGISTRY = {
    "hist_gradient_boosting": {
        "name": "Hist Gradient Boosting",
        "filename": "hist_gradient_boosting.pkl",
        "needs_scaling": False,
        "factory": lambda random_state: HistGradientBoostingClassifier(
            max_iter=150, max_depth=12, min_samples_leaf=35,
            class_weight="balanced", random_state=random_state
        )
    },
    "random_forest": {
        "name": "Random Forest",
        "filename": "random_forest.pkl",
        "needs_scaling": False,
        "factory": lambda random_state: RandomForestClassifier(
            n_estimators=120, max_depth=16, min_samples_split=4, max_samples=0.25,
            class_weight="balanced_subsample", n_jobs=-1, random_state=random_state
        )
    },
    "extra_trees": {
        "name": "Extra Trees",
        "filename": "extra_trees.pkl",
        "needs_scaling": False,
        "factory": lambda random_state: ExtraTreesClassifier(
            n_estimators=120, max_depth=16, min_samples_split=4, max_samples=0.25,
            bootstrap=True, class_weight="balanced", n_jobs=-1, random_state=random_state
        )
    },
    "calibrated_linear_sgd": {
        "name": "Calibrated Linear SGD",
        "filename": "calibrated_linear_sgd.pkl",
        "needs_scaling": True,
        "factory": lambda random_state: SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1500,
            class_weight="balanced", random_state=random_state
        )
    },
    "k_nearest_neighbors": {
        "name": "K-Nearest Neighbors",
        "filename": "k_nearest_neighbors.pkl",
        "needs_scaling": True,
        "factory": lambda random_state: KNeighborsClassifier(
            n_neighbors=5, weights="distance", algorithm="kd_tree", n_jobs=-1
        )
    }
}

META_LEARNER_REGISTRY = {
    "logistic_regression_balanced": {
        "name": "Logistic Regression (L2, Balanced, C=0.5)",
        "factory": lambda random_state: LogisticRegression(
            C=0.5, penalty="l2", class_weight="balanced", max_iter=2000, random_state=random_state
        )
    },
    "ridge_balanced": {
        "name": "Ridge Classifier (Balanced, alpha=1.0)",
        "factory": lambda random_state: RidgeClassifier(
            alpha=1.0, class_weight="balanced", random_state=random_state
        )
    }
}
