import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

def select_features_by_importance(X, y, top_k=7):
    """用LightGBM的重要性做一次粗筛，只保留top_k个特征。"""
    model = LGBMClassifier(
        n_estimators=100, learning_rate=0.03, max_depth=3,
        num_leaves=7, min_child_samples=50, random_state=42, verbosity=-1,
    )
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False)
    selected = imp.head(top_k).index.tolist()
    print(f"[筛选] 保留特征：{selected}")
    return selected
