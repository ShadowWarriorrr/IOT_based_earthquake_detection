import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import xgboost as xgb
data = pd.read_csv("earthquake_dataset_heritage_master.csv")

feature_cols = [
    "lat","lon","epic_dist_km","pga_g","pgv_cms","pgd_cm",
    "sa_0p3_g","sa_1p0_g","sa_3p0_g","struct_apk_g"
]

for c in feature_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

X = data[feature_cols]

y = (
    data["label_damage_state"]
        .astype(str)
        .str.strip()
        .replace({"nan": np.nan})
        .fillna("None")
        .replace({"Severe": "High"})
)

preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

base_estimators = [
    ("rf", RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=42
    )),
    ("lr", make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, multi_class="multinomial", solver="lbfgs", class_weight="balanced", random_state=42)
    )),
    ("xgb", xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softmax", eval_metric="mlogloss",
        random_state=42, n_jobs=-1
    ))
]

stack_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, multi_class="multinomial", solver="lbfgs", class_weight="balanced", random_state=42)
    ),
    stack_method="predict_proba",
    cv=5,
    n_jobs=-1
)

model = Pipeline([
    ("prep", preprocessor),
    ("stack", stack_clf)
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model, X, y, cv=skf)

acc = accuracy_score(y, y_pred_cv)
f1m = f1_score(y, y_pred_cv, average="macro")
report = classification_report(y, y_pred_cv)
cm_abs = confusion_matrix(y, y_pred_cv, labels=sorted(y.unique()))
cm_norm = confusion_matrix(y, y_pred_cv, labels=sorted(y.unique()), normalize="true")

print("\n=== Stacking Ensemble CV (RF + LR + XGBoost) ===")
print("Mean Accuracy:", acc)
print("Macro-F1:", f1m)
print(report)

model.fit(X, y)

joblib.dump(model, "damage_model_tabular_hybrid_cv.pkl")
print("✅ Saved hybrid model trained on full data: damage_model_tabular_hybrid_cv.pkl")

plt.figure(figsize=(7,5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix - Hybrid Stacking (CV)")
plt.tight_layout()
plt.savefig("stack_confusion_matrix_norm_cv.png", dpi=160)
plt.show()

report_dict = classification_report(y, y_pred_cv, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

keep_cols = ["precision", "recall", "f1-score", "support"]
report_df = report_df.loc[:, [c for c in keep_cols if c in report_df.columns]].round(4)

if "accuracy" in report_dict:
    acc_val = report_dict["accuracy"]
    report_df.loc["overall_accuracy", ["precision", "recall", "f1-score", "support"]] = [
        acc_val, acc_val, acc_val, report_df["support"].sum()
    ]

report_df.to_csv("results_hybrid.csv", index=True)
print("✅ Saved results CSV: results_hybrid.csv")
