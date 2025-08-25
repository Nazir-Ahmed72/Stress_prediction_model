import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# ================
# Load Data
# ================
st.title("⌨️ Keystroke Analysis Dashboard")
st.write("A simple dashboard for exploring keystroke behavior and classification.")

df = pd.read_csv("keystrokes_labeled.csv")
st.subheader("📂 Data Preview")
st.dataframe(df.head())

# =====================
# EDA
# =====================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

st.subheader("📊 Summary Statistics")
st.write(df[numeric_cols].describe())

st.subheader("🔢 Label Distribution")
st.bar_chart(df["label"].value_counts())

# Histograms
st.subheader("📈 Feature Distributions")
for col in ["typing_speed","errors","backspaces","mean_inter_key_interval"]:
    if col in df.columns:
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"{col} Distribution")
        st.pyplot(fig)

# =====================
# Train/Test split
# =====================
features = ["typing_speed","errors","backspaces","mean_inter_key_interval"]
X = df[features].copy()
y = df["label"].copy()

le = LabelEncoder()
y_enc = le.fit_transform(y)

min_class = np.min(np.bincount(y_enc))
stratify = y_enc if min_class >= 2 else None
test_size = 0.3 if len(df) >= 40 else 0.4

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=test_size, random_state=42, stratify=stratify
)

# =====================
# Random Forest + GridSearch
# =====================
st.subheader("🌲 Model Training & Tuning")

pipe = Pipeline([
    ("scaler", StandardScaler()), 
    ("rf", RandomForestClassifier(random_state=42))
])

param_grid = {
    "rf__n_estimators": [100, 250],
    "rf__max_depth": [None, 10],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
    "rf__max_features": ["sqrt", "log2"]
}

cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)

grid = GridSearchCV(pipe, param_grid=param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

st.write("**Best Parameters:**", grid.best_params_)
st.write("**Best CV Accuracy:**", round(grid.best_score_, 3))

# Evaluate
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"**Test Accuracy:** {acc:.3f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(le.classes_)))
ax.set_yticks(range(len(le.classes_)))
ax.set_xticklabels(le.classes_, rotation=45)
ax.set_yticklabels(le.classes_)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.colorbar(im, ax=ax)
st.pyplot(fig)

# =====================
# Feature Importances
# =====================
st.subheader("📌 Feature Importances")
rf_stage = best_model.named_steps["rf"]
fi = pd.DataFrame({
    "feature": features,
    "importance": rf_stage.feature_importances_
}).sort_values("importance", ascending=False)
st.bar_chart(fi.set_index("feature"))

# =====================
# PCA Visualization
# =====================
st.subheader("🌀 PCA Analysis")

scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X)
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

fig, ax = plt.subplots()
for cls_idx, cls_name in enumerate(le.classes_):
    mask = (y_enc == cls_idx)
    ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1], label=cls_name, alpha=0.7)
ax.set_title("PCA (2D)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
st.pyplot(fig)

# =====================
# Decision Boundary
# =====================
st.subheader("⚖️ Decision Boundary (PCA Space)")

X_train_scaled = scaler_full.fit_transform(X_train)
pca2_boundary = PCA(n_components=2).fit(X_train_scaled)
X_train_pca = pca2_boundary.transform(X_train_scaled)
X_test_scaled = scaler_full.transform(X_test)
X_test_pca = pca2_boundary.transform(X_test_scaled)

rf_2d = RandomForestClassifier(random_state=42, n_estimators=300)
rf_2d.fit(X_train_pca, y_train)

x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = rf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
for cls_idx, cls_name in enumerate(le.classes_):
    m_tr = (y_train == cls_idx)
    ax.scatter(X_train_pca[m_tr, 0], X_train_pca[m_tr, 1], label=f"Train-{cls_name}", marker='o')
    m_te = (y_test == cls_idx)
    ax.scatter(X_test_pca[m_te, 0], X_test_pca[m_te, 1], label=f"Test-{cls_name}", marker='x')
ax.set_title("RandomForest Decision Boundary (2D PCA)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
st.pyplot(fig)

st.success("✅ Dashboard ready! Explore your keystroke behavior interactively.")
