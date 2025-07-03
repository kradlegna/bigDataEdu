# Breast Cancer 데이터 분석 및 모델링 실습

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

# 1. 데이터 로딩 및 요약 분석
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names

# 1-a. 샘플 수 및 특성 수 확인
print("전체 샘플 수:", X.shape[0])
print("특성 수:", X.shape[1])

# 1-b. 라벨 분포 확인 (0: malignant, 1: benign)
unique, counts = np.unique(y, return_counts=True)
print("Target 분포:", dict(zip(unique, counts)))

# 1-c. 결측값 확인
if np.isnan(X).any():
    print("결측값 존재")
else:
    print("결측값 없음")

# 1-d. mean 관련 feature 요약 (예: mean radius)
radius_idx = list(feature_names).index("mean radius")
radius = X[:, radius_idx]
print("mean radius 평균:", np.mean(radius))
print("mean radius 최소값:", np.min(radius))
print("mean radius 최대값:", np.max(radius))

# 2. 데이터 시각화
texture_idx = list(feature_names).index("mean texture")
texture = X[:, texture_idx]

# 2-a. 히스토그램 시각화
plt.hist(radius, bins=20)
plt.title("Distribution of Mean Radius")
plt.xlabel("Radius")
plt.ylabel("Frequency")
plt.show()

plt.hist(texture, bins=20)
plt.title("Distribution of Mean Texture")
plt.xlabel("Texture")
plt.ylabel("Frequency")
plt.show()

# 2-b. 산점도 (target별 색상)
plt.scatter(radius, texture, c=y, cmap="coolwarm", alpha=0.7)
plt.title("Mean Radius vs Mean Texture")
plt.xlabel("Mean Radius")
plt.ylabel("Mean Texture")
plt.colorbar(label="Target (0=malignant, 1=benign)")
plt.show()

# 2-c. mean area의 라벨별 분포 (박스플롯)
area_idx = list(feature_names).index("mean area")
area = X[:, area_idx]
plt.boxplot([area[y == 1], area[y == 0]], labels=["benign", "malignant"])
plt.title("Distribution of Mean Area by Class")
plt.ylabel("Mean Area")
plt.show()

# 3. 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=77
)
print("학습 데이터 수:", len(X_train))
print("테스트 데이터 수:", len(X_test))

# 4. 기본 모델 학습 및 평가
base_clf = RandomForestClassifier(random_state=77, n_estimators=100)
base_clf.fit(X_train, y_train)
y_pred = base_clf.predict(X_test)
y_proba = base_clf.predict_proba(X_test)[:, 1]

# ROC 곡선 및 AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

# 정확도 및 리포트
print("정확도:", accuracy_score(y_test, y_pred))
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred))
print("분류 리포트:\n", classification_report(y_test, y_pred))

# 5. 하이퍼파라미터 튜닝 (GridSearchCV)
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
}
grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=77),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)
grid.fit(X_train, y_train)
print("최적 하이퍼파라미터 (Grid):", grid.best_params_)
print("최고 CV 정확도 (Grid):", grid.best_score_)

# 6. 하이퍼파라미터 튜닝 (RandomizedSearchCV)
param_dist = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=77),
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
    random_state=42,
)
random_search.fit(X_train, y_train)
print("최적 하이퍼파라미터 (Randomized):", random_search.best_params_)
print("최고 CV 정확도 (Randomized):", random_search.best_score_)

# 7. 튜닝 전/후 모델 성능 비교
print("=== 튜닝 전 모델 ===")
y_pred_base = base_clf.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred_base))
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred_base))
print("분류 리포트:\n", classification_report(y_test, y_pred_base))

print("=== 튜닝 후 모델 (GridSearchCV 기준) ===")
best_clf = grid.best_estimator_
y_pred_best = best_clf.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred_best))
print("혼동 행렬:\n", confusion_matrix(y_test, y_pred_best))
print("분류 리포트:\n", classification_report(y_test, y_pred_best))

# 8. ROC curve 시각화
plt.plot(
    *roc_curve(y_test, base_clf.predict_proba(X_test)[:, 1])[:2], label="Base Model"
)
plt.plot(
    *roc_curve(y_test, best_clf.predict_proba(X_test)[:, 1])[:2], label="Tuned Model"
)
plt.plot([0, 1], [0, 1], "--", label="Random")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.show()

# 9. Feature Importance 시각화
importances = best_clf.feature_importances_
indices = np.argsort(importances)[-10:][::-1]
labels = [feature_names[i] for i in indices]
values = importances[indices]

plt.figure(figsize=(8, 5))
plt.barh(labels[::-1], values[::-1])
plt.xlabel("Importance")
plt.title("Top 10 Important Features")
plt.show()
