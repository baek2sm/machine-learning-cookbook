#-*- coding:utf-8 -*-
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np

# 사이킷런에서 붓꽃 품종 데이터 세트를 가져옵니다.
dataset = datasets.load_iris()

# 입력 데이터와 타깃을 준비합니다.
X, y = dataset['data'], dataset['target']

# 데이터 세트를 학습 세트와 테스트 세트로 분할합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 데이터 스케일 조정을 위해 표준화를 적용합니다.
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 선형 서포트 벡터 머신 모델 객체를 만듭니다.
classifier = LinearSVC(C=1.0)

# 그리드 서치를 정의합니다.
param_grid = [{'C': np.arange(0.1, 1.1, 0.1)}]
grid = GridSearchCV(estimator=classifier, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)

# 그리드 서치를 사용해서 학습 세트의 입력 데이터와 타깃을 입력하고 학습시킵니다.
grid.fit(X_train_std, y_train)

# 학습된 최적의 하이퍼 파라미터 값을 확인합니다.
print('최적의 하이퍼 파라미터: {}'.format(grid.best_params_))

# 학습 세트에서의 모델 정확도와 테스트 세트에서의 모델 정확도를 계산합니다.
train_score = grid.score(X_train_std, y_train)
test_score = grid.score(X_test_std, y_test)

# 학습 세트에서의 정확도와 테스트 세트에서의 정확도를 출력합니다.
print('학습 세트 정확도: {score:.3f}'.format(score=train_score))
print('테스트 세트 정확도: {score:.3f}'.format(score=test_score))
