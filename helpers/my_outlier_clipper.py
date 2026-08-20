import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    수치형 컬럼의 이상치를 IQR 또는 Z-score 기반 경계값으로 클리핑한다.
    학습 단계에서 컬럼별로 상/하한 경계를 통해 해당 범위 밖의 값을 경계값으로 잘라낸다.

    Args:
        method (str): 이상치 판단 방식. 'iqr' 또는 'zscore' (기본값: 'iqr')
            - 'iqr'   : [Q1 - 1.5*IQR, Q3 + 1.5*IQR] 범위로 클리핑
            - 'zscore': [mean - 3*std, mean + 3*std] 범위로 클리핑
    """

    def __init__(self, method='iqr'):
        self.method = method

    def fit(self, X, y=None):
        # 입력 피처 이름 저장 (set_output(transform='pandas') 지원용)
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.asarray(X.columns)
        else:
            self.feature_names_in_ = np.asarray(
                [f'x{i}' for i in range(np.asarray(X).shape[1])]
            )

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got ndim={X.ndim}")

        if self.method == 'iqr':
            q1 = np.nanpercentile(X, 25, axis=0)
            q3 = np.nanpercentile(X, 75, axis=0)
            iqr = q3 - q1
            self.lower_ = q1 - 1.5 * iqr
            self.upper_ = q3 + 1.5 * iqr
        elif self.method == 'zscore':
            mean = np.nanmean(X, axis=0)
            std = np.nanstd(X, axis=0)
            self.lower_ = mean - 3.0 * std
            self.upper_ = mean + 3.0 * std
        else:
            raise ValueError(
                f"method must be 'iqr' or 'zscore', got {self.method!r}"
            )

        return self

    def transform(self, X):
        check_is_fitted(self, ['lower_', 'upper_'])
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        """클리핑은 컬럼 수를 바꾸지 않으므로 입력 피처 이름을 그대로 반환."""
        check_is_fitted(self, ['feature_names_in_'])
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features)
