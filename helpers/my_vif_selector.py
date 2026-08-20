import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
from pandas import DataFrame
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# --------------------------------------------------------
# VIF 기반 다중공선성 제거기
# --------------------------------------------------------

def _is_dataframe(obj):
    """
    객체가 pandas DataFrame인지 확인하는 헬퍼 함수
    네임스페이스 불일치 문제를 회피하기 위해 클래스명으로 확인
    """
    # 방법 1: 클래스 이름으로 확인 (네임스페이스 불일치 해결)
    if type(obj).__name__ == 'DataFrame':
        return True
    
    # 방법 2: 폴백 - 덕 타이핑 (DataFrame의 핵심 속성 확인)
    return (hasattr(obj, 'columns') and 
            hasattr(obj, 'index') and 
            hasattr(obj, 'values') and
            hasattr(obj, 'dropna'))


class VIFSelector(BaseEstimator, TransformerMixin):
    """
    VIF(Variance Inflation Factor) 기반 다중공선성 제거기

    fit 단계에서 결측치를 제거한 후 VIF 임계값을 초과하는 변수를 제거합니다.
    반복적으로 VIF가 가장 높은 변수부터 제거합니다.

    Args:
        threshold (float): VIF 임계값 (기본값: 10.0)
        check_cols (list or None): VIF 계산에 사용할 열 목록 (기본값: None, 모든 열 사용)

    Attributes:
        drop_cols_ (list): 제거된 열 목록
        vif_cols_ (list): VIF 계산에 사용된 열 목록
        feature_names_in_ (list): fit 시점의 입력 피처 이름

    Example:
        >>> from helpers.my_vif_selector import VIFSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 4, 6], 'c': [1, 1, 1]})
        >>> vif = VIFSelector(threshold=5.0)
        >>> vif.fit(X)
        >>> X_transformed = vif.transform(X)
    """

    def __init__(self, threshold=10.0, check_cols=None):
        self.threshold = threshold
        self.check_cols = check_cols

    def _compute_vifs(self, X: DataFrame):
        exog = sm.add_constant(X, prepend=False)

        vifs = {}
        
        # divide by zero 경고 억제 (완벽한 선형관계로 인한 R²=1 경우)
        with np.errstate(divide='ignore', invalid='ignore'):
            for i, col in enumerate(X.columns):
                try:
                    vif_value = float(variance_inflation_factor(exog.values, i))
                    # VIF이 inf인 경우 큰 값으로 제한
                    vifs[col] = min(vif_value, 1e6) if np.isfinite(vif_value) else 1e6
                except Exception:
                    vifs[col] = 1e6

        vdf = DataFrame(vifs.items(), columns=["Variable", "VIF"])
        return vdf.sort_values("VIF", ascending=False)

    def fit(self, X, y=None):
        # 입력 검증 및 변환 (ndarray -> DataFrame 자동 변환)
        if not _is_dataframe(X):
            # ndarray인 경우 DataFrame으로 변환
            if isinstance(X, np.ndarray):
                # 2D 배열인 경우만 처리
                if X.ndim == 2:
                    # 컬럼명이 없으므로 숫자로 생성
                    X = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
                else:
                    raise ValueError("X must be 2-dimensional")
            else:
                raise TypeError(f"X must be a pandas DataFrame or numpy ndarray, got {type(X).__name__}")
        
        # 결측치 제거 (fit 단계에서만)
        X_clean = X.dropna()
        
        if X_clean.empty:
            raise ValueError("No data left after removing NaN values")
        
        # sklearn 표준: 피처 이름 저장
        self.feature_names_in_ = X.columns.tolist()
        
        # VIF 계산에 사용할 열 설정
        self.vif_cols_ = self.check_cols if self.check_cols else self.feature_names_in_
        
        X_vif = X_clean[self.vif_cols_].copy()

        self.drop_cols_ = []
        i = 0

        while True:
            if X_vif.shape[1] == 0:
                break

            vdf = self._compute_vifs(X_vif)
            max_vif = vdf.iloc[0]["VIF"]
            max_col = vdf.iloc[0]["Variable"]

            if max_vif <= self.threshold:
                break

            X_vif = X_vif.drop(columns=[max_col])
            self.drop_cols_.append(max_col)
            i += 1

        return self

    def transform(self, X):
        # fit 여부 확인
        check_is_fitted(self, ['drop_cols_', 'feature_names_in_'])
        
        # 입력 검증 및 변환 (ndarray -> DataFrame 자동 변환)
        if not _is_dataframe(X):
            if isinstance(X, np.ndarray):
                if X.ndim == 2:
                    X = pd.DataFrame(X, columns=self.feature_names_in_)
                else:
                    raise ValueError("X must be 2-dimensional")
            else:
                raise TypeError(f"X must be a pandas DataFrame or numpy ndarray, got {type(X).__name__}")
        
        # 피처 일치성 확인
        if X.columns.tolist() != self.feature_names_in_:
            raise ValueError(
                f"X columns {X.columns.tolist()} do not match "
                f"training columns {self.feature_names_in_}"
            )

        return X.drop(columns=self.drop_cols_)

    def get_feature_names_out(self, input_features=None):
        """drop_cols_ 를 제외한 남은 피처 이름을 반환 (set_output(transform='pandas') 지원용)."""
        check_is_fitted(self, ['drop_cols_', 'feature_names_in_'])
        kept = [c for c in self.feature_names_in_ if c not in self.drop_cols_]
        return np.asarray(kept)