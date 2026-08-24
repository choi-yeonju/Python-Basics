# ---- 기본 참조 ----------------------------------------------------
import joblib                                   # 모델 저장
import numpy as np                              # 배열 연산
import datetime as dt                           # 작업 폴더명에 쓸 타임스탬프
from pathlib import Path                        # 경로 처리
from pandas import concat, DataFrame, Series    # 데이터프레임 처리
from IPython.display import display             # 출력 기능

from . import RANDOM_STATE                      # 재현성을 위한 랜덤시드
from . import my_prep                           # 스케일러 목록(SCALERS)
from . import my_plot                           # 시각화 참조
from .my_vif_selector import VIFSelector        # 다중공선성 제거
from .my_outlier_clipper import OutlierClipper  # 이상치 경계값 대체

# ---- 머신러닝 파이프라인 구축 관련 참조 --------------------------------
from sklearn.pipeline import Pipeline               # 전처리 + 모델 연결
from sklearn.impute import SimpleImputer            # 결측치 대체
from sklearn.preprocessing import OneHotEncoder     # 더미변수 인코딩
from sklearn.preprocessing import LabelEncoder     # 종속변수 라벨 인코딩(cls_baseline)
from sklearn.preprocessing import label_binarize   # 다중분류 지표의 일대다(OvR) 변환
from sklearn.compose import ColumnTransformer       # 연속형·명목형 분기 처리
from sklearn.decomposition import PCA               # 차원 축소
from sklearn.base import is_classifier              # 회귀·분류 과제 판별

# ---- 성능 지표 계산 -------------------------------------------------
# 회귀 지표
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_squared_log_error, mean_absolute_percentage_error
)

# 분류 지표
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, log_loss
)

# cls_overfit 이 교차검증에 쓸 scorer 를 cls_score 와 같은 정의로 만들 때 사용
from sklearn.metrics import make_scorer

# ---- 머신러닝 모델 참조 참조 -----------------------------------------
# xgboost·lightgbm·catboost 는 별도 설치가 필요한 무거운 패키지라 함수 안에서 import 한다
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 분류 모델 (cls_baseline)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ---- 과적합 판정 ----------------------------------------------------
# reg_overfit : 교차검증 점수와 학습곡선
from sklearn.model_selection import cross_validate, learning_curve as sk_learning_curve

# ---- XAI -----------------------------------------------------------
# SHAP 이 그린 그래프의 축 라벨 조정을 위한 참조 --> SHAP 그래프는 my_plot으로 제어할 수 없다.
from matplotlib import pyplot as plt



# --------------------------------------------------------
# 학습된 모델을 joblib 으로 직렬화해 저장
# --------------------------------------------------------
def save_model(model, save_path):
    """학습된 모델을 joblib 으로 직렬화해 저장한다. 상위 디렉토리는 자동 생성.

    Args:
        model: 저장할 sklearn 모델/파이프라인
        save_path (str | Path): 저장 경로 (.pkl 권장)

    Returns:
        Path: 저장된 파일의 절대 경로.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)

    # 절대경로로 변환하여 리턴
    return save_path.resolve()


# --------------------------------------------------------
# 저장된 모델 파일을 로드해서 반환
# --------------------------------------------------------
def load_model(load_path):
    """저장된 모델 파일을 로드해서 반환한다.

    Args:
        load_path (str | Path): 모델 파일 경로

    Returns:
        저장 시점의 모델 객체.

    Raises:
        FileNotFoundError: 파일이 없는 경우.
    """
    # 경로 문자열을 경로 객체로 변환
    load_path = Path(load_path)

    # 파일이 없으면 예외 발생
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")

    # joblib 로 모델 로드후 반환
    return joblib.load(load_path)


# --------------------------------------------------------
# 전처리 파이프라인 + 모델 학습
# --------------------------------------------------------
def fit_pipeline(model, x_train, y_train, nominal_cols=None, *,
                 # --- 1) 결측치 대체 ---
                 impute=False,                        # 결측치 대체 수행 여부
                 numeric_impute='median',             # 연속형 대체 전략
                 categorical_impute='most_frequent',  # 명목형 대체 전략
                 # --- 2) 이상치 대체 (경계값 클리핑, 행 삭제 없음) ---
                 outlier=False,                       # 이상치 대체 수행 여부
                 outlier_method='iqr',                # 이상치 판단 방식 (iqr / zscore)
                 # --- 3) 다중공선성 제거 (VIF) ---
                 vif=False,                           # 다중공선성 제거 수행 여부
                 vif_threshold=10.0,                  # VIF 임계값
                 # --- 4) 정규화 ---
                 scale=False,                         # 정규화 수행 여부
                 scale_method='standard',             # 사용할 스케일러 이름 (standard / minmax / robust / maxabs)
                 # --- 5) 차원 축소 (PCA) ---
                 pca=False,                           # 차원 축소 수행 여부
                 pca_variance=0.95,                   # 유지할 누적 설명분산 비율
                 # --- 6) 더미변수 인코딩 ---
                 encode=True,                         # 더미변수 인코딩 수행 여부
                 drop_first=False,                    # 첫 번째 더미 제거 여부 (더미 트랩 방지)
                 # --- 기타 ---
                 name=None,                           # 모델을 구분할 이름. 결과 객체의 `name_` 속성이 된다
                 save_path=None,                      # 학습이 끝난 모델의 저장 경로 (.pkl)
                 verbose=True,                        # 단계별 전처리 내역 출력 여부
                 **fit_params):                       # 모델의 fit 에 그대로 넘길 인자 (예: model__cat_features)
    """전처리 단계를 쌓아 모델까지 연결한 뒤, 훈련 데이터로 학습해서 반환한다.

    Args:
        model: 파이프라인 끝에 연결할 사이킷런 모델.
        x_train (DataFrame): 훈련 데이터의 독립변수.
        y_train (Series): 훈련 데이터의 종속변수.
        nominal_cols (list): 명목형 컬럼명. None 이면 자동 선택 (기본값: None).
        impute (bool): 결측치 대체 여부 (기본값: False).
        numeric_impute (str): 연속형 대체 전략 — mean/median/most_frequent/constant (기본값: 'median').
        categorical_impute (str): 명목형 대체 전략 — most_frequent/constant (기본값: 'most_frequent').
        outlier (bool): 이상치를 경계값으로 대체할지 여부 (기본값: False).
        outlier_method (str): 이상치 판단 방식 — iqr/zscore (기본값: 'iqr').
        vif (bool): 다중공선성 제거 여부 (기본값: False).
        vif_threshold (float): VIF 임계값 (기본값: 10.0).
        scale (bool): 정규화 여부 (기본값: False).
        scale_method (str): 스케일러 이름 — standard/minmax/robust/maxabs (기본값: 'standard').
        pca (bool): 차원 축소 여부 (기본값: False).
        pca_variance (float): 유지할 누적 설명분산 비율 (기본값: 0.95).
        encode (bool): 더미변수 인코딩 여부. False 면 명목형을 원본 그대로 넘긴다 (기본값: True).
        drop_first (bool): 첫 번째 더미 제거 여부 (기본값: False).
        name (str): 모델을 구분할 이름 (기본값: None).
        save_path (str): 학습된 모델의 저장 경로(.pkl) (기본값: None).
        verbose (bool): 전처리 내역 출력 여부 (기본값: True).
        **fit_params: 모델의 fit 에 넘길 인자. `단계명__인자명` 형식 (예: model__cat_features).

    Returns:
        Pipeline: 학습된 파이프라인. 전처리 구성을 담은 `pipeline_info_` 와 이름 `name_` 속성이 붙는다.

    Raises:
        KeyError: nominal_cols 에 x_train 이 갖고 있지 않은 컬럼이 있는 경우.
        ValueError: 스케일러 이름이 유효하지 않은 경우.
    """
    # --- 1) 명목형 컬럼 확정 ---
    # 지정이 없으면 category/object 타입을 자동으로 선택한다
    if nominal_cols is None:
        nominal_cols = list(x_train.select_dtypes(include=['category', 'object']).columns)
    else:
        missing = []
        for c in nominal_cols:
            if c not in x_train.columns:
                missing.append(c)

        if missing:
            raise KeyError(f'x_train 에 존재하지 않는 컬럼입니다: {missing}')

        nominal_cols = list(nominal_cols)

    # --- 2) 연속형 컬럼 확정 ---
    # 수치형 중에서 명목형으로 지정된 것을 뺀 나머지.
    # 이상치대체·정규화·다중공선성·차원축소의 대상이 된다
    continuous = []
    for c in x_train.select_dtypes(include='number').columns:
        if c not in nominal_cols:
            continuous.append(c)

    # --- 3) 대상 요약 출력 ---
    if name:
        model_name = name
    else:
        model_name = model.__class__.__name__.lower()
        model_name = model_name.removesuffix("regressor")
        model_name = model_name.removesuffix("regression")
        model_name = model_name.removesuffix("classifier")

    if verbose:
        print(f'대상: {x_train.shape[0]}행 x {x_train.shape[1]}열 | 모델: {model_name}')
        print(f'명목형: {nominal_cols}')
        print(f'연속형: {continuous}')

        # impute 를 끈 채 결측치가 남아 있으면 대부분의 모델이 학습 도중 멈춘다.
        # 다만 XGBoost·LightGBM 처럼 결측치를 자체 처리하는 모델도 있으므로 안내만 한다
        if not impute:
            na_cols = x_train.columns[x_train.isna().any()].tolist()

            if na_cols:
                print(f'참고: 결측치가 있는 컬럼 {na_cols} | '
                      f'모델이 결측치를 직접 다루지 못하면 impute=True 로 설정하세요.')

    # --- 4) 연속형 전처리 단계 구성 ---
    numeric_steps = []      # 연속형 변수의 처리 순서를 저장할 리스트
    numeric_report = []     # 전처리 단계별 설명을 저장할 리스트

    if continuous and impute:
        # 처리 단계의 이름(imputer)과 수행할 클래스(SimpleImputer)를 지정한다.
        # SimpleImputer는 sklearn에서 제공하는 결측치 처리 클래스
        numeric_steps.append(('imputer', SimpleImputer(strategy=numeric_impute)))
        numeric_report.append(f'결측치 대체({numeric_impute})')

    if continuous and outlier:
        # 이상치 대체 단계의 이름(outlier_clipper)과 수행할 클래스(OutlierClipper)를 지정한다.
        # OutlierClipper는 my_outlier_clipper.py에서 정의한 이상치 처리 클래스
        numeric_steps.append(('outlier_clipper', OutlierClipper(method=outlier_method)))
        numeric_report.append(f'이상치 대체({outlier_method})')

    if continuous and vif:
        # 다중공선성 제거 단계의 이름(vif_selector)과 수행할 클래스(VIFSelector)를 지정한다.
        # VIFSelector는 my_vif_selector.py에서 정의한 다중공선성 제거 클래스
        numeric_steps.append(('vif_selector', VIFSelector(threshold=vif_threshold)))
        numeric_report.append(f'다중공선성 제거(VIF >= {vif_threshold})')

    if continuous and scale:
        # 스케일러 이름은 my_prep.scaling() 과 같은 표기를 받는다 ('StandardScaler' -> 'standard')
        scale_name = scale_method.lower().replace('scaler', '').strip()

        # 오타를 냈을 때 KeyError 대신 사용 가능한 이름을 알려준다
        if scale_name not in my_prep.SCALERS:
            raise ValueError(f"지원하지 않는 스케일러입니다: '{scale_method}' "
                             f"(사용 가능: {list(my_prep.SCALERS.keys())})")

        # 연속형 전처리 단계에 스케일러를 추가한다.
        # my_prep.SCALERS[scale_name]()는 해당 스케일러 클래스의 인스턴스를 생성한다.
        # 예: my_prep.SCALERS['standard']()는 StandardScaler()를 반환한다.
        numeric_steps.append(('scaler', my_prep.SCALERS[scale_name]()))
        numeric_report.append(f'정규화({scale_name})')

    if continuous and pca:
        # 차원 축소 단계의 이름(pca)과 수행할 클래스(PCA)를 지정한다.
        # PCA는 sklearn에서 제공하는 차원 축소 클래스
        numeric_steps.append(('pca', PCA(n_components=pca_variance, random_state=RANDOM_STATE)))
        numeric_report.append(f'차원 축소(설명분산 {pca_variance})')

    # --- 5) 명목형 전처리 단계 구성 ---
    categorical_steps = []
    categorical_report = []

    if nominal_cols and impute:
        categorical_steps.append(('imputer', SimpleImputer(strategy=categorical_impute)))
        categorical_report.append(f'결측치 대체({categorical_impute})')

    if nominal_cols and encode:
        categorical_steps.append(('onehot', OneHotEncoder(
            # 학습 알고리즘 유형에 따라 drop_first가 선택적으로 수행되어야 한다.
            drop='first' if drop_first else None,
            # handle_unknown='ignore' 는 훈련 때 없던 범주가 검증 데이터에 나타나도
            # 예외 대신 전부 0인 더미로 처리해, 예측이 중간에 끊기지 않게 한다
            handle_unknown='ignore',
            sparse_output=False,
        )))
        categorical_report.append(f'더미변수 인코딩(drop_first={drop_first})')

    # --- 6) 전처리 단계 출력 ---
    if verbose:
        print('\n전처리 단계')

        for label, columns, report in (('연속형', continuous, numeric_report),
                                       ('명목형', nominal_cols, categorical_report)):
            if not columns:      text = '(대상 없음)'
            elif not report:     text = '(변환 없음)'
            else:                text = ' → '.join(report)

            print(f'  {label}: {text}')

    # --- 7) 파이프라인 조립 ---
    # 컬럼 종류별로 다른 전처리를 적용해야 하므로 ColumnTransformer 로 갈래를 나눈다.
    # set_output(transform='pandas') 를 해야 VIFSelector 가 컬럼명을 보고 변수를 고를 수 있다
    preprocessor = ColumnTransformer([
        ('num', Pipeline(numeric_steps) if numeric_steps else 'passthrough', continuous),
        ('cat', Pipeline(categorical_steps) if categorical_steps else 'passthrough', nominal_cols),
    ], remainder='passthrough', n_jobs=-1, verbose_feature_names_out=False)

    preprocessor.set_output(transform='pandas')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    # --- 8) 모델 학습 ---
    pipeline.fit(x_train, y_train, **fit_params)

    if verbose:
        print(f'\n모델 학습 완료: {model_name}')

    # --- 9) 보고에 필요한 정보를 결과 객체에 붙여 반환 ---
    pipeline.pipeline_info_ = {
        'model_class': model_name,
        'nominal_cols': nominal_cols,
        'continuous_cols': continuous,
        'impute': impute,
        'numeric_impute': numeric_impute,
        'categorical_impute': categorical_impute,
        'outlier': outlier,
        'outlier_method': outlier_method,
        'vif': vif,
        'vif_threshold': vif_threshold,
        'scale': scale,
        'scale_method': scale_method,
        'pca': pca,
        'pca_variance': pca_variance,
        'encode': encode,
        'drop_first': drop_first,
    }

    # 모델을 구분할 이름 (성능 비교표의 인덱스로 쓴다)
    pipeline.name_ = model_name

    # --- 10) 학습된 모델 저장 (선택) ---
    if save_path:
        save_model(pipeline, save_path)

        if verbose:
            print(f'모델 저장: {save_path}')

    return pipeline


# --------------------------------------------------------
# 회귀 모델의 성능 지표 계산
# --------------------------------------------------------
def reg_score(estimator, x_test, y_test):
    """회귀 모델의 성능 지표(R2/MAE/MSE/RMSE/RMSLE/MAPE/MPE)를 계산한다.

    Args:
        estimator: 학습이 완료된 사이킷런 회귀 모델 또는 파이프라인. GridSearchCV 같은
            하이퍼파라미터 탐색 객체를 주면 내부의 best_estimator_ 로 평가한다.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series | ndarray): 검증 데이터의 종속변수.

    Returns:
        DataFrame: 모델 클래스명을 인덱스로 하는 지표 1행. 음수·0 으로 계산이 불가능하면 NaN.
    """
    # --- 1) 평가할 모델을 확정하고 예측을 수행한다 ---
    # GridSearchCV·RandomizedSearchCV 등 탐색 객체면 최적 모델을 꺼내 쓴다.
    # 탐색 객체 자체로 예측해도 값은 같지만, 클래스명이 'GridSearchCV' 로 잡혀
    # 어떤 모델의 점수인지 알 수 없게 된다.
    if hasattr(estimator, 'best_estimator_'):
        estimator = estimator.best_estimator_

    # 파이프라인이면 마지막 단계의 모델에서 클래스명을 꺼낸다
    if hasattr(estimator, 'named_steps'):
        classname = estimator.named_steps['model'].__class__.__name__
    else:
        classname = estimator.__class__.__name__

    y_pred = estimator.predict(x_test)

    # y_test 를 1차원 배열로 통일 (DataFrame·Series·ndarray 무엇이 와도 동일하게 계산)
    if isinstance(y_test, DataFrame):
        y_test_array = y_test.values.ravel()
    else:
        y_test_array = np.asarray(y_test).ravel()

    # --- 2) 성능 지표 계산 ---
    # 기본 지표: 어떤 데이터에서든 항상 계산된다
    r2 = r2_score(y_test_array, y_pred)
    mae = mean_absolute_error(y_test_array, y_pred)
    mse = mean_squared_error(y_test_array, y_pred)
    rmse = np.sqrt(mse)

    # RMSLE: 로그를 취하므로 음수가 하나라도 있으면 계산 불가
    if np.any(y_test_array < 0) or np.any(y_pred < 0):
        rmsle = np.nan
    else:
        rmsle = np.sqrt(mean_squared_log_error(y_test_array, y_pred))

    # MAPE·MPE: 실제값으로 나누므로 0 이 하나라도 있으면 계산 불가.
    # 두 지표 모두 비율(0.05 = 5%)로 돌려준다. R2 와 단위를 맞추기 위함이며,
    # 백분율이 필요하면 사용하는 쪽에서 100 을 곱한다
    if np.any(y_test_array == 0):
        mape = np.nan
        mpe = np.nan
    else:
        mape = mean_absolute_percentage_error(y_test_array, y_pred)
        mpe = np.mean((y_test_array - y_pred) / y_test_array)

    # --- 3) 계산한 지표를 모델명 1행짜리 표로 정리해 반환 ---
    score_df = DataFrame({
        'R2': r2,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'RMSLE': rmsle,
        'MAPE': mape,
        'MPE': mpe,
    }, index=[classname])
    score_df.index.name = 'Model'

    return score_df


# ==========================================================================
# 모델 비교·과적합 판정의 공용 엔진 — reg_* 와 cls_* 가 함께 쓴다
# ==========================================================================
# 순위를 매기는 규칙(4단계 전략)과 과적합을 판정하는 규칙(훈련↔CV 격차)은 회귀든
# 분류든 똑같다. 달라지는 것은 '어떤 지표를 쓰는가' 뿐이라, 지표표(metric_specs)만
# 갈아 끼우면 되도록 알고리즘을 이곳에 모아 둔다.


# --------------------------------------------------------
# 파이프라인·탐색 객체를 풀어 모델과 전처리 단계를 분리
# --------------------------------------------------------
def _unwrap_estimator(estimator):
    """학습 결과물을 풀어 (모델, 전처리 단계, 원본 추정기, best_params) 를 반환한다.

    feature_importance 와 SHAP 은 둘 다 '최종 모델' 과 '모델 직전까지의 전처리' 를
    따로 필요로 한다. 어느 쪽이든 GridSearchCV 로 감싼 파이프라인, 맨 파이프라인,
    단독 모델을 모두 받을 수 있어야 해서 이 풀어내기를 한곳에 모아 둔다.

    Args:
        estimator: 학습된 모델·파이프라인 또는 GridSearchCV 등 탐색 객체

    Returns:
        tuple: (model, pre, base_est, search_params). pre 는 모델 직전까지를 잘라낸
            전처리 파이프라인이며, 단독 모델이면 None 이다.
    """
    # 탐색 객체면 최적 모델을 꺼낸다
    base_est = getattr(estimator, 'best_estimator_', estimator)
    search_params = getattr(estimator, 'best_params_', None)

    if isinstance(base_est, Pipeline):
        # fit_pipeline 은 마지막 단계 이름을 'model' 로 붙이지만, 직접 만든
        # 파이프라인은 이름이 다를 수 있어 맨 뒤 단계로 대신한다
        model = base_est.named_steps.get('model', base_est[-1])
        pre = base_est[:-1]                     # 모델 직전까지의 전처리 단계
    else:
        # 파이프라인이 아닌 단독 모델도 허용 — 전처리 단계가 없다
        model = base_est
        pre = None

    return model, pre, base_est, search_params


# --------------------------------------------------------
# 주 지표·보조 지표 이름이 지표표에 있는지 검증
# --------------------------------------------------------
def _validate_metrics(primary, aux, metric_specs):
    """주 지표·보조 지표가 계산 가능한 이름인지 확인하고 보조 지표를 리스트로 통일한다.

    Args:
        primary (str): 순위를 가르는 주 지표
        aux (list | str): 결함 판정에 쓸 보조 지표
        metric_specs (dict): 사용 가능한 지표표

    Returns:
        list: 리스트로 통일된 보조 지표.

    Raises:
        ValueError: 계산되지 않는 지표명을 준 경우.
    """
    if isinstance(aux, str):
        aux = [aux]     # 보조 지표를 문자열 하나로 준 경우도 허용

    if primary not in metric_specs:
        raise ValueError(f"지원하지 않는 주 지표입니다: '{primary}' "
                         f"(사용 가능: {sorted(metric_specs)})")

    for m in aux:
        if m not in metric_specs:
            raise ValueError(f"지원하지 않는 보조 지표입니다: '{m}' "
                             f"(사용 가능: {sorted(metric_specs)})")

    return list(aux)


# --------------------------------------------------------
# 모델별 점수표에 4단계 전략으로 Rank 를 매긴다
# --------------------------------------------------------
def _rank_score_table(final_score_table, metric_specs, primary, aux, verbose):
    """모델별 점수표를 받아 4단계 전략으로 순위를 매긴 비교표를 돌려준다.

    주 지표 하나만으로는 소수점 차이로 1등이 갈리므로, ① 주 지표로 정렬 → ② 1등의 5%
    이내를 '근소 격차 그룹' 으로 묶기 → ③ 그룹 내부의 보조 지표 결함 개수 세기 →
    ④ (결함 수, 주 지표) 순으로 그룹 내부 재정렬, 의 순서로 순위를 매긴다.

    Args:
        final_score_table (DataFrame): index=모델 이름, 컬럼=지표인 점수표
        metric_specs (dict): 지표별 방향·결함 판정 방식
        primary (str): 순위를 가르는 주 지표
        aux (list): 결함 판정에 쓸 보조 지표
        verbose (bool): 판정 과정 출력 여부

    Returns:
        DataFrame: Rank 순 비교표. 맨 앞에 `Rank`·`Group`, 맨 끝에 `{primary}_Gap` 컬럼.
    """
    p_better = metric_specs[primary]['better']

    if verbose:
        print('\n' + '=' * 70)
        print(f'◆ Score Table Ranking : primary={primary!r}, aux={aux}')
        print('=' * 70)

    # --- 4) step1: 주 지표 기준 정렬 ---
    if p_better == 'closer_to_zero':
        # 부호가 아니라 크기가 문제이므로 절대값으로 줄을 세운다
        order_idx = final_score_table[primary].abs().sort_values(kind='mergesort').index
        sorted_table = final_score_table.loc[order_idx]
        primary_ascending = True
    else:
        primary_ascending = (p_better == 'lower')
        sorted_table = final_score_table.sort_values(
            primary, ascending=primary_ascending, kind='mergesort'
        )

    if verbose:
        direction_label = {
            'lower': '낮을수록 좋음 (ASC)',
            'higher': '높을수록 좋음 (DESC)',
            'closer_to_zero': '0에 가까울수록 좋음 (|x| ASC)',
        }[p_better]
        print(f'\n▲ step1: 주 지표({primary}) 기준 정렬 — {direction_label}')
        for i, (name, val) in enumerate(sorted_table[primary].items(), 1):
            print(f'   {i:>2}. {name:<14} {primary:<6}= {val:>16.3f}')

    # --- 5) step2: 1등과 5% 이내인 모델을 '근소 격차 그룹' 으로 묶기 ---
    primary_col = sorted_table[primary]

    if p_better == 'lower':
        best_primary = primary_col.min(skipna=True)
        close_mask = primary_col <= best_primary * 1.05
        band_str = f'{primary} ≤ {best_primary * 1.05:.3f}'
    elif p_better == 'higher':
        best_primary = primary_col.max(skipna=True)
        # 1등의 5% 아래까지 — 1등이 음수일 수도 있으므로(R2·MCC) 크기의 5% 를 뺀다.
        # best * 0.95 로 계산하면 음수일 때 기준선이 1등보다 위로 올라가 1등이 탈락한다
        band = best_primary - abs(best_primary) * 0.05
        close_mask = primary_col >= band
        band_str = f'{primary} ≥ {band:.3f}'
    else:   # closer_to_zero
        best_primary = primary_col.abs().min(skipna=True)
        close_mask = primary_col.abs() <= best_primary * 1.05
        band_str = f'|{primary}| ≤ {abs(best_primary) * 1.05:.3f}'

    close_group = sorted_table[close_mask].copy()
    outside_group = sorted_table[~close_mask]

    if verbose:
        print(f'\n▲ step2: 근소 격차 그룹 묶기 (1등의 5% 이내)')
        print(f'   - 1등 {primary:<6} : {best_primary:.3f}')
        print(f'   - 허용 범위    : {band_str}')
        print(f'   - 근소 격차 그룹 ({len(close_group)}) : {list(close_group.index)}')
        print(f'   - 그룹 외부     ({len(outside_group)}) : {list(outside_group.index)}')

    # 그룹에 1등만 있으면 비교할 상대가 없으므로 step3 을 건너뛴다 (1등 압도적)
    if len(close_group) > 1:
        # --- 6) step3: 그룹 내부에서 보조 지표의 결정적 결함 세기 ---
        # 보조 지표별 그룹 1등 점수 (결함 판정의 기준값)
        aux_bests = {}
        for m in aux:
            aux_col = close_group[m]
            m_better = metric_specs[m]['better']

            if m_better == 'lower':
                aux_bests[m] = aux_col.min(skipna=True)
            elif m_better == 'higher':
                aux_bests[m] = aux_col.max(skipna=True)
            else:   # closer_to_zero
                aux_bests[m] = aux_col.abs().min(skipna=True)

        # 모델별로 결함을 세면서, 어떤 지표에서 걸렸는지 이름도 남긴다
        flaw_details = {}
        for idx in close_group.index:
            row = close_group.loc[idx]
            triggered = []

            for m in aux:
                spec = metric_specs[m]
                value = row[m]
                best = aux_bests[m]

                if np.isnan(best):
                    is_flaw = False     # 그룹 전체가 미측정 → 비교 불가
                elif np.isnan(value):
                    is_flaw = True      # 본인만 측정 불가 → 결함
                elif spec['flaw_type'] == 'rel_excess':
                    is_flaw = value > best * (1 + spec['threshold'])
                elif spec['flaw_type'] == 'abs_drop':
                    is_flaw = value < best - spec['threshold']
                else:   # abs_excess
                    is_flaw = abs(value) > best + spec['threshold']

                if is_flaw:
                    triggered.append(m)

            flaw_details[idx] = triggered

        close_group['_flaws'] = [len(flaw_details[idx]) for idx in close_group.index]

        if verbose:
            print(f'\n▲ step3: 보조 지표 결정적 결함 점검 (근소 격차 그룹 내부)')
            print(f'   - 그룹 1등 점수 / 결함 임계치:')

            for m in aux:
                spec = metric_specs[m]
                b = aux_bests[m]

                if spec['flaw_type'] == 'rel_excess':
                    rule = f"row > {b * (1 + spec['threshold']):.3f}  (1등 × {1 + spec['threshold']:.2f})"
                elif spec['flaw_type'] == 'abs_drop':
                    rule = f"row < {b - spec['threshold']:.3f}  (1등 − {spec['threshold']:.2f})"
                else:   # abs_excess
                    rule = f"|row| > {b + spec['threshold']:.3f}  (1등 + {spec['threshold']:.2f})"

                print(f'       · {m:<6} best={b:>10.3f}   결함조건: {rule}')

            print(f'   - 모델별 결함:')
            for idx in close_group.index:
                triggered = flaw_details[idx]
                tag = '(결함 없음)' if not triggered else f'→ {triggered}'
                print(f'       · {idx:<14} {len(triggered)}개 {tag}')

        # --- 7) step4: (결함 수, 주 지표) 오름차순 재정렬 ---
        # 결함이 적은 모델이 위로, 동률이면 주 지표가 좋은 모델이 위로 간다
        if p_better == 'closer_to_zero':
            close_group['_primary_key'] = close_group[primary].abs()
            close_group = close_group.sort_values(
                ['_flaws', '_primary_key'], ascending=[True, True], kind='mergesort'
            ).drop(columns=['_flaws', '_primary_key'])
        else:
            close_group = close_group.sort_values(
                ['_flaws', primary],
                ascending=[True, primary_ascending],
                kind='mergesort',
            ).drop(columns=['_flaws'])
    elif verbose:
        print(f'\n▲ step3: 스킵 — 근소 격차 그룹에 1등만 존재 (1등 압도적, step4 직행)')

    # --- 8) 재정렬한 근소 격차 그룹 + 주 지표 순 그룹 외부를 이어 붙이고 Rank 부여 ---
    final_score_table = concat([close_group, outside_group])
    final_score_table.insert(0, 'Rank', range(1, len(final_score_table) + 1))

    close_set = set(close_group.index)
    final_score_table.insert(1, 'Group', [
        'Contender' if name in close_set else 'Outside'
        for name in final_score_table.index
    ])

    # --- 9) 맨 끝 컬럼: 주 지표가 1등 대비 몇 % 나쁜지 (양수일수록 나쁨) ---
    gap_col = f'{primary}_Gap'

    if p_better == 'closer_to_zero':
        ref = abs(final_score_table[primary].iloc[0])
        diff = final_score_table[primary].abs() - ref
    else:
        ref = final_score_table[primary].iloc[0]
        # 높을수록 좋은 지표면 1등보다 낮을수록 나쁜 것이므로 부호를 뒤집어 양수로 만든다
        sign = 1.0 if p_better == 'lower' else -1.0
        diff = sign * (final_score_table[primary] - ref)

    if ref == 0:
        # 기준값이 0 이면 비율을 계산할 수 없다
        final_score_table[gap_col] = np.where(diff == 0, 0.0, np.nan)
    else:
        final_score_table[gap_col] = (diff / abs(ref)).round(3)

    if verbose:
        print(f'\n▲ step4: 최종 Rank')
        for rank, name in zip(final_score_table['Rank'], final_score_table.index):
            tag = '[그룹 내]' if name in close_set else '[그룹 외 · 주 지표 순]'
            print(f'   {rank:>2}. {name:<14} {tag}')
        print('=' * 70 + '\n')

    return final_score_table


# --------------------------------------------------------
# 여러 회귀 모델의 지표를 한 번에 계산하고 4단계 전략으로 순위를 매긴 비교표 생성
# --------------------------------------------------------
def reg_compare_models(estimator, x_test, y_test, primary='RMSE',
                       aux=['MAE', 'R2'], verbose=True):
    """여러 회귀 모델의 지표를 계산하고 4단계 전략으로 'Rank' 를 매긴 비교표를 만든다.

    주 지표 하나만으로는 소수점 차이로 1등이 갈리므로, ① 주 지표로 정렬 → ② 1등의 5%
    이내를 '근소 격차 그룹' 으로 묶기 → ③ 그룹 내부의 보조 지표 결함 개수 세기 →
    ④ (결함 수, 주 지표) 순으로 그룹 내부 재정렬, 의 순서로 순위를 매긴다.

    Args:
        estimator (list | dict): 비교할 모델의 리스트 또는 {'이름': 모델} 딕셔너리.
            리스트면 모델의 `name_` 속성을, 없으면 `Model 1` … 을 이름으로 쓴다.
            GridSearchCV 같은 탐색 객체는 내부의 best_estimator_ 를 꺼내 평가한다.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series | ndarray): 검증 데이터의 종속변수.
        primary (str): 순위를 가르는 주 지표 (기본값: 'RMSE').
        aux (list): 결함 판정에 쓸 보조 지표 (기본값: ['MAE', 'R2']).
        verbose (bool): 판정 과정 출력 여부 (기본값: True).

    primary·aux 에는 reg_score 가 계산하는 R2·MAE·MSE·RMSE·RMSLE·MAPE·MPE 를 쓴다.

    Returns:
        DataFrame: Rank 순 비교표. 맨 앞에 `Rank`·`Group`(Contender=근소 격차 그룹 /
            Outside=그룹 외부), 맨 끝에 `{primary}_Gap`(1등 대비 격차, 양수일수록 나쁨) 컬럼.

    Raises:
        TypeError: estimator 가 리스트도 딕셔너리도 아닌 경우.
        ValueError: primary·aux 에 계산되지 않는 지표명을 준 경우.
    """
    # --- 1) 지표 메타데이터 ---
    # better    : 'lower' | 'higher' | 'closer_to_zero'  — 어느 쪽이 좋은 값인가
    # flaw_type : 보조 지표의 결정적 결함 판정 방식
    #     rel_excess  :  값 > 1등 * (1 + threshold)   낮을수록 좋은 지표용
    #     abs_drop    :  값 < 1등 - threshold         높을수록 좋은 지표용
    #     abs_excess  : |값| > 1등 + threshold        0에 가까울수록 좋은 지표용
    # threshold : 1등 대비 결함으로 판정할 임계치
    metric_specs = {
        'R2':    {'better': 'higher',         'flaw_type': 'abs_drop',   'threshold': 0.05},
        'MAE':   {'better': 'lower',          'flaw_type': 'rel_excess', 'threshold': 0.10},
        'MSE':   {'better': 'lower',          'flaw_type': 'rel_excess', 'threshold': 0.10},
        'RMSE':  {'better': 'lower',          'flaw_type': 'rel_excess', 'threshold': 0.10},
        'RMSLE': {'better': 'lower',          'flaw_type': 'rel_excess', 'threshold': 0.10},
        'MAPE':  {'better': 'lower',          'flaw_type': 'rel_excess', 'threshold': 0.10},
        'MPE':   {'better': 'closer_to_zero', 'flaw_type': 'abs_excess', 'threshold': 0.05},
    }

    # --- 2) 파라미터 검증 ---
    aux = _validate_metrics(primary, aux, metric_specs)

    # --- 3) 모델별 점수 계산 ---
    # 딕셔너리면 키가 곧 이름이고, 리스트면 아래 루프에서 이름을 정한다
    if isinstance(estimator, dict):
        models = list(estimator.items())
    elif isinstance(estimator, list):
        models = [(None, m) for m in estimator]
    else:
        raise TypeError('estimator 는 모델의 리스트 또는 딕셔너리여야 합니다: '
                        f'{type(estimator).__name__}')

    score_tables = []
    for i, (name, model) in enumerate(models):
        # GridSearchCV·RandomizedSearchCV 등 탐색 객체면 최적 모델을 꺼내 쓴다.
        # 그대로 두면 모델명이 전부 'GridSearchCV' 로 찍혀 구분이 되지 않는다.
        best_model = getattr(model, 'best_estimator_', model)

        # 리스트로 받았으면 name_ 을 이름으로 쓴다. 탐색 객체는 내부 모델을 clone 해서
        # 재학습하므로 안쪽 name_ 은 사라진다. 탐색 객체 자신에 붙여둔 이름을 먼저 찾는다.
        if name is None:
            name = (getattr(model, 'name_', None)
                    or getattr(best_model, 'name_', None)
                    or f'Model {i + 1}')

        score_df = reg_score(best_model, x_test, y_test)
        score_df.reset_index(inplace=True)   # 모델 클래스명을 'Model' 컬럼으로 내린다
        score_df.index = [name]
        score_tables.append(score_df)

    final_score_table = concat(score_tables)
    final_score_table.index.name = 'name'

    return _rank_score_table(final_score_table, metric_specs, primary, aux, verbose)


# --------------------------------------------------------
# 훈련·교차검증·검증 성능을 한 표로 비교해 과적합 여부를 판정
# --------------------------------------------------------
def reg_overfit(estimator, x_train, y_train, x_test, y_test,
                metrics=['RMSE', 'MAE', 'R2'], threshold=0.15, underfit_threshold=0.3,
                cv=5, fit_params=None, learning_curve=True,
                width=1280, height=640, grid=True, save_path=None, verbose=True):
    """Train / CV / Test 성능을 한 표로 보여주고 과적합 여부를 판정한다.

    판정은 훈련 성능과 교차검증(out-of-fold) 성능의 격차로 한다. 검증 데이터(Test)는
    표에 참고용으로만 싣고 판정에는 쓰지 않는다. 반복 진단할수록 검증 데이터가 오염되고,
    한 번 나눈 홀드아웃보다 K개 폴드의 평균이 안정적이기 때문이다.
        - 과소적합: 훈련 성능 자체가 낮음 (train R2 < underfit_threshold).
            격차가 아니라 절대 성능 문제이므로 과대적합보다 먼저 판정한다.
        - 과대적합: 훈련↔CV 격차가 큼 (Gap% >= threshold).
        - 일반화: 위 둘 다 아님.

    Args:
        estimator: 학습된 회귀 모델·파이프라인 또는 GridSearchCV 등 탐색 객체.
        x_train (DataFrame): 훈련 데이터의 독립변수.
        y_train (Series | ndarray): 훈련 데이터의 종속변수.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series | ndarray): 검증 데이터의 종속변수.
        metrics (list): 표시·판정할 지표 (기본값: ['RMSE', 'MAE', 'R2']).
        threshold (float): 과대적합으로 볼 Gap% (기본값: 0.15).
        underfit_threshold (float): 과소적합으로 볼 train R2 하한 (기본값: 0.3).
        cv (int): 교차검증 폴드 수 (기본값: 5).
        fit_params (dict): CV 재학습에 넘길 인자 (예: {'model__cat_features': [...]}) (기본값: None).
        learning_curve (bool): 학습곡선 출력 여부 (기본값: True).
        width (int): 학습곡선 가로 크기(픽셀) (기본값: 1280).
        height (int): 학습곡선 세로 크기(픽셀) (기본값: 640).
        grid (bool): 학습곡선 격자 표시 여부 (기본값: True).
        save_path (str): 학습곡선 이미지 저장 경로 (기본값: None).
        verbose (bool): 판정 과정 출력 여부 (기본값: True).

    threshold·underfit_threshold 는 학술 표준이 아닌 경험칙이다. 기본값은 노이즈가
    중간~높은 사회·도시 데이터 기준이며, 임계값보다 학습곡선의 추세가 더 믿을 만하다.

    Returns:
        DataFrame: index=Metric, 컬럼=[Train, CV, CV_Std, Test, Gap, Gap%, Overfit].
            `result.attrs['diagnosis']` 에 모델 수준 최종 진단이 담긴다.

    Raises:
        ValueError: metrics 에 지원하지 않는 지표명을 준 경우.
    """
    # --- 1) 지표 메타데이터 ---
    # 지표: (최적 방향, 사이킷런 scoring 문자열, fold 점수를 reg_score 와 같은 단위로 되돌리는 함수)
    # neg_* scorer 는 '클수록 좋게' 부호가 뒤집혀 있어 되돌리고, RMSLE 는 제곱근을 취한다.
    # MAPE 는 사이킷런도 reg_score 도 비율이므로 부호만 되돌리면 단위가 맞는다.
    # MPE 는 0에 가까울수록 좋은 지표라 '훈련보다 얼마나 나쁜가'를 정의할 수 없어 제외한다.
    metric_specs = {
        'R2':    ('higher', 'r2',                                 lambda s: s),
        'MAE':   ('lower',  'neg_mean_absolute_error',            lambda s: -s),
        'MSE':   ('lower',  'neg_mean_squared_error',             lambda s: -s),
        'RMSE':  ('lower',  'neg_root_mean_squared_error',        lambda s: -s),
        'RMSLE': ('lower',  'neg_mean_squared_log_error',         lambda s: np.sqrt(-s)),
        'MAPE':  ('lower',  'neg_mean_absolute_percentage_error', lambda s: -s),
    }

    # --- 2) 탐색 객체면 최적 모델을 꺼내기 ---
    # 탐색 객체 그대로 교차검증하면 폴드마다 탐색을 다시 도는 중첩 CV 가 되어 버린다.
    _, _, base_est, search_params = _unwrap_estimator(estimator)

    # --- 3) 공용 엔진에 회귀용 지표표를 넘겨 판정 ---
    # 과소적합은 스케일에 좌우되지 않는 R2 로만 본다 (RMSE 등은 절대 임계치를 정할 수 없다)
    return _fit_diagnosis(
        base_est, search_params, x_train, y_train, x_test, y_test,
        metrics=metrics, metric_specs=metric_specs, score_func=reg_score,
        underfit_metric='R2', threshold=threshold, underfit_threshold=underfit_threshold,
        cv=cv, fit_params=fit_params, learning_curve=learning_curve,
        width=width, height=height, grid=grid, save_path=save_path, verbose=verbose)


# --------------------------------------------------------
# Train↔CV 격차로 과적합을 판정하는 공용 엔진
# --------------------------------------------------------
def _fit_diagnosis(base_est, search_params, x_train, y_train, x_test, y_test,
                   metrics, metric_specs, score_func, underfit_metric,
                   threshold, underfit_threshold, cv, fit_params, learning_curve,
                   width, height, grid, save_path, verbose):
    """Train / CV / Test 점수를 모아 과적합·과소적합을 판정한다 (회귀·분류 공용).

    판정 규칙은 두 과제가 같다 — 훈련 성능과 교차검증 성능의 격차가 크면 과대적합,
    훈련 성능 자체가 낮으면 과소적합. 달라지는 것은 어떤 지표로 점수를 내고(score_func)
    어떤 지표로 과소적합을 보는가(underfit_metric) 뿐이다.

    Args:
        base_est: 탐색 객체를 푼 모델·파이프라인
        search_params (dict): 탐색 객체의 best_params (없으면 None)
        x_train (DataFrame): 훈련 데이터의 독립변수
        y_train (Series | ndarray): 훈련 데이터의 종속변수
        x_test (DataFrame): 검증 데이터의 독립변수
        y_test (Series | ndarray): 검증 데이터의 종속변수
        metrics (list | str): 표시·판정할 지표
        metric_specs (dict): 지표 → (최적 방향, scoring, fold 점수 보정 함수)
        score_func (callable): Train·Test 점수를 계산할 함수 (reg_score · cls_score)
        underfit_metric (str): 과소적합 판정에 쓸 지표 (회귀=R2, 분류=ROC_AUC)
        threshold (float): 과대적합으로 볼 Gap%
        underfit_threshold (float): 과소적합으로 볼 훈련 성능 하한
        cv (int): 교차검증 폴드 수
        fit_params (dict): CV 재학습에 넘길 인자
        learning_curve (bool): 학습곡선 출력 여부
        width (int): 학습곡선 가로 크기(픽셀)
        height (int): 학습곡선 세로 크기(픽셀)
        grid (bool): 학습곡선 격자 표시 여부
        save_path (str): 학습곡선 이미지 저장 경로
        verbose (bool): 판정 과정 출력 여부

    Returns:
        DataFrame: index=Metric, 컬럼=[Train, CV, CV_Std, Test, Gap, Gap%, Overfit].
            `result.attrs['diagnosis']` 에 모델 수준 최종 진단이 담긴다.

    Raises:
        ValueError: metrics 에 지원하지 않는 지표명을 준 경우.
    """
    # --- 1) 파라미터 검증 ---
    if isinstance(metrics, str):
        metrics = [metrics]     # 지표를 문자열 하나로 준 경우도 허용

    for m in metrics:
        if m not in metric_specs:
            raise ValueError(f"지원하지 않는 지표입니다: '{m}' "
                             f"(사용 가능: {sorted(metric_specs)})")

    # --- 4) Train / Test 점수 ---
    train_scores = score_func(base_est, x_train, y_train)
    test_scores = score_func(base_est, x_test, y_test)
    classname = train_scores.index[0]

    # --- 5) CV 점수 (out-of-fold) — 판정의 일반화 기준 ---
    # 과소적합 판정 지표는 metrics 에 없어도 항상 함께 계산한다.
    y_cv = y_train.values.ravel() if isinstance(y_train, DataFrame) else np.asarray(y_train).ravel()
    wanted = list(dict.fromkeys(list(metrics) + [underfit_metric]))
    out = cross_validate(base_est, x_train, y_cv, cv=cv,
                         scoring={m: metric_specs[m][1] for m in wanted},
                         n_jobs=-1, params=fit_params, error_score=np.nan)
    cv_scores = {m: metric_specs[m][2](out[f'test_{m}']) for m in wanted}   # 부호·단위 보정

    # --- 6) 과소적합 선판정 (모델 수준) ---
    # 훈련 성능이 낮으면 격차 분석 자체가 무의미하므로 모든 지표 행에 우선 적용한다.
    # 판정 지표는 스케일에 좌우되지 않는 것으로 고른다 (RMSE 등은 절대 임계치를 정할 수 없다).
    train_base = train_scores[underfit_metric].iloc[0]
    cv_base = float(np.mean(cv_scores[underfit_metric]))

    # 판정 지표를 계산할 수 없으면(예: 확률을 못 내는 분류 모델의 ROC_AUC) 판정을 보류한다
    underfit_available = not np.isnan(train_base)
    underfit = bool(underfit_available and train_base < underfit_threshold)

    # --- 7) 지표별 격차와 판정 ---
    rows = {}
    for m in metrics:
        direction = metric_specs[m][0]
        tr = train_scores[m].iloc[0]
        te = test_scores[m].iloc[0]
        cv_mean = float(np.mean(cv_scores[m]))
        cv_std = float(np.std(cv_scores[m]))

        # 방향에 맞춰 'CV 가 훈련보다 나쁜 정도' 를 양수 격차로 만든다
        gap = (tr - cv_mean) if direction == 'higher' else (cv_mean - tr)

        # Gap% : 격차를 스케일 보정.
        #   R2 류(상한 1)는 이미 정규화된 지표라 점수 차를 그대로 쓴다. 상대화하면
        #   R2 가 0 부근일 때 분모가 0 에 수렴해 허위 과대적합이 발생한다.
        #   오차 류(상한 없음)는 gap / max(|train|, |cv|). 분모의 max 는 훈련 오차가
        #   0 에 수렴할 때(완전 암기) 0 으로 나누는 것을 막는다.
        denom = 1.0 if direction == 'higher' else max(abs(tr), abs(cv_mean))
        gap_pct = gap / denom if denom else np.nan

        if underfit:
            label = '과소적합'
        elif np.isnan(gap_pct):
            label = 'N/A'
        elif gap_pct >= threshold:
            label = '과대적합'
        else:
            label = '일반화'

        rows[m] = {'Train': tr, 'CV': cv_mean, 'CV_Std': cv_std, 'Test': te,
                   'Gap': gap, 'Gap%': round(gap_pct * 100, 2), 'Overfit': label}

    result = DataFrame.from_dict(rows, orient='index')
    result.index.name = 'Metric'

    # --- 8) 모델 수준 최종 진단 ---
    # 과소적합(高편향) > 과대적합(高분산) > 일반화 순으로 우선한다.
    if underfit:
        diagnosis = '과소적합'
    elif any(rows[m]['Overfit'] == '과대적합' for m in metrics):
        diagnosis = '과대적합'
    else:
        diagnosis = '일반화'
    result.attrs['diagnosis'] = diagnosis

    # --- 9) 학습곡선 ---
    # 수치 판정보다 먼저 그려서 '곡선을 보고 표로 확인' 하는 흐름을 만든다.
    if learning_curve:
        # 주 지표(metrics[0]) 기준으로 그려 판정 표와 기준을 맞춘다
        lc_metric = metrics[0]
        _, lc_scoring, lc_transform = metric_specs[lc_metric]

        sizes, train_curve, cv_curve = sk_learning_curve(
            base_est, x_train, y_cv, train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv, scoring=lc_scoring, n_jobs=-1, params=fit_params,
        )
        train_curve, cv_curve = lc_transform(train_curve), lc_transform(cv_curve)
        train_mean, train_std = train_curve.mean(axis=1), train_curve.std(axis=1)
        cv_mean_c, cv_std_c = cv_curve.mean(axis=1), cv_curve.std(axis=1)

        fig, ax = my_plot.init(width=width, height=height, grid=grid,
                               title=f'Learning Curve: {classname}',
                               xlabel='Training samples', ylabel=f'{lc_metric} score')
        # 두 곡선의 간격이 벌어진 채 유지되면 과대적합, 둘 다 나쁘면 과소적합이다
        ax.plot(sizes, train_mean, marker='o', color='tab:blue', label='Train')
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color='tab:blue')
        ax.plot(sizes, cv_mean_c, marker='s', color='tab:orange', label='Validation (CV)')
        ax.fill_between(sizes, cv_mean_c - cv_std_c, cv_mean_c + cv_std_c,
                        alpha=0.15, color='tab:orange')
        ax.legend(fontsize=13)
        my_plot.show(save_path=save_path)

    # --- 10) 판정 출력 ---
    if verbose:
        print('\n' + '=' * 78)
        print(f'◆ Fit Diagnosis: {classname}  '
              f'(threshold={threshold:.0%}, 기준=Train↔CV {cv}-Fold)')
        if search_params is not None:
            print(f'   ▷ 탐색 객체의 best_params: {search_params}')
        print('   ※ threshold 는 학술 표준이 아닌 경험칙입니다. '
              '임계값보다 학습곡선 추세를 함께 보세요.')
        print('=' * 78)

        for m in metrics:
            r = rows[m]
            tag = {'일반화': '', '과대적합': ' ⚠', '과소적합': ' ⚑', 'N/A': ''}[r['Overfit']]
            print(f"   - {m:<6}  Train={r['Train']:>11.4f}  "
                  f"CV={r['CV']:>11.4f} (±{r['CV_Std']:.4f})  Test={r['Test']:>11.4f}  "
                  f"Gap%={r['Gap%']:>8.2f}%  [{r['Overfit']}]{tag}")

        message = {
            '일반화':   '✔ 일반화 (Good fit) — 격차가 작고 훈련 성능도 양호',
            '과대적합': '⚠ 과대적합 (Overfit · 高분산) — 훈련↔CV 격차가 큼',
            '과소적합': '⚑ 과소적합 (Underfit · 高편향) — 훈련 성능 자체가 낮음',
        }[diagnosis]
        print(f'\n   ▶ 진단: {message}')

        if underfit_available:
            print(f'     · train {underfit_metric}={train_base:.4f} '
                  f'(과소적합 기준 < {underfit_threshold}) · '
                  f'CV {underfit_metric}={cv_base:.4f}')
        else:
            print(f'     · {underfit_metric} 를 계산할 수 없어 과소적합 판정은 보류했습니다 '
                  f'(확률을 내지 못하는 모델).')

        print('=' * 78 + '\n')

    return result


# --------------------------------------------------------
# 베이스라인 모델 11종을 한 번에 학습·저장하고 성능 순위표를 만든다
# --------------------------------------------------------
def reg_baseline(project_name, x_train, y_train, x_test, y_test,
                 primary='RMSE', aux=['MAE', 'R2'], plot=True,
                 width=1280, height=640, save_path=None, verbose=True):
    """회귀 베이스라인 모델을 모두 학습·저장하고, 성능 순위표와 비교 그래프를 만든다.

    `{project_name}/baseline_{YYMMDD_HHMMSS}` 폴더를 만들어 모델별 pkl 파일을 저장한다.
    학습하는 모델은 선형 4종(linear·ridge·lasso·elasticnet), 비선형 2종(kneighbors·svr),
    트리 1종(decisiontree), 앙상블 4종(randomforest·xgb·lgbm·catboost)의 11개다.
    모델별 전처리 조합은 계열별 권장 조합으로 고정되어 있다.

    Args:
        project_name (str): 작업 폴더의 이름이 될 프로젝트명.
        x_train (DataFrame): 훈련 데이터의 독립변수.
        y_train (Series): 훈련 데이터의 종속변수.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series): 검증 데이터의 종속변수.
        primary (str): 순위를 가르는 주 지표 (기본값: 'RMSE').
        aux (list): 결함 판정에 쓸 보조 지표 (기본값: ['MAE', 'R2']).
        plot (bool): 성능 비교 그래프 출력 여부 (기본값: True).
        width (int): 그래프 가로 크기(픽셀) (기본값: 1280).
        height (int): 그래프 세로 크기(픽셀) (기본값: 640).
        save_path (str): 그래프 이미지 저장 경로 (기본값: None).
        verbose (bool): 모델별 학습 진행 상황 출력 여부 (기본값: True).

    마지막에 순위표(reg_compare_models 의 결과에 시각화용 `Score` 컬럼을 더한 표)를
    화면에 출력한다. 별도로 반환하는 값은 없다.
    """
    # 부스팅 3종은 무겁고 별도 설치가 필요한 패키지라 모듈 로드 시가 아니라 함수 안에서 import 한다
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    # --- 1) 학습 결과물을 담을 작업 폴더 생성 ---
    workdir = Path(project_name) / f'baseline'
    workdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f'작업 폴더: {workdir}')

    # 학습을 마친 모델을 {이름: 파이프라인} 으로 모아 마지막 비교표에 넘긴다
    models = {}

    # --- 2) 선형 계열 ---
    # 계수 해석과 규제의 전제가 되는 다중공선성을 VIF 로 제거하고, 더미 트랩도 함께 막는다.
    # LinearRegression 은 규제가 없어 스케일에 좌우되지 않으므로 정규화를 하지 않는다
    model = LinearRegression()
    model_name = model.__class__.__name__.lower().removesuffix('regression')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=False, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 1/11] : {model_name}')

    # Ridge·Lasso·ElasticNet 은 계수의 크기에 벌점을 매기므로 정규화가 반드시 필요하다
    model = Ridge(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 2/11] : {model_name}')

    model = Lasso(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 3/11] : {model_name}')

    model = ElasticNet(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 4/11] : {model_name}')

    # --- 3) 비선형 계열 ---
    # 거리·마진으로 학습하는 모델이라 변수의 단위가 다르면 큰 값의 변수가 거리를 독점한다
    model = KNeighborsRegressor(n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 5/11] : {model_name}')

    model = SVR()
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 6/11] : {model_name}')

    # --- 4) 트리 계열 ---
    # 분기 기준이 값의 대소 관계뿐이라 정규화가 필요 없고, 더미도 전부 남겨야
    # 각 범주가 독립적인 분기 후보가 된다 (drop_first 를 쓰지 않는 이유)
    model = DecisionTreeRegressor(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 7/11] : {model_name}')

    # --- 5) 앙상블 계열 ---
    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 8/11] : {model_name}')

    model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 9/11] : {model_name}')

    model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [10/11] : {model_name}')

    # CatBoost 는 범주형을 자체 방식(Ordered Target Statistics)으로 처리하므로
    # 더미 인코딩을 끄고, 어떤 컬럼이 범주형인지만 fit 인자로 알려준다.
    # 범주형 판정은 fit_pipeline 의 명목형 자동 선택과 같은 기준(category·object)으로 한다 —
    # my_qtcheck 는 category 만 보므로 CSV 에서 흔한 object 컬럼을 놓쳐 CatBoost 가 학습에 실패한다
    model = CatBoostRegressor(random_state=RANDOM_STATE, verbose=0)
    model_name = model.__class__.__name__.lower().removesuffix('regressor')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=False, encode=False,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False,
        model__cat_features=list(x_train.select_dtypes(include=['category', 'object']).columns))

    if verbose:
        print(f'학습 완료 [11/11] : {model_name}')

    # --- 6) 모델간 성능 비교 ---
    # 개별 모델의 지표는 출력하지 않고, 순위가 매겨진 표 하나만 남긴다
    score_table = reg_compare_models(models, x_test, y_test,
                                     primary=primary, aux=aux, verbose=False)

    # --- 7) 시각화용 점수 계산 ---
    # RMSE·MAE 처럼 낮을수록 좋은 지표를 그대로 그리면 막대가 길수록 나쁜 모델이 되어
    # 그래프가 직관과 어긋난다. 역수를 취해 '클수록 좋은 값' 으로 방향을 통일한다.
    if primary == 'R2':
        # 이미 높을수록 좋은 지표이므로 변환하지 않는다
        score_table['Score'] = score_table[primary]
        score_label = primary
    elif primary == 'MPE':
        # 0 에 가까울수록 좋은 지표라 부호를 떼고 크기만 역수로 바꾼다
        score_table['Score'] = 1 / score_table[primary].abs()
        score_label = f'1/|{primary}|'
    else:
        score_table['Score'] = 1 / score_table[primary]
        score_label = f'1/{primary}'

    # --- 8) 성능 비교 그래프 ---
    if plot:
        my_plot.barplot(score_table, y=score_table.index, x='Score', hue='Group',
                        palette='tab10', width=width, height=height,
                        title=f'모델 성능 비교({score_label} 기준)', save_path=save_path)

    # --- 9) 최종 순위표 출력 ---
    # 순위표가 이 함수의 결과물이므로 화면에 직접 출력한다.
    # 학습된 모델은 이미 pkl 로 저장했으니, 다시 쓸 때는 workdir 에서 load_model 로 불러온다
    print(f'\n모델 {len(models)}개 학습·저장 완료 → {workdir}')
    display(score_table)


# ==========================================================================
# 분류 모형 평가 — 회귀의 reg_* 4종에 대응하며 이진·다중분류를 모두 다룬다
# ==========================================================================
# 순위를 매기는 규칙도 과적합을 판정하는 규칙도 회귀와 같으므로 위의 공용 엔진
# (_rank_score_table · _fit_diagnosis) 을 그대로 쓰고, 지표를 계산하는 부분만 새로 만든다.
#
# 분류 지표는 회귀와 달리 '어떻게 평균 내는가' 를 정해야 한다. 이진은 양성 클래스
# 하나만 보면 되지만(average='binary'), 다중분류는 클래스마다 정밀도·재현율이 따로
# 나오기 때문이다. 기본값 average='auto' 는 이진이면 'binary', 다중분류면 'macro'
# (클래스를 같은 비중으로 평균) 로 자동 전환한다. 소수 클래스도 다수 클래스와 같은
# 무게로 세는 'macro' 가 불균형 데이터의 성능 과대평가를 막아 준다.


# --------------------------------------------------------
# 분류 지표 8종의 성격 — cls_score·cls_compare_models·cls_overfit 이 함께 본다
# --------------------------------------------------------
# needs_score : 예측 라벨이 아니라 확률·결정함수 점수가 있어야 계산되는 지표
# better      : 'higher' | 'lower'  — 어느 쪽이 좋은 값인가
# flaw_type   : 비교표에서 보조 지표의 결정적 결함을 판정하는 방식
#     abs_drop   : 값 < 1등 - threshold        상한이 1 로 정해진 지표용
#     rel_excess : 값 > 1등 × (1 + threshold)  상한이 없는 지표용(LogLoss)
# 딕셔너리에 적은 순서가 곧 cls_score 결과표의 컬럼 순서다.
_CLS_METRIC_SPECS = {
    'Accuracy':  {'needs_score': False, 'better': 'higher', 'flaw_type': 'abs_drop',   'threshold': 0.05},
    'Precision': {'needs_score': False, 'better': 'higher', 'flaw_type': 'abs_drop',   'threshold': 0.05},
    'Recall':    {'needs_score': False, 'better': 'higher', 'flaw_type': 'abs_drop',   'threshold': 0.05},
    'F1':        {'needs_score': False, 'better': 'higher', 'flaw_type': 'abs_drop',   'threshold': 0.05},
    'ROC_AUC':   {'needs_score': True,  'better': 'higher', 'flaw_type': 'abs_drop',   'threshold': 0.05},
    'PR_AUC':    {'needs_score': True,  'better': 'higher', 'flaw_type': 'abs_drop',   'threshold': 0.05},
    'LogLoss':   {'needs_score': True,  'better': 'lower',  'flaw_type': 'rel_excess', 'threshold': 0.10},
    'MCC':       {'needs_score': False, 'better': 'higher', 'flaw_type': 'abs_drop',   'threshold': 0.05},
}


# --------------------------------------------------------
# 예측 라벨로 계산하는 지표 (Accuracy·Precision·Recall·F1·MCC)
# --------------------------------------------------------
def _cls_label_metric(y_true, y_pred, *, name, average, classes):
    """예측 라벨만으로 계산하는 지표 하나를 돌려준다 (이진·다중분류 공용).

    cls_score 가 직접 호출하고, cls_overfit 은 make_scorer 로 감싸 교차검증에 쓴다.
    두 곳이 같은 함수를 쓰므로 표의 Train·CV·Test 를 나란히 놓고 비교할 수 있다.

    Args:
        y_true (ndarray): 실제 라벨
        y_pred (ndarray): 예측 라벨
        name (str): 지표 이름
        average (str): 'auto' · 'binary' · 'macro' · 'micro' · 'weighted'
        classes (ndarray): 모델이 학습한 클래스 목록 (마지막이 양성 클래스)

    Returns:
        float: 지표값. 계산할 수 없으면 NaN.
    """
    try:
        if name == 'Accuracy':
            return float(accuracy_score(y_true, y_pred))

        if name == 'MCC':
            # 매튜스 상관계수 — 혼동행렬 네 칸을 모두 쓰므로 불균형에 강하고
            # 이진·다중분류 구분 없이 하나의 값으로 나온다 (1=완벽, 0=무작위)
            return float(matthews_corrcoef(y_true, y_pred))

        # 이진이면 양성 클래스(마지막) 하나만, 다중분류면 클래스 평균으로 계산한다
        if len(classes) == 2 and average in ('auto', 'binary'):
            kwargs = {'average': 'binary', 'pos_label': classes[-1]}
        else:
            kwargs = {'average': 'macro' if average in ('auto', 'binary') else average}

        funcs = {'Precision': precision_score, 'Recall': recall_score, 'F1': f1_score}

        return float(funcs[name](y_true, y_pred, zero_division=0, **kwargs))
    except Exception:
        # 라벨로 계산할 수 없는 지표를 물었거나(LogLoss 등), 폴드에 특정 클래스가
        # 하나도 없어 점수를 낼 수 없는 경우 → NaN
        return np.nan


# --------------------------------------------------------
# 확률·결정함수 점수로 계산하는 지표 (ROC_AUC·PR_AUC·LogLoss)
# --------------------------------------------------------
def _cls_score_metric(y_true, y_score, *, name, average, classes, is_proba=True):
    """점수 출력으로 계산하는 지표 하나를 돌려준다 (이진·다중분류 공용).

    임계값 0.5 로 자른 예측 라벨이 아니라 점수 자체를 쓰므로, 임계값을 어디에 두든
    변하지 않는 '순위를 매기는 능력' 을 본다. 다중분류는 클래스별로 일대다(OvR) 곡선을
    그린 뒤 평균한다. _cls_label_metric 과 마찬가지로 cls_score 와 cls_overfit 이
    함께 쓴다.

    Args:
        y_true (ndarray): 실제 라벨
        y_score (ndarray): predict_proba 또는 decision_function 의 출력
        name (str): 지표 이름
        average (str): 다중분류에서 클래스 평균 방식
        classes (ndarray): 모델이 학습한 클래스 목록 (마지막이 양성 클래스)
        is_proba (bool): y_score 가 확률인지 여부. LogLoss 는 확률이어야 계산된다.

    Returns:
        float: 지표값. 계산할 수 없으면 NaN.
    """
    try:
        classes = np.asarray(classes)

        # dtype 을 float64 로 올리지 않는다. XGBoost 는 확률을 float32 로 내는데,
        # float32 안에서 정확히 1 이던 두 확률의 합이 float64 로 펼쳐지면 3e-8 만큼
        # 어긋난다. log_loss 는 배열의 dtype 으로 허용 오차를 정하므로(sqrt(eps)),
        # 값은 그대로인데 잣대만 2만 배 엄격해져 '합이 1이 아니다' 경고가 뜬다
        score = np.asarray(y_score)

        # LogLoss 는 '확률을 얼마나 자신 있게 맞혔나' 를 재므로 확률이 아니면 의미가 없다
        if name == 'LogLoss':
            if not is_proba or score.ndim != 2:
                return np.nan

            return float(log_loss(y_true, score, labels=list(classes)))

        if len(classes) == 2:
            # 이진 — 양성 클래스 점수 한 줄만 있으면 된다
            pos = score[:, -1] if score.ndim == 2 else score
            y_bin = (np.asarray(y_true) == classes[-1]).astype(int)

            if name == 'ROC_AUC':
                return float(roc_auc_score(y_bin, pos))

            return float(average_precision_score(y_bin, pos))

        # 다중분류 — 클래스별 점수 행렬이 있어야 일대다 곡선을 그릴 수 있다.
        # 클래스마다 '이 클래스인가 아닌가' 의 0/1 정답을 만들어 열별로 곡선을 그린다
        if score.ndim != 2 or score.shape[1] != len(classes):
            return np.nan

        y_bin = label_binarize(y_true, classes=classes)
        avg = 'macro' if average in ('auto', 'binary') else average

        if name == 'ROC_AUC':
            return float(roc_auc_score(y_bin, score, average=avg))

        return float(average_precision_score(y_bin, score, average=avg))
    except Exception:
        # 폴드에 한 클래스만 존재하는 등 곡선을 그릴 수 없는 경우 → NaN
        return np.nan


# --------------------------------------------------------
# 학습된 분류 모델의 성능 지표를 한 줄로 계산
# --------------------------------------------------------
def cls_score(estimator, x_test, y_test, average='auto'):
    """학습된 분류 모델의 성능 지표 8종을 계산해 1행짜리 표로 반환한다.

    회귀의 reg_score 에 대응하며 이진·다중분류를 모두 다룬다. 지표는 성격이 셋으로
    갈린다.
        - 임계값 0.5 로 자른 예측 라벨 기준: Accuracy · Precision · Recall · F1 · MCC
        - 점수의 순위 기준(임계값 무관): ROC_AUC · PR_AUC
        - 확률의 정확도 기준: LogLoss
    불균형 데이터에서는 Accuracy 가 다수 클래스만 맞혀도 높게 나오므로 F1·PR_AUC·MCC
    를 함께 본다. 확률을 내지 못하는 모델(RidgeClassifier·LinearSVC)은 LogLoss 가 NaN
    이지만, ROC_AUC·PR_AUC 는 결정함수의 마진으로 계산한다.

    Args:
        estimator: 학습이 완료된 사이킷런 분류 모델 또는 파이프라인. GridSearchCV 같은
            하이퍼파라미터 탐색 객체를 주면 내부의 best_estimator_ 로 평가한다.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series | ndarray): 검증 데이터의 종속변수.
        average (str): 다중분류의 클래스 평균 방식 (기본값: 'auto').
            'auto' 는 이진이면 'binary'(양성 클래스만), 다중분류면 'macro'.
            'micro'·'weighted' 도 쓸 수 있다.

    Returns:
        DataFrame: 모델 클래스명을 인덱스로 하는 지표 1행. 컬럼=[Accuracy, Precision,
            Recall, F1, ROC_AUC, PR_AUC, LogLoss, MCC]. 계산이 불가능하면 NaN.
    """
    # --- 1) 평가할 모델을 확정하고 예측을 수행한다 ---
    # 탐색 객체면 최적 모델을, 파이프라인이면 마지막 단계에서 모델명을 꺼낸다.
    # 예측 자체는 전처리가 붙은 파이프라인 전체(base_est)로 해야 한다.
    model, _, base_est, _ = _unwrap_estimator(estimator)
    classname = type(model).__name__

    y_pred = base_est.predict(x_test)

    # y_test 를 1차원 배열로 통일 (DataFrame·Series·ndarray 무엇이 와도 동일하게 계산)
    if isinstance(y_test, DataFrame):
        y_true = y_test.values.ravel()
    else:
        y_true = np.asarray(y_test).ravel()

    # 클래스 목록 — 모델이 학습한 순서를 그대로 따른다 (마지막이 양성 클래스)
    classes = getattr(model, 'classes_', None)
    if classes is None:
        classes = np.unique(y_true)
    classes = np.asarray(classes)

    # --- 2) 확률(없으면 결정함수 마진) 확보 ---
    # RidgeClassifier 처럼 확률을 못 내는 모델도 마진이 있으면 순위 기반 지표는 계산된다
    y_score = None
    is_proba = False

    if hasattr(base_est, 'predict_proba'):
        # 모델이 낸 dtype 을 그대로 둔다 (_cls_score_metric 의 주석 참고)
        y_score = np.asarray(base_est.predict_proba(x_test))
        is_proba = True
    elif hasattr(base_est, 'decision_function'):
        y_score = np.asarray(base_est.decision_function(x_test))

    # --- 3) 성능 지표 계산 ---
    # 라벨 기준 지표는 항상, 점수 기준 지표는 점수를 낼 수 있을 때만 계산된다
    scores = {}
    for name, spec in _CLS_METRIC_SPECS.items():
        if not spec['needs_score']:
            scores[name] = _cls_label_metric(y_true, y_pred, name=name,
                                             average=average, classes=classes)
        elif y_score is None:
            scores[name] = np.nan
        else:
            scores[name] = _cls_score_metric(y_true, y_score, name=name, average=average,
                                             classes=classes, is_proba=is_proba)

    # --- 4) 계산한 지표를 모델명 1행짜리 표로 정리해 반환 ---
    score_df = DataFrame(scores, index=[classname])
    score_df.index.name = 'Model'

    return score_df


# --------------------------------------------------------
# 여러 분류 모델의 지표를 한 번에 계산하고 4단계 전략으로 순위를 매긴 비교표 생성
# --------------------------------------------------------
def cls_compare_models(estimator, x_test, y_test, primary='F1',
                       aux=['ROC_AUC', 'Accuracy'], average='auto', verbose=True):
    """여러 분류 모델의 지표를 계산하고 4단계 전략으로 'Rank' 를 매긴 비교표를 만든다.

    회귀의 reg_compare_models 에 대응하며 순위 규칙은 완전히 같다. ① 주 지표로 정렬 →
    ② 1등의 5% 이내를 '근소 격차 그룹' 으로 묶기 → ③ 그룹 내부의 보조 지표 결함 개수
    세기 → ④ (결함 수, 주 지표) 순으로 그룹 내부 재정렬.

    주 지표를 Accuracy 로 두면 불균형 데이터에서 다수 클래스만 맞히는 모델이 1등이
    되기 쉬워 기본값을 F1 로 잡았다.

    Args:
        estimator (list | dict): 비교할 모델의 리스트 또는 {'이름': 모델} 딕셔너리.
            리스트면 모델의 `name_` 속성을, 없으면 `Model 1` … 을 이름으로 쓴다.
            GridSearchCV 같은 탐색 객체는 내부의 best_estimator_ 를 꺼내 평가한다.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series | ndarray): 검증 데이터의 종속변수.
        primary (str): 순위를 가르는 주 지표 (기본값: 'F1').
        aux (list): 결함 판정에 쓸 보조 지표 (기본값: ['ROC_AUC', 'Accuracy']).
        average (str): 다중분류의 클래스 평균 방식 (기본값: 'auto').
        verbose (bool): 판정 과정 출력 여부 (기본값: True).

    primary·aux 에는 cls_score 가 계산하는 Accuracy·Precision·Recall·F1·ROC_AUC·
    PR_AUC·LogLoss·MCC 를 쓴다.

    Returns:
        DataFrame: Rank 순 비교표. 맨 앞에 `Rank`·`Group`(Contender=근소 격차 그룹 /
            Outside=그룹 외부), 맨 끝에 `{primary}_Gap`(1등 대비 격차, 양수일수록 나쁨) 컬럼.

    Raises:
        TypeError: estimator 가 리스트도 딕셔너리도 아닌 경우.
        ValueError: primary·aux 에 계산되지 않는 지표명을 준 경우.
    """
    # --- 1) 파라미터 검증 ---
    aux = _validate_metrics(primary, aux, _CLS_METRIC_SPECS)

    # --- 2) 모델별 점수 계산 ---
    # 딕셔너리면 키가 곧 이름이고, 리스트면 아래 루프에서 이름을 정한다
    if isinstance(estimator, dict):
        models = list(estimator.items())
    elif isinstance(estimator, list):
        models = [(None, m) for m in estimator]
    else:
        raise TypeError('estimator 는 모델의 리스트 또는 딕셔너리여야 합니다: '
                        f'{type(estimator).__name__}')

    score_tables = []
    for i, (name, model) in enumerate(models):
        # GridSearchCV·RandomizedSearchCV 등 탐색 객체면 최적 모델을 꺼내 쓴다.
        # 그대로 두면 모델명이 전부 'GridSearchCV' 로 찍혀 구분이 되지 않는다.
        best_model = getattr(model, 'best_estimator_', model)

        # 리스트로 받았으면 name_ 을 이름으로 쓴다. 탐색 객체는 내부 모델을 clone 해서
        # 재학습하므로 안쪽 name_ 은 사라진다. 탐색 객체 자신에 붙여둔 이름을 먼저 찾는다.
        if name is None:
            name = (getattr(model, 'name_', None)
                    or getattr(best_model, 'name_', None)
                    or f'Model {i + 1}')

        score_df = cls_score(best_model, x_test, y_test, average=average)
        score_df.reset_index(inplace=True)   # 모델 클래스명을 'Model' 컬럼으로 내린다
        score_df.index = [name]
        score_tables.append(score_df)

    final_score_table = concat(score_tables)
    final_score_table.index.name = 'name'

    # --- 3) 공용 엔진으로 순위 산정 (회귀와 같은 4단계 전략) ---
    return _rank_score_table(final_score_table, _CLS_METRIC_SPECS, primary, aux, verbose)


# --------------------------------------------------------
# 훈련·교차검증·검증 성능을 한 표로 비교해 과적합 여부를 판정
# --------------------------------------------------------
def cls_overfit(estimator, x_train, y_train, x_test, y_test,
                metrics=['F1', 'ROC_AUC', 'Accuracy'], average='auto',
                threshold=0.15, underfit_threshold=0.6,
                cv=5, fit_params=None, learning_curve=True,
                width=1280, height=640, grid=True, save_path=None, verbose=True):
    """Train / CV / Test 성능을 한 표로 보여주고 과적합 여부를 판정한다 (이진·다중분류).

    회귀의 reg_overfit 에 대응하며 판정 규칙도 같다. 훈련 성능과 교차검증(out-of-fold)
    성능의 격차로 판정하고, 검증 데이터(Test)는 표에 참고용으로만 싣는다.
        - 과소적합: 훈련 성능 자체가 낮음 (train ROC_AUC < underfit_threshold).
            ROC_AUC 0.5 는 동전 던지기와 같아, 기본값 0.6 은 '거의 학습하지 못한' 수준이다.
            다만 AUC 는 순위를 매기는 능력만 보므로, AUC 는 높은데 F1·Accuracy 의 Train
            값이 낮다면 임계값 0.5 가 이 데이터에 맞지 않는 것이니 표를 함께 읽는다.
        - 과대적합: 훈련↔CV 격차가 큼 (Gap% >= threshold).
        - 일반화: 위 둘 다 아님.

    교차검증 점수는 cls_score 와 완전히 같은 함수로 계산한다. 사이킷런의 scoring 문자열
    ('f1'·'roc_auc')은 이진 전용이거나 평균 방식이 달라 Train 과 CV 를 나란히 놓을 수
    없기 때문이다. 다중분류에는 계층별 비율을 유지하는 StratifiedKFold 가 자동 적용된다.

    Args:
        estimator: 학습된 분류 모델·파이프라인 또는 GridSearchCV 등 탐색 객체.
        x_train (DataFrame): 훈련 데이터의 독립변수.
        y_train (Series | ndarray): 훈련 데이터의 종속변수.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series | ndarray): 검증 데이터의 종속변수.
        metrics (list): 표시·판정할 지표 (기본값: ['F1', 'ROC_AUC', 'Accuracy']).
        average (str): 다중분류의 클래스 평균 방식 (기본값: 'auto').
        threshold (float): 과대적합으로 볼 Gap% (기본값: 0.15).
        underfit_threshold (float): 과소적합으로 볼 train ROC_AUC 하한 (기본값: 0.6).
        cv (int): 교차검증 폴드 수 (기본값: 5).
        fit_params (dict): CV 재학습에 넘길 인자 (예: {'model__cat_features': [...]}) (기본값: None).
        learning_curve (bool): 학습곡선 출력 여부 (기본값: True).
        width (int): 학습곡선 가로 크기(픽셀) (기본값: 1280).
        height (int): 학습곡선 세로 크기(픽셀) (기본값: 640).
        grid (bool): 학습곡선 격자 표시 여부 (기본값: True).
        save_path (str): 학습곡선 이미지 저장 경로 (기본값: None).
        verbose (bool): 판정 과정 출력 여부 (기본값: True).

    threshold·underfit_threshold 는 학술 표준이 아닌 경험칙이다. 임계값보다 학습곡선의
    추세가 더 믿을 만하다.

    Returns:
        DataFrame: index=Metric, 컬럼=[Train, CV, CV_Std, Test, Gap, Gap%, Overfit].
            `result.attrs['diagnosis']` 에 모델 수준 최종 진단이 담긴다.

    Raises:
        ValueError: metrics 에 지원하지 않는 지표명을 준 경우.
    """
    # --- 1) 탐색 객체면 최적 모델을 꺼내기 ---
    # 탐색 객체 그대로 교차검증하면 폴드마다 탐색을 다시 도는 중첩 CV 가 되어 버린다.
    model, _, base_est, search_params = _unwrap_estimator(estimator)

    # --- 2) 클래스 목록 확정 — 지표를 계산하려면 필요하다 ---
    classes = getattr(model, 'classes_', None)
    if classes is None:
        classes = np.unique(y_train.values.ravel() if isinstance(y_train, DataFrame)
                            else np.asarray(y_train).ravel())
    classes = np.asarray(classes)

    # --- 3) cls_score 와 같은 함수를 scorer 로 감싸 지표표 구성 ---
    # 지표표는 {지표: (최적 방향, scorer, fold 점수 보정 함수)} 형태로 공용 엔진에 넘긴다.
    # scorer 가 이미 cls_score 와 같은 단위로 돌려주므로 보정은 하지 않는다.
    has_proba = hasattr(base_est, 'predict_proba')
    has_margin = has_proba or hasattr(base_est, 'decision_function')

    metric_specs = {}
    for name, spec in _CLS_METRIC_SPECS.items():
        if not spec['needs_score'] or not has_margin or (name == 'LogLoss' and not has_proba):
            # 라벨로 계산하는 지표거나, 점수를 낼 수 없어 계산이 불가능한 지표.
            # 후자는 _cls_label_metric 이 이름을 알아보지 못해 NaN 을 돌려주고,
            # 그 행은 표에서 'N/A' 로 남는다
            scorer = make_scorer(_cls_label_metric, response_method='predict',
                                 name=name, average=average, classes=classes)
        else:
            scorer = make_scorer(_cls_score_metric,
                                 response_method='predict_proba' if has_proba
                                 else 'decision_function',
                                 name=name, average=average, classes=classes,
                                 is_proba=has_proba)

        metric_specs[name] = (spec['better'], scorer, lambda s: s)

    # --- 4) 공용 엔진에 넘겨 판정 (회귀와 같은 규칙) ---
    return _fit_diagnosis(
        base_est, search_params, x_train, y_train, x_test, y_test,
        metrics=metrics, metric_specs=metric_specs,
        score_func=lambda est, x, y: cls_score(est, x, y, average=average),
        underfit_metric='ROC_AUC', threshold=threshold,
        underfit_threshold=underfit_threshold,
        cv=cv, fit_params=fit_params, learning_curve=learning_curve,
        width=width, height=height, grid=grid, save_path=save_path, verbose=verbose)


# --------------------------------------------------------
# 베이스라인 모델 11종을 한 번에 학습·저장하고 성능 순위표를 만든다
# --------------------------------------------------------
def cls_baseline(project_name, x_train, y_train, x_test, y_test,
                 primary='F1', aux=['ROC_AUC', 'Accuracy'], average='auto', plot=True,
                 width=1280, height=640, save_path=None, verbose=True):
    """분류 베이스라인 모델을 모두 학습·저장하고, 성능 순위표와 비교 그래프를 만든다.

    회귀의 reg_baseline 에 대응한다. `{project_name}/baseline` 폴더를 만들어 모델별 pkl
    파일을 저장한다. 학습하는 모델은 선형 3종(logistic·ridge·sgd), 확률 1종(gaussiannb),
    비선형 2종(kneighbors·svc), 트리 1종(decisiontree), 앙상블 4종(randomforest·xgb·
    lgbm·catboost)의 11개다. 모델별 전처리 조합은 계열별 권장 조합으로 고정되어 있다.

    XGBoost 는 종속변수가 0부터 시작하는 연속된 정수여야 학습되므로, 라벨이 문자열이나
    띄엄띄엄한 정수면 모든 모델에 같은 라벨 인코딩을 적용하고 대응 관계를 출력한다.

    Args:
        project_name (str): 작업 폴더의 이름이 될 프로젝트명.
        x_train (DataFrame): 훈련 데이터의 독립변수.
        y_train (Series): 훈련 데이터의 종속변수.
        x_test (DataFrame): 검증 데이터의 독립변수.
        y_test (Series): 검증 데이터의 종속변수.
        primary (str): 순위를 가르는 주 지표 (기본값: 'F1').
        aux (list): 결함 판정에 쓸 보조 지표 (기본값: ['ROC_AUC', 'Accuracy']).
        average (str): 다중분류의 클래스 평균 방식 (기본값: 'auto').
        plot (bool): 성능 비교 그래프 출력 여부 (기본값: True).
        width (int): 그래프 가로 크기(픽셀) (기본값: 1280).
        height (int): 그래프 세로 크기(픽셀) (기본값: 640).
        save_path (str): 그래프 이미지 저장 경로 (기본값: None).
        verbose (bool): 모델별 학습 진행 상황 출력 여부 (기본값: True).

    SVC 는 확률을 얻으려고 내부에서 교차검증을 한 번 더 돌리므로(probability=True)
    표본이 많으면 11종 가운데 가장 오래 걸린다.

    마지막에 순위표(cls_compare_models 의 결과에 시각화용 `Score` 컬럼을 더한 표)를
    화면에 출력한다. 별도로 반환하는 값은 없다.
    """
    # 부스팅 3종은 무겁고 별도 설치가 필요한 패키지라 모듈 로드 시가 아니라 함수 안에서 import 한다
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    # --- 1) 학습 결과물을 담을 작업 폴더 생성 ---
    workdir = Path(project_name) / f'baseline'
    workdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f'작업 폴더: {workdir}')

    # --- 2) 종속변수를 XGBoost 가 받는 형태(0부터 시작하는 연속된 정수)로 맞춘다 ---
    # 라벨이 문자열('Yes'/'No')이거나 띄엄띄엄한 정수(1/2)면 XGBoost 만 학습에 실패한다.
    # 11종을 같은 조건에서 비교해야 하므로 훈련·검증 모두에 같은 인코딩을 적용한다
    y_labels = np.unique(y_train.values.ravel() if isinstance(y_train, DataFrame)
                         else np.asarray(y_train).ravel())

    if not (np.issubdtype(y_labels.dtype, np.integer)
            and np.array_equal(y_labels, np.arange(len(y_labels)))):
        encoder = LabelEncoder().fit(y_labels)
        y_train = Series(encoder.transform(np.asarray(y_train).ravel()),
                         index=getattr(y_train, 'index', None), name='y')
        y_test = Series(encoder.transform(np.asarray(y_test).ravel()),
                        index=getattr(y_test, 'index', None), name='y')

        if verbose:
            print('   ▷ 종속변수를 정수로 인코딩했습니다 (XGBoost 요구사항): '
                  f'{ {str(c): i for i, c in enumerate(encoder.classes_)} }')

    # 학습을 마친 모델을 {이름: 파이프라인} 으로 모아 마지막 비교표에 넘긴다
    models = {}

    # --- 3) 선형 계열 ---
    # 계수 해석과 규제의 전제가 되는 다중공선성을 VIF 로 제거하고, 더미 트랩도 함께 막는다.
    # 세 모델 모두 계수의 크기에 벌점을 매기므로 정규화가 반드시 필요하다
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('regression')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 1/11] : {model_name}')

    # RidgeClassifier 는 확률을 내지 못해 LogLoss 가 NaN 이 된다 (ROC_AUC 는 마진으로 계산)
    model = RidgeClassifier(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 2/11] : {model_name}')

    # loss='log_loss' 로 두어야 로지스틱 회귀와 같은 확률 출력을 얻는다 (기본값은 확률이 없다)
    model = SGDClassifier(loss='log_loss', random_state=RANDOM_STATE, n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 3/11] : {model_name}')

    # --- 4) 확률 계열 ---
    # 변수끼리 독립이라고 가정하고 확률을 직접 계산한다. 가정이 강한 대신 매우 빨라
    # '이보다 못하면 곤란한' 하한선 역할을 한다
    model = GaussianNB()
    model_name = model.__class__.__name__.lower()
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 4/11] : {model_name}')

    # --- 5) 비선형 계열 ---
    # 거리·마진으로 학습하는 모델이라 변수의 단위가 다르면 큰 값의 변수가 거리를 독점한다
    model = KNeighborsClassifier(n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 5/11] : {model_name}')

    # probability=True 는 확률을 얻기 위해 내부에서 교차검증을 한 번 더 돌린다 (느린 이유)
    model = SVC(probability=True, random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower()
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=True, drop_first=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 6/11] : {model_name}')

    # --- 6) 트리 계열 ---
    # 분기 기준이 값의 대소 관계뿐이라 정규화가 필요 없고, 더미도 전부 남겨야
    # 각 범주가 독립적인 분기 후보가 된다 (drop_first 를 쓰지 않는 이유)
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 7/11] : {model_name}')

    # --- 7) 앙상블 계열 ---
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 8/11] : {model_name}')

    model = XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [ 9/11] : {model_name}')

    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=False, encode=True,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False)

    if verbose:
        print(f'학습 완료 [10/11] : {model_name}')

    # CatBoost 는 범주형을 자체 방식(Ordered Target Statistics)으로 처리하므로
    # 더미 인코딩을 끄고, 어떤 컬럼이 범주형인지만 fit 인자로 알려준다.
    # 범주형 판정은 fit_pipeline 의 명목형 자동 선택과 같은 기준(category·object)으로 한다 —
    # my_qtcheck 는 category 만 보므로 CSV 에서 흔한 object 컬럼을 놓쳐 CatBoost 가 학습에 실패한다
    model = CatBoostClassifier(random_state=RANDOM_STATE, verbose=0)
    model_name = model.__class__.__name__.lower().removesuffix('classifier')
    models[model_name] = fit_pipeline(
        model=model, x_train=x_train, y_train=y_train,
        vif=True, scale=False, encode=False,
        name=model_name, save_path=workdir / f'{model_name}.pkl', verbose=False,
        model__cat_features=list(x_train.select_dtypes(include=['category', 'object']).columns))

    if verbose:
        print(f'학습 완료 [11/11] : {model_name}')

    # --- 8) 모델간 성능 비교 ---
    # 개별 모델의 지표는 출력하지 않고, 순위가 매겨진 표 하나만 남긴다
    score_table = cls_compare_models(models, x_test, y_test, primary=primary,
                                     aux=aux, average=average, verbose=False)

    # --- 9) 시각화용 점수 계산 ---
    # 분류 지표는 대부분 높을수록 좋아 그대로 그리면 되지만, LogLoss 만 낮을수록 좋아
    # 막대가 길수록 나쁜 모델이 된다. 역수를 취해 '클수록 좋은 값' 으로 방향을 통일한다.
    if primary == 'LogLoss':
        score_table['Score'] = 1 / score_table[primary]
        score_label = f'1/{primary}'
    else:
        score_table['Score'] = score_table[primary]
        score_label = primary

    # --- 10) 성능 비교 그래프 ---
    if plot:
        my_plot.barplot(score_table, y=score_table.index, x='Score', hue='Group',
                        palette='tab10', width=width, height=height,
                        title=f'모델 성능 비교({score_label} 기준)', save_path=save_path)

    # --- 11) 최종 순위표 출력 ---
    # 순위표가 이 함수의 결과물이므로 화면에 직접 출력한다.
    # 학습된 모델은 이미 pkl 로 저장했으니, 다시 쓸 때는 workdir 에서 load_model 로 불러온다
    print(f'\n모델 {len(models)}개 학습·저장 완료 → {workdir}')
    display(score_table)


# --------------------------------------------------------
# feature_importance : 모델 유형별 중요도 산출 기준
# --------------------------------------------------------
# 트리·부스팅 계열 → feature_importances_ (gain / MDI 등) 사용
_IMPORTANCE_TREE = {
    # 회귀
    'DecisionTreeRegressor', 'RandomForestRegressor',
    'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor',
    # 분류
    'DecisionTreeClassifier', 'RandomForestClassifier',
    'XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier',
}

# 선형 계열 → |coef_| 사용 (다중클래스면 클래스축 평균)
_IMPORTANCE_LINEAR = {
    # 회귀
    'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'SGDRegressor',
    # 분류
    'LogisticRegression', 'RidgeClassifier', 'SGDClassifier', 'LinearSVC',
}

# SHAP TreeExplainer 가 지원하는 트리 계열 — _IMPORTANCE_TREE 보다 넓다.
# feature_importance 의 분류(중요도 산출 기준)와 SHAP 의 분류(explainer 선택)는 목적이
# 다르다. 예를 들어 HistGradientBoosting 은 feature_importances_ 가 없어 중요도는 못
# 뽑지만 TreeExplainer 는 지원한다. 지원 여부는 shap 버전에 따라 달라지므로 이 집합은
# '먼저 시도해 볼 후보' 일 뿐이고, 실패하면 KernelExplainer 로 자동 폴백한다.
_SHAP_TREE = _IMPORTANCE_TREE | {
    # 회귀
    'ExtraTreeRegressor', 'ExtraTreesRegressor',
    'GradientBoostingRegressor', 'HistGradientBoostingRegressor',
    # 분류
    'ExtraTreeClassifier', 'ExtraTreesClassifier',
    'GradientBoostingClassifier', 'HistGradientBoostingClassifier',
}

# |coef_| 비교의 전제(동일 스케일) 충족 여부를 탐지할 때 쓰는 스케일러 클래스명
_FI_SCALER_CLASSES = {
    'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler',
    'Normalizer', 'PowerTransformer', 'QuantileTransformer',
}


# --------------------------------------------------------
# 트리·부스팅 모델의 중요도 배열과 기준 라벨을 반환
# --------------------------------------------------------
def _fi_tree_importance(model, model_class, importance_type):
    """트리·부스팅 모델에서 (중요도 배열, 사용된 기준 라벨)을 반환한다.

    importance_type='auto' 면 라이브러리별 권장 기준(XGBoost·LightGBM=gain)을 쓰고,
    'native' 면 모델 생성 시 설정된 feature_importances_ 를 그대로 쓴다.

    Args:
        model: 최종 추정기 (파이프라인이 아닌 모델 객체)
        model_class (str): 모델 클래스명
        importance_type (str): 'auto' 또는 'native'

    Returns:
        tuple: (중요도 ndarray, 기준 라벨 str)
    """
    use_native = importance_type == 'native'

    # XGBoost — 학습된 booster 에서 gain 을 직접 추출 (모델 속성은 건드리지 않는다)
    # get_score 는 분할에 쓰인 변수만 dict 로 주므로, booster 의 변수 순서대로 0 을 채워 정렬한다
    if model_class in ('XGBRegressor', 'XGBClassifier') and not use_native:
        booster = model.get_booster()
        score = booster.get_score(importance_type='gain')

        names = booster.feature_names
        if names is None:
            # 이름 없이 학습된 경우 f0, f1 … 로 대체
            names = []
            for i in range(model.n_features_in_):
                names.append(f'f{i}')

        imp = []
        for f in names:
            imp.append(score.get(f, 0.0))

        return np.array(imp, dtype=float), 'gain (xgboost·권장)'

    # LightGBM — 기본값 split(사용 횟수)은 고카디널리티 변수를 과대평가하므로 gain 을 직접 추출
    if model_class in ('LGBMRegressor', 'LGBMClassifier') and not use_native:
        imp = model.booster_.feature_importance(importance_type='gain')
        return np.asarray(imp, dtype=float).ravel(), 'gain (lightgbm·권장)'

    # 그 외 · native — 모델이 제공하는 feature_importances_ 를 그대로 사용
    imp = np.asarray(model.feature_importances_, dtype=float).ravel()

    labels = {
        'DecisionTreeRegressor':  'MDI(불순도 감소)',
        'RandomForestRegressor':  'MDI(불순도 감소)',
        'DecisionTreeClassifier': 'MDI(불순도 감소)',
        'RandomForestClassifier': 'MDI(불순도 감소)',
        'CatBoostRegressor':      'PredictionValuesChange',
        'CatBoostClassifier':     'PredictionValuesChange',
    }
    label = labels.get(model_class, 'feature_importances_')

    return imp, label


# --------------------------------------------------------
# 원핫 더미 컬럼명을 원본 컬럼명으로 되돌리는 매핑 생성
# --------------------------------------------------------
def _fi_dummy_to_origin(ohe):
    """학습된 OneHotEncoder 의 더미 출력명 → 원본 컬럼명 매핑을 만든다.

    get_feature_names_out 의 출력은 입력 컬럼 순서대로 묶여 있고, 입력 컬럼별 더미
    개수는 카테고리 수에서 drop 된 개수를 뺀 값이다. 이 개수만큼 출력명을 끊어
    원본 컬럼에 귀속시킨다.

    Args:
        ohe: 학습이 끝난 OneHotEncoder 객체

    Returns:
        dict: {더미 컬럼명: 원본 컬럼명}
    """
    in_cols = list(ohe.feature_names_in_)
    out_names = list(ohe.get_feature_names_out(in_cols))

    # drop=None 이면 None, drop='first'/'if_binary' 면 컬럼별 (드롭 인덱스 | None)
    drop_idx = getattr(ohe, 'drop_idx_', None)

    mapping = {}
    pos = 0

    for i, col in enumerate(in_cols):
        n_out = len(ohe.categories_[i])

        # 해당 컬럼에서 카테고리 하나가 제거되었으면 더미 개수도 하나 줄어든다
        if drop_idx is not None and drop_idx[i] is not None:
            n_out = n_out - 1

        for _ in range(n_out):
            mapping[out_names[pos]] = col
            pos = pos + 1

    return mapping


# --------------------------------------------------------
# 변환 후 변수명 → 원본 컬럼명 매핑 생성 (더미 합산용)
# --------------------------------------------------------
def _fi_name_map(base_est):
    """파이프라인 전처리기에서 원핫 더미의 원본 컬럼 매핑을 모아 반환한다.

    원핫 더미만 원본 컬럼으로 되돌린다. 연속형·passthrough 는 이름이 그대로 유지되므로
    매핑에 넣지 않고, PCA 출력(pca0, pca1 …)은 원본 복원이 불가능해 그대로 남는다.

    Args:
        base_est: 탐색 객체를 푼 추정기 (파이프라인이면 preprocessor 를 탐색)

    Returns:
        dict: {더미 컬럼명: 원본 컬럼명}. 매핑할 대상이 없으면 빈 dict.
    """
    name_map = {}

    if not isinstance(base_est, Pipeline):
        return name_map

    pre = base_est.named_steps.get('preprocessor')
    if pre is None:
        return name_map

    # 연속형·명목형 분기별 변환기를 돌며 OneHotEncoder 를 찾는다
    for _name, trans, _cols in pre.transformers_:
        if isinstance(trans, Pipeline):
            ohe = trans.named_steps.get('onehot')
        elif isinstance(trans, OneHotEncoder):
            ohe = trans
        else:
            ohe = None

        if ohe is not None and hasattr(ohe, 'categories_'):
            name_map.update(_fi_dummy_to_origin(ohe))

    return name_map


# --------------------------------------------------------
# 전처리 파이프라인에 스케일러가 포함되어 있는지 탐지
# --------------------------------------------------------
def _fi_has_scaler(base_est):
    """전처리 단계에 스케일러가 있는지 확인한다 (|coef_| 비교 전제 점검용).

    Args:
        base_est: 탐색 객체를 푼 추정기

    Returns:
        bool | None: 스케일러가 있으면 True, 파이프라인인데 없으면 False,
            파이프라인이 아니라 판단할 수 없으면 None.
    """
    if not isinstance(base_est, Pipeline):
        return None

    pre = base_est.named_steps.get('preprocessor')
    if pre is None:
        return None

    for _name, trans, _cols in pre.transformers_:
        if isinstance(trans, Pipeline):
            steps = list(trans.named_steps.values())
        else:
            steps = [trans]

        for step in steps:
            if type(step).__name__ in _FI_SCALER_CLASSES:
                return True

    return False


# --------------------------------------------------------
# 파이프라인·탐색 객체를 풀어 최종 모델과 변수명을 추출
# --------------------------------------------------------
def _fi_unwrap(estimator):
    """파이프라인·탐색 객체를 풀어 (모델, 변수명, 원본명 매핑, base_est, best_params).

    변수명은 모델 직전까지의 변환 결과 기준이다. 즉 모델이 실제로 학습한 피처 이름
    (원핫·PCA 변환 후 이름)이다.

    Args:
        estimator: 학습된 모델·파이프라인 또는 GridSearchCV 등 탐색 객체

    Returns:
        tuple: (model, feature_names, name_map, base_est, search_params)
    """
    model, pre, base_est, search_params = _unwrap_estimator(estimator)

    if pre is not None:
        # 모델 직전 단계까지의 출력 변수명 = 모델이 학습한 변수
        feature_names = np.asarray(pre.get_feature_names_out())
    elif hasattr(model, 'feature_names_in_'):
        # 단독 모델은 학습할 때 기억해 둔 컬럼명을 쓴다
        feature_names = np.asarray(model.feature_names_in_)
    else:
        feature_names = None

    name_map = _fi_name_map(base_est)

    return model, feature_names, name_map, base_est, search_params


# --------------------------------------------------------
# 변수 중요도를 산출해 상위 변수를 추려서 반환
# --------------------------------------------------------
def feature_importance(estimator, cum_ratio=0.9, top_n=None, aggregate_dummies=True,
                       importance_type='auto', max_display=30,
                       plot=True, palette=None, title=None, xlabel=None, ylabel=None,
                       width=1280, height=640, save_path=None, verbose=True):
    """학습된 회귀·분류 모델에서 변수 중요도를 도출해 상위 변수만 추려서 반환한다.

    중요도 산출 방식은 모델 유형에 따라 갈린다.
        - 트리·부스팅: feature_importances_
            (XGBoost·LightGBM 은 gain, sklearn 트리는 MDI, CatBoost 는 PredictionValuesChange)
        - 선형: |coef_| — 계수 비교가 공정하려면 변수가 스케일링되어 있어야 한다.
        - KNN·비선형 커널 SVR/SVC: 중요도를 정의할 수 없어 예외 발생.

    중요도 기준은 라이브러리마다 다르므로 모델 간 절대 비교가 아니라 동일 모델 내
    상대 순위로만 해석한다. 원핫 더미는 기본적으로 원본 컬럼 단위로 합산해, 재학습 시
    원본 컬럼 단위로 변수를 고를 수 있게 한다.

    채택 규칙: 중요도를 합 1 로 정규화 → 내림차순 정렬 → 누적 비율이 cum_ratio 에
    처음 도달하는 변수까지 채택(경계 변수 포함). top_n 을 주면 상위 top_n 개를 채택한다.

    Args:
        estimator: 학습된 회귀·분류 모델·파이프라인 또는 GridSearchCV 등 탐색 객체.
        cum_ratio (float): 채택할 누적 중요도 비율 (기본값: 0.9).
        top_n (int): 지정하면 누적 비율 대신 상위 top_n 개를 채택 (기본값: None).
        aggregate_dummies (bool): 원핫 더미를 원본 컬럼으로 합산할지 여부 (기본값: True).
        importance_type (str): 'auto' 면 라이브러리별 권장 기준, 'native' 면 모델 기본값 (기본값: 'auto').
        max_display (int): 그래프에 표시할 최대 변수 개수 (기본값: 30).
        plot (bool): 결과를 시각화할지 여부 (기본값: True).
        palette (str or list): 색상 팔레트 (기본값: None).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 라벨 (기본값: None).
        ylabel (str): y축 라벨 (기본값: None).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 640).
        save_path (str): 그래프 저장 경로 (기본값: None).
        verbose (bool): 채택 결과를 출력할지 여부 (기본값: True).

    Returns:
        DataFrame: 중요도 내림차순 전체 변수표. index=Feature,
            컬럼=[Importance, Ratio, CumRatio, 채택여부].
            `result.attrs['selected_features']` 에 채택된 변수명 리스트,
            `result.attrs['importance_metric']` 에 사용된 중요도 기준,
            `result.attrs['model_class']` 에 모델 클래스명,
            `result.attrs['scaled']` 에 스케일러 유무(True/False/None)가 담긴다.

    Raises:
        ValueError: cum_ratio 가 (0, 1] 밖이거나 중요도 총합이 0 인 경우.
        TypeError: 변수 중요도를 도출할 수 없는 모델인 경우.
    """
    # --- 1) 파라미터 검증 ---
    if not 0.0 < cum_ratio <= 1.0:
        raise ValueError(f"cum_ratio 는 (0.0, 1.0] 범위여야 합니다: {cum_ratio}")

    # --- 2) 언랩 — 최종 모델·변수명·더미 매핑 추출 ---
    model, feature_names, name_map, base_est, search_params = _fi_unwrap(estimator)
    model_class = type(model).__name__
    scaled = None   # 선형 모델일 때만 의미를 갖는다 (스케일러 유무)

    # --- 3) 중요도 산출 (모델 유형 분기) ---
    if hasattr(model, 'feature_importances_'):
        # 트리·부스팅 — gain / MDI / PredictionValuesChange
        importances, importance_metric = _fi_tree_importance(model, model_class, importance_type)
    elif hasattr(model, 'coef_'):
        # 선형 — 부호는 무관하므로 절대값을 쓰고, 다중클래스(2차원)면 클래스축 평균
        coef = np.abs(np.asarray(model.coef_, dtype=float))

        if coef.ndim == 2 and coef.shape[0] > 1:
            importances = coef.mean(axis=0)
        else:
            importances = coef.ravel()

        importance_metric = '|coef_|'

        # |coef_| 는 동일 스케일을 전제하므로 스케일러 유무를 확인해 둔다
        if model_class in _IMPORTANCE_LINEAR:
            scaled = _fi_has_scaler(base_est)
    else:
        # 두 속성이 모두 없으면 중요도를 정의할 수 없다 (대안: Permutation Importance / SHAP)
        if model_class in ('KNeighborsRegressor', 'KNeighborsClassifier'):
            hint = '거리 기반 모델이라 변수별 기여도를 분리할 수 없습니다'
        elif model_class in ('SVR', 'SVC'):
            hint = '비선형 커널은 coef_ 가 없습니다 (linear 커널만 도출 가능)'
        else:
            hint = 'feature_importances_·coef_ 속성이 모두 없습니다'

        raise TypeError(
            f"'{model_class}' 모델은 변수 중요도를 도출할 수 없습니다 — {hint}. "
            f"중요도 산출 가능: 트리·부스팅{sorted(_IMPORTANCE_TREE)} 또는 "
            f"선형{sorted(_IMPORTANCE_LINEAR)} 계열."
        )

    # 변환 후 변수명을 얻지 못했으면 임시 이름(x0, x1 …)을 만들어 쓴다
    names_synthesized = feature_names is None
    if names_synthesized:
        temp_names = []
        for i in range(len(importances)):
            temp_names.append(f'x{i}')
        feature_names = np.array(temp_names)

    # --- 4) 원핫 더미 → 원본 컬럼 단위로 중요도 합산 ---
    n_raw = len(feature_names)

    if aggregate_dummies and name_map:
        # 더미명을 원본명으로 바꿔(매핑이 없으면 자기 자신) 원본 단위로 더한다
        agg = {}
        for fname, imp in zip(feature_names, importances):
            origin = name_map.get(fname, fname)
            agg[origin] = agg.get(origin, 0.0) + imp

        feature_names = np.array(list(agg.keys()))
        importances = np.array(list(agg.values()), dtype=float)

    n_aggregated = len(feature_names)

    # --- 5) 정규화·정렬·누적 비율 ---
    total = importances.sum()
    if total <= 0:
        raise ValueError(
            f"중요도 총합이 0 입니다 ({model_class}). 모델이 학습되지 않았거나 "
            f"(Lasso·ElasticNet 등) 모든 계수가 0 으로 규제되었을 수 있습니다."
        )

    ratio = importances / total                 # 합 1 로 정규화
    order = np.argsort(ratio)[::-1]             # 내림차순 정렬 순서
    sorted_names = feature_names[order]
    sorted_ratio = ratio[order]
    sorted_imp = importances[order]
    cum = np.cumsum(sorted_ratio)               # 누적 비율
    n = len(sorted_ratio)

    # --- 6) 채택 개수 결정 (top_n 이 지정되면 우선) ---
    if top_n is not None:
        k = min(top_n, n)
        select_mode = f'top_n={top_n}'
    else:
        # 누적 비율이 cum_ratio 에 처음 도달하는 지점까지 (경계 변수 포함)
        k = int(np.searchsorted(cum, cum_ratio)) + 1
        k = max(1, min(k, n))
        select_mode = f'cum_ratio={cum_ratio:.0%}'

    selected_mask = np.arange(n) < k

    # --- 7) 결과표 구성 ---
    result = DataFrame({
        'Importance': sorted_imp,
        'Ratio': sorted_ratio,
        'CumRatio': cum,
        '채택여부': np.where(selected_mask, '채택', '탈락'),
    }, index=sorted_names)
    result.index.name = 'Feature'

    result.attrs['selected_features'] = sorted_names[selected_mask].tolist()
    result.attrs['importance_metric'] = importance_metric
    result.attrs['model_class'] = model_class
    result.attrs['aggregated_dummies'] = bool(aggregate_dummies and n_aggregated < n_raw)
    result.attrs['scaled'] = scaled

    if search_params is not None:
        result.attrs['search_params'] = search_params

    # 그래프에 표시할 개수 (채택 로직과 무관하게 가독성만 제한)
    if max_display:
        n_show = min(n, max_display)
    else:
        n_show = n

    # --- 8) 시각화 ---
    if plot:
        # 상위 n_show 개만 잘라 가로 막대그래프로 그린다 (채택·탈락을 색으로 구분)
        plot_df = result.head(n_show).reset_index()

        if title is None:
            title = f'Feature Importance: {model_class} ({importance_metric})'
            if n_show < n:
                title = f'{title} — 상위 {n_show}/{n}'

        if xlabel is None:
            xlabel = '중요도 비율'

        if ylabel is None:
            ylabel = '변수'

        fig, ax = my_plot.init(title=title, width=width, height=height,
                               xlabel=xlabel, ylabel=ylabel)
        my_plot.barplot(data=plot_df, x='Ratio', y='Feature', hue='채택여부',
                        palette=palette, errorbar=None, ax=ax)
        my_plot.show(save_path=save_path)

    # --- 9) 채택 결과 출력 ---
    if verbose:
        kept_ratio = float(cum[k - 1])

        print('\n' + '=' * 78)
        print(f'◆ Feature Importance: {model_class}  '
              f'(중요도 기준={importance_metric}, 채택 기준={select_mode})')
        print('=' * 78)

        # 선형 모델인데 스케일러가 없으면 |coef_| 비교 전제가 깨진다
        if scaled is False:
            print(f'   ⚠ {model_class} 는 |coef_| 기준이라 변수가 동일 스케일이어야 공정합니다.')
            print(f'     파이프라인에 스케일러가 없습니다 → fit_pipeline(scale=True) 로 재학습 권장.')

        if names_synthesized:
            print('   ⚠ 변환 후 변수명을 얻지 못해 임시 이름(x0, x1 …)을 사용합니다.')

        if search_params is not None:
            print(f'   ▷ 탐색 객체의 best_params: {search_params}')

        if result.attrs['aggregated_dummies']:
            print(f'   ▷ 원핫 더미 {n_raw}개 → 원본 컬럼 {n_aggregated}개로 중요도 합산')

        print(f'   전체 변수 {n}개 중 {k}개 채택 (누적 중요도 {kept_ratio:.2%} 설명)')
        print('=' * 78)
        print(f"   ▶ 채택된 변수: {result.attrs['selected_features']}")
        print('=' * 78 + '\n')

    return result



# --------------------------------------------------------
# 전처리 단계에 리샘플러(SMOTE 등)가 섞여 있는지 점검
# --------------------------------------------------------
def _shap_check_resampler(base_est, pre):
    """전처리 구간에 transform 이 없는 리샘플러가 있으면 우회 방법과 함께 예외를 던진다.

    클래스 불균형 실습에서 쓰는 imblearn 파이프라인은 SMOTE 처럼 fit_resample 만 있는
    단계를 포함한다. 이런 단계는 학습할 때만 동작하고 예측할 때는 건너뛰도록 설계되어
    파이프라인 슬라이스가 transform 을 제공하지 못한다.

    Args:
        base_est: 탐색 객체를 푼 파이프라인
        pre: 모델 직전까지를 잘라낸 전처리 파이프라인

    Raises:
        TypeError: 리샘플러가 있거나 전처리 구간에 transform 이 없는 경우.
    """
    samplers = []
    for name, step in base_est.steps[:-1]:
        # 리샘플러의 표식 — fit_resample 은 있고 transform 은 없다
        if hasattr(step, 'fit_resample') and not hasattr(step, 'transform'):
            samplers.append(f'{name}({type(step).__name__})')

    if not samplers and hasattr(pre, 'transform'):
        return

    if samplers:
        found = f"리샘플링 단계 {samplers} 가 들어 있습니다"
    else:
        found = 'transform 을 제공하지 않는 단계가 들어 있습니다'

    raise TypeError(
        f"SHAP 은 전처리를 통과시킨 피처공간에서 계산해야 하는데, 이 파이프라인에는 {found}. "
        f"리샘플링은 학습에만 쓰이고 변환 기능이 없어 SHAP 입력을 만들 수 없습니다.\n"
        f"   우회 1) 리샘플러를 뺀 전처리를 직접 적용한 뒤 모델 단계만 넘기세요 — 예:\n"
        f"        xt = pipe.named_steps['scaler'].transform(x)\n"
        f"        shap_analysis(pipe.named_steps['model'], DataFrame(xt, columns=x.columns, index=x.index))\n"
        f"   우회 2) 리샘플링 대신 class_weight='balanced' 로 학습한 모델을 쓰세요."
    )


# --------------------------------------------------------
# 파이프라인·탐색 객체를 풀어 모델과 모델공간 입력을 준비
# --------------------------------------------------------
def _shap_prepare(estimator, x):
    """파이프라인·탐색 객체를 풀어 (모델, 모델공간 입력, base_est, best_params) 를 반환한다.

    SHAP 은 모델이 실제 학습한 피처공간에서 계산해야 의미가 있으므로, 전처리(스케일·
    원핫·PCA)를 통과시킨 결과를 변환 후 변수명으로 DataFrame 화해 돌려준다.
    전처리 출력이 DataFrame 이면 dtype 을 그대로 보존한다 — float 로 강제 변환하면
    CatBoost 의 네이티브 범주형이 깨져 TreeExplainer 가 이를 거부하기 때문이다.

    Args:
        estimator: 학습된 모델·파이프라인 또는 GridSearchCV 등 탐색 객체
        x (DataFrame): 설명에 사용할 원본 입력

    Returns:
        tuple: (model, x_df, base_est, search_params)

    Raises:
        TypeError: 파이프라인에 리샘플러(SMOTE 등)가 들어 있어 전처리를 통과시킬 수 없는 경우.
    """
    model, pre, base_est, search_params = _unwrap_estimator(estimator)

    if pre is not None:
        # imblearn 의 SMOTE·언더샘플러 등 리샘플러는 fit_resample 만 있고 transform 이 없다.
        # 그대로 두면 pre.transform(x) 가 알아보기 어려운 AttributeError 로 터지므로,
        # 어느 단계가 문제인지와 우회 방법을 알려주고 먼저 멈춘다
        _shap_check_resampler(base_est, pre)

        x_trans = pre.transform(x)
        feat_names = list(pre.get_feature_names_out())
    else:
        # 파이프라인이 아닌 단독 모델은 입력을 그대로 쓴다
        x_trans = x

        if isinstance(x, DataFrame):
            feat_names = list(x.columns)
        else:
            feat_names = None

    if isinstance(x_trans, DataFrame):
        # 전처리가 pandas 로 출력 — 컬럼명·인덱스·dtype 을 그대로 보존
        x_df = x_trans.copy()
    else:
        # ndarray·희소행렬로 나온 경우 밀집 변환 후 DataFrame 으로 감싼다
        if hasattr(x_trans, 'toarray'):
            x_trans = x_trans.toarray()

        arr = np.asarray(x_trans)

        if feat_names is None or len(feat_names) != arr.shape[1]:
            feat_names = []
            for i in range(arr.shape[1]):
                feat_names.append(f'x{i}')

        if isinstance(x, DataFrame):
            index = x.index
        else:
            index = None

        x_df = DataFrame(arr, columns=feat_names, index=index)

    return model, x_df, base_est, search_params


# --------------------------------------------------------
# 모델 유형에 맞는 SHAP explainer 생성
# --------------------------------------------------------
def _shap_make_explainer(model, model_class, background):
    """모델 유형별로 적합한 explainer 를 만들어 (explainer, 종류 라벨, 폴백 사유) 를 반환한다.

    트리·부스팅은 TreeExplainer(정확·고속), 선형은 LinearExplainer(배경분포 사용),
    그 외(KNN·SVM 등)는 모델 구조를 따지지 않는 KernelExplainer 로 폴백한다.

    트리 판정을 클래스명 목록으로만 하면 목록에 없는 트리 계열(ExtraTrees·
    GradientBoosting·HistGradientBoosting 등)이 전부 KernelExplainer 로 떨어져
    수십~수백 배 느려지고 결과도 근사값이 된다. 그래서 트리로 볼 만한 모델은 일단
    TreeExplainer 를 시도하고, shap 이 거부할 때만 KernelExplainer 로 내려간다
    (예: GradientBoostingClassifier 는 이진만 지원, AdaBoost 는 미지원).

    Args:
        model: 최종 추정기
        model_class (str): 모델 클래스명
        background (DataFrame): 기준분포로 쓸 모델공간 표본

    Returns:
        tuple: (explainer 객체, 종류 라벨 str, 폴백 사유 str | None)
    """
    import shap

    if model_class in _IMPORTANCE_LINEAR:
        # LinearExplainer 는 kmeans 요약을 받지 못하므로 원시 표본을 그대로 넘긴다
        return shap.LinearExplainer(model, background), 'LinearExplainer', None

    # 트리 후보 — 알려진 목록에 있거나, 목록에 없어도 트리의 표식(feature_importances_)이 있으면 시도
    fallback_reason = None

    if model_class in _SHAP_TREE or hasattr(model, 'feature_importances_'):
        try:
            return shap.TreeExplainer(model), 'TreeExplainer', None
        except Exception as e:
            # shap 이 지원하지 않는 구조 — 아래 KernelExplainer 로 내려가되 이유를 남긴다
            fallback_reason = f'TreeExplainer 거부 — {type(e).__name__}: {e}'

    # 폴백 — 모델 무관 KernelExplainer. 분류는 확률, 회귀는 예측값을 설명 대상으로 삼는다
    if hasattr(model, 'predict_proba'):
        f = model.predict_proba
    elif hasattr(model, 'decision_function'):
        f = model.decision_function
    else:
        f = model.predict

    # Kernel 은 (배경 수 × 표본 수) 에 비례해 느리므로 배경을 kmeans 로 한 번 더 줄인다
    bg = shap.kmeans(background, min(50, len(background)))

    return shap.KernelExplainer(f, bg), 'KernelExplainer', fallback_reason


# --------------------------------------------------------
# SHAP 출력에서 설명 대상 클래스 한 개를 골라내기
# --------------------------------------------------------
def _shap_select_class(values, expected_value, task, class_index, n_classes):
    """SHAP 출력에서 설명할 클래스 1개의 (n, f) 슬라이스와 base value 를 골라낸다.

    회귀는 (n, f), 분류는 (n, f, c) 배열이나 클래스별 리스트로 나온다.

    Args:
        values: explainer.shap_values 의 반환값
        expected_value: explainer 의 base value
        task (str): 'regression' 또는 'classification'
        class_index (int): 설명할 클래스 인덱스
        n_classes (int): 클래스 개수

    Returns:
        tuple: (shap_2d ndarray, base_value float, 사용된 클래스 인덱스 | None,
            단일출력 여부 bool). 단일출력이면 클래스축이 없어 값이 항상 마지막 클래스
            (이진의 양성) 기준이므로, 다른 클래스를 보려면 호출하는 쪽에서 뒤집어야 한다.
    """
    # 구버전 형태 — 클래스별 (n, f) 리스트
    if isinstance(values, list):
        if task == 'classification':
            idx = class_index
        else:
            idx = 0

        arr = np.asarray(values[idx])

        if np.ndim(expected_value):
            ev = expected_value[idx]
        else:
            ev = expected_value

        if task == 'classification':
            used = idx
        else:
            used = None

        return arr, float(np.ravel(ev)[0]), used, False

    arr = np.asarray(values)

    # (n, f, c) — 다중 출력
    if arr.ndim == 3:
        if task == 'classification':
            idx = class_index
        else:
            idx = arr.shape[2] - 1

        ev_arr = np.atleast_1d(expected_value)

        if idx < ev_arr.shape[0]:
            ev = ev_arr[idx]
        else:
            ev = ev_arr[-1]

        if task == 'classification':
            used = idx
        else:
            used = None

        return arr[:, :, idx], float(ev), used, False

    # (n, f) — 회귀 또는 이진 분류의 단일 출력.
    # 이진 분류의 단일 출력은 XGBoost·LightGBM·CatBoost·GradientBoosting 처럼
    # '양성 클래스 하나' 만 내보내는 모델에서 나온다. 클래스축이 없으므로 여기서는
    # 고를 것이 없고, class_index 반영은 호출하는 쪽이 부호를 뒤집어 처리한다.
    if np.ndim(expected_value):
        ev = float(np.ravel(expected_value)[0])
    else:
        ev = float(expected_value)

    if task == 'classification':
        used = n_classes - 1 if n_classes else class_index
        single = True
    else:
        used = None
        single = False

    return arr, ev, used, single


# --------------------------------------------------------
# SHAP 값이 어떤 단위인지 실제 예측과 대조해 판정
# --------------------------------------------------------
def _shap_output_space(model, task, values, expected_value, x_explain):
    """SHAP 값의 단위를 모델의 실제 출력과 대조해 알아낸다.

    SHAP 은 '가산성' 을 만족한다 — 행마다 (base value + 그 행의 SHAP 합) 이 모델 출력과
    같아진다. 그래서 이 합을 확률·로그오즈·예측값과 차례로 맞춰보면 단위를 역으로 알 수
    있다. 단위는 모델과 explainer 조합에 따라 갈린다.
        - RandomForest·DecisionTree·ExtraTrees + TreeExplainer → 확률
        - XGBoost·LightGBM·CatBoost·GradientBoosting + TreeExplainer → 로그오즈
        - LogisticRegression 등 + LinearExplainer → 로그오즈
        - KernelExplainer → predict_proba 를 쓰면 확률, decision_function 을 쓰면 마진
    같은 데이터·같은 문제라도 모델을 바꾸면 base value 0.45 가 '확률 45%' 일 수도
    '로그오즈 0.45(≈ 확률 61%)' 일 수도 있어, 표시 없이는 해석이 어긋난다.

    Args:
        model: 최종 추정기
        task (str): 'regression' 또는 'classification'
        values: 클래스 선택 전의 explainer.shap_values 반환값
        expected_value: explainer 의 base value (다중 출력이면 배열)
        x_explain (DataFrame): SHAP 을 계산한 모델공간 입력

    Returns:
        str: 단위 라벨 — '확률' · '로그오즈' · '마진' · '예측값(y 단위)' · '알 수 없음'.
    """
    unknown = '알 수 없음'

    try:
        # 클래스별 리스트로 온 경우 (n, f, c) 배열로 합친다
        if isinstance(values, list):
            arr = np.stack([np.asarray(v, dtype=float) for v in values], axis=2)
        else:
            arr = np.asarray(values, dtype=float)

        ev = np.atleast_1d(np.asarray(expected_value, dtype=float))

        # 가산성 — 행별 (base + SHAP 합) = 모델 출력
        if arr.ndim == 3:
            preds = arr.sum(axis=1) + ev[:arr.shape[2]][None, :]
        else:
            preds = arr.sum(axis=1) + ev[0]
    except Exception:
        return unknown

    # --- 회귀 — y 단위 예측값과 맞아떨어지는지만 본다 ---
    if task != 'classification':
        try:
            y_hat = np.asarray(model.predict(x_explain), dtype=float).ravel()
        except Exception:
            return unknown

        scale = float(np.abs(y_hat).mean()) + 1e-9

        if float(np.abs(preds.ravel() - y_hat).mean()) < scale * 1e-3:
            return '예측값(y 단위)'

        return unknown

    # --- 분류 — 확률 · 로그오즈 · 마진 순으로 후보를 세워 가장 잘 맞는 것을 고른다 ---
    candidates = {}

    try:
        proba = np.asarray(model.predict_proba(x_explain), dtype=float)
    except Exception:
        proba = None

    if proba is not None:
        if preds.ndim == 2:
            # 다중 출력 — 클래스 전체를 한꺼번에 비교한다
            target = proba[:, :preds.shape[1]]
            # 로그오즈(마진)라면 softmax 를 씌워야 확률이 된다
            shift = preds - preds.max(axis=1, keepdims=True)
            exp = np.exp(shift)
            softmax = exp / exp.sum(axis=1, keepdims=True)
        else:
            # 단일 출력 — 양성(마지막) 클래스 확률과 비교한다
            target = proba[:, -1]
            # 로그오즈라면 시그모이드를 씌워야 확률이 된다 (오버플로 방지용 클리핑)
            softmax = 1.0 / (1.0 + np.exp(-np.clip(preds, -500, 500)))

        candidates['확률'] = float(np.abs(preds - target).mean())
        candidates['로그오즈'] = float(np.abs(softmax - target).mean())

    if hasattr(model, 'decision_function'):
        try:
            margin = np.asarray(model.decision_function(x_explain), dtype=float)

            if margin.ndim == preds.ndim:
                candidates['마진'] = float(np.abs(preds - margin).mean())
        except Exception:
            pass

    if not candidates:
        return unknown

    best = min(candidates, key=candidates.get)

    # 어느 후보와도 맞지 않으면 단위를 단정하지 않는다
    if candidates[best] > 0.05:
        return unknown

    return best


# --------------------------------------------------------
# 그래프·출력에 붙일 단위 꼬리표 만들기
# --------------------------------------------------------
def _shap_unit_tag(task, output_space):
    """SHAP 값의 단위를 축 라벨·제목에 덧붙일 꼬리표로 만든다.

    회귀는 언제나 y 단위라 굳이 적지 않는다. 단위가 갈리는 분류와, 단위를 확정하지
    못한 경우에만 표기해 그래프를 어지럽히지 않는다.

    Args:
        task (str): 'regression' 또는 'classification'
        output_space (str): _shap_output_space 가 판정한 단위

    Returns:
        str: ' [로그오즈]' 같은 꼬리표. 표기할 필요가 없으면 빈 문자열.
    """
    if task == 'classification' or output_space == '알 수 없음':
        return f' [{output_space}]'

    return ''


# --------------------------------------------------------
# 단일 출력 이진 분류에서 반대 클래스로 뒤집기
# --------------------------------------------------------
def _shap_flip_class(shap_2d, base_value, output_space):
    """양성 클래스 기준 SHAP 값을 음성 클래스 기준으로 뒤집는다.

    XGBoost·LightGBM 등은 이진 분류에서도 출력이 하나라 SHAP 값이 항상 양성 클래스
    기준으로 나온다. 두 클래스의 출력은 서로 반대로 움직이므로(로그오즈는 부호 반전,
    확률은 1에서 뺀 값) 기여도의 부호를 뒤집어 반대 클래스 설명으로 바꿀 수 있다.

    Args:
        shap_2d (ndarray): (n, f) 양성 클래스 기준 SHAP 값
        base_value (float): 양성 클래스 기준 base value
        output_space (str): _shap_output_space 가 판정한 단위

    Returns:
        tuple: (뒤집은 SHAP 값, 뒤집은 base value)

    Raises:
        ValueError: 단위를 알 수 없어 뒤집는 규칙을 정할 수 없는 경우.
    """
    if output_space == '확률':
        # 확률은 두 클래스의 합이 1 — base 는 1 에서 빼고 기여는 부호만 뒤집는다
        return -shap_2d, 1.0 - base_value

    if output_space in ('로그오즈', '마진'):
        # 로그오즈·마진은 두 클래스가 부호만 반대 — base 도 함께 뒤집는다
        return -shap_2d, -base_value

    raise ValueError(
        f"이 모델은 이진 분류에서도 출력이 하나라 SHAP 값이 양성 클래스 기준으로만 "
        f"나오는데, 단위를 판정하지 못해(={output_space}) 반대 클래스로 뒤집을 수 없습니다. "
        f"class_index 를 지정하지 말고 기본값(양성 클래스)으로 해석하세요."
    )


# --------------------------------------------------------
# 여러 장을 저장할 때 파일명에 꼬리표 붙이기
# --------------------------------------------------------
def _shap_tag_path(save_path, tag):
    """저장 경로의 파일명 뒤에 꼬리표를 붙여 여러 장이 서로 덮어쓰지 않게 한다.

    Args:
        save_path (str): 원래 저장 경로 (None 이면 저장하지 않음)
        tag (str): 파일명에 덧붙일 꼬리표

    Returns:
        str | None: 꼬리표가 붙은 경로. save_path 가 없으면 None.
    """
    if not save_path:
        return None

    path = Path(save_path)

    return str(path.with_name(f'{path.stem}_{tag}{path.suffix}'))


# --------------------------------------------------------
# 전역 영향력 막대그래프 (mean|SHAP| + 누적 비율)
# --------------------------------------------------------
def _shap_bar_plot(summary, model_class, cls_tag, unit_tag, cum_ratio, max_display,
                   palette, title, xlabel, ylabel, width, height, save_path):
    """변수별 mean|SHAP| 막대와 누적 비율 텍스트를 그린다 (채택·탈락 색 구분).

    누적 비율이 cum_ratio 에 처음 도달하는 지점까지를 '채택' 으로 표시한다 —
    shap_dependence_plot 이 주변수를 고르는 규칙과 같아, 어떤 변수가 뒤이어 쓰일지
    미리 보여주는 셈이다.

    Args:
        summary (DataFrame): shap_analysis 가 만든 요약표 (mean_abs_shap 내림차순)
        model_class (str): 모델 클래스명
        cls_tag (str): 분류일 때 제목에 붙일 클래스 꼬리표
        unit_tag (str): SHAP 값의 단위를 축 라벨에 붙일 꼬리표
        cum_ratio (float): 채택 기준이 되는 누적 비율
        max_display (int): 표시할 최대 변수 개수
        palette (str or list): 색상 팔레트
        title (str): 그래프 제목
        xlabel (str): x축 라벨
        ylabel (str): y축 라벨
        width (int): 그래프 너비
        height (int): 그래프 높이
        save_path (str): 그래프 저장 경로
    """
    n_all = len(summary)

    # 채택 개수 — 누적 비율이 cum_ratio 에 처음 도달하는 지점까지 (경계 포함)
    k = int(np.searchsorted(summary['cum_ratio'].values, cum_ratio)) + 1
    k = max(1, min(k, n_all))

    if max_display:
        n_show = min(n_all, max_display)
    else:
        n_show = n_all

    plot_df = summary.head(n_show).reset_index()

    # 표시할 막대의 채택 여부를 문자열로 만들어 색 구분에 쓴다
    plot_df['채택여부'] = np.where(np.arange(n_show) < k, '채택', '탈락')

    if title is None:
        title = f'SHAP Bar (mean|SHAP| · 누적 {cum_ratio:.0%} 채택): {model_class}{cls_tag}'
        if n_show < n_all:
            title = f'{title} — 상위 {n_show}/{n_all}'

    if xlabel is None:
        xlabel = f'mean|SHAP|{unit_tag}'

    if ylabel is None:
        ylabel = '변수'

    fig, ax = my_plot.init(title=title, width=width, height=height,
                           xlabel=xlabel, ylabel=ylabel)
    my_plot.barplot(data=plot_df, x='mean_abs_shap', y='Feature', hue='채택여부',
                    palette=palette, errorbar=None, ax=ax)

    # 막대 오른쪽에 누적 비율을 적어 '상위 k개가 영향력의 N% 를 설명' 을 읽을 수 있게 한다
    vmax = float(plot_df['mean_abs_shap'].max())
    ax.set_xlim(0, vmax * 1.20)

    for i in range(n_show):
        value = float(plot_df['mean_abs_shap'].iloc[i])
        cum_text = f"{plot_df['cum_ratio'].iloc[i]:.0%}"
        ax.text(value + vmax * 0.01, i, cum_text,
                va='center', ha='left', fontsize=10, color='tab:red')

    my_plot.show(save_path=save_path)


# --------------------------------------------------------
# 전역 영향력 파레토 차트 (비율 막대 + 누적 비율 선)
# --------------------------------------------------------
def _shap_pareto_plot(summary, model_class, cls_tag, unit_tag, max_display,
                      palette, title, xlabel, ylabel, width, height, save_path):
    """막대(mean|SHAP| 비율)와 선(누적 비율)을 겹쳐 파레토 차트를 그린다.

    '상위 몇 개 변수가 전체 영향력의 몇 %를 설명하는지' 를 한눈에 보여준다.

    Args:
        summary (DataFrame): shap_analysis 가 만든 요약표
        model_class (str): 모델 클래스명
        cls_tag (str): 분류일 때 제목에 붙일 클래스 꼬리표
        unit_tag (str): SHAP 값의 단위를 축 라벨에 붙일 꼬리표
        max_display (int): 표시할 최대 변수 개수
        palette (str or list): 색상 팔레트
        title (str): 그래프 제목
        xlabel (str): x축 라벨
        ylabel (str): y축 라벨
        width (int): 그래프 너비
        height (int): 그래프 높이
        save_path (str): 그래프 저장 경로
    """
    n_all = len(summary)

    if max_display:
        n_show = min(n_all, max_display)
    else:
        n_show = n_all

    plot_df = summary.head(n_show).reset_index()

    if title is None:
        title = f'SHAP Cumulative (Pareto): {model_class}{cls_tag}'

    if xlabel is None:
        xlabel = '변수'

    if ylabel is None:
        ylabel = f'mean|SHAP|{unit_tag} 비율'

    # 좌축(막대)·우축(누적선)을 함께 쓰기 위해 twinx 로 축을 두 개 만든다
    fig, (ax_left, ax_right) = my_plot.init(title=title, width=width, height=height,
                                            xlabel=xlabel, ylabel=ylabel, twinx=True)

    my_plot.barplot(data=plot_df, x='Feature', y='ratio',
                    palette=palette, errorbar=None, ax=ax_left)
    ax_left.set_xlabel(xlabel, fontsize=16)
    ax_left.set_xticks(range(n_show))
    ax_left.set_xticklabels(plot_df['Feature'], rotation=45, ha='right', fontsize=10)

    xs = np.arange(n_show)
    cum = plot_df['cum_ratio'].values
    ax_right.plot(xs, cum, color='tab:red', marker='o', linewidth=2, label='누적 비율')
    ax_right.set_ylim(0, 1.05)
    ax_right.set_ylabel('누적 비율', fontsize=14)
    ax_right.grid(False)        # 격자가 겹치지 않도록 우축 격자는 끈다

    # 꺾은선 위에 누적 비율을 텍스트로 표기
    for i in range(n_show):
        ax_right.annotate(f'{cum[i]:.0%}', (xs[i], cum[i]), textcoords='offset points',
                          xytext=(0, 7), ha='center', fontsize=9, color='tab:red')

    my_plot.show(save_path=save_path)


# --------------------------------------------------------
# SHAP 값을 계산해 변수 기여도 요약표를 반환
# --------------------------------------------------------
def shap_analysis(estimator, x, max_samples=200, background_samples=100, class_index=None,
                  bar=True, beeswarm=True, cumulative=False, cum_ratio=0.9, max_display=20,
                  palette=None, title=None, xlabel=None, ylabel=None,
                  width=1280, height=640, save_path=None, verbose=True,
                  random_state=RANDOM_STATE):
    """학습된 회귀·분류 모델을 SHAP 으로 분석해 변수 기여도 요약표를 반환한다.

    모델 유형에 따라 explainer 를 자동으로 고른다.
        - 트리·부스팅 → TreeExplainer (정확·고속). shap 이 거부하면 Kernel 로 폴백한다
        - 선형 → LinearExplainer (배경분포 사용)
        - KNN·SVM 등 → KernelExplainer (모델 구조를 따지지 않지만 느림)

    파이프라인이면 전처리를 통과시킨 '모델 학습 피처공간' 에서 SHAP 을 계산하고
    변환 후 변수명으로 설명한다. 탐색 객체면 best_estimator_ 를 자동으로 푼다.

    분류에서 SHAP 값의 단위는 모델·explainer 조합에 따라 확률일 수도 로그오즈일 수도
    있다. 실제 예측과 대조해 단위를 판정한 뒤 `attrs['output_space']` 에 담고, 그래프
    축과 요약 출력에도 함께 표시한다.

    Args:
        estimator: 학습된 회귀·분류 모델·파이프라인 또는 GridSearchCV 등 탐색 객체.
        x (DataFrame): 설명에 사용할 원본 입력 (보통 x_train 또는 x_test).
        max_samples (int): SHAP 을 계산할 행 수 상한 (기본값: 200). None 이면 전체 사용.
        background_samples (int): 기준분포로 쓸 표본 수 (기본값: 100).
        class_index (int): 분류에서 설명할 클래스 인덱스 (기본값: None → 마지막 클래스).
            출력이 하나뿐인 이진 분류(XGBoost 등)에서는 부호를 뒤집어 반영한다.
        bar (bool): mean|SHAP| 막대그래프를 그릴지 여부 (기본값: True).
        beeswarm (bool): SHAP 값의 방향·분포를 점으로 그릴지 여부 (기본값: True).
        cumulative (bool): 파레토 차트를 그릴지 여부 (기본값: False).
        cum_ratio (float): 막대그래프의 채택·탈락을 가르는 누적 비율 (기본값: 0.9).
        max_display (int): 그래프에 표시할 최대 변수 개수 (기본값: 20).
        palette (str or list): 색상 팔레트 (기본값: None).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 라벨 (기본값: None).
        ylabel (str): y축 라벨 (기본값: None).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 640).
        save_path (str): 그래프 저장 경로 (기본값: None). 여러 장이면 종류 꼬리표가 붙는다.
        verbose (bool): 분석 요약을 출력할지 여부 (기본값: True).
        random_state (int): 행·배경 샘플링 재현용 시드 (기본값: RANDOM_STATE).

    Returns:
        DataFrame: mean_abs_shap 내림차순 요약표. index=Feature, 컬럼=
            [mean_abs_shap, ratio, cum_ratio, mean_shap, std_shap, direction, cv, stability].
            `attrs['shap_values']` 에 (n, f) 기여도 배열, `attrs['data']` 에 모델공간 입력,
            `attrs['expected_value']` 에 base value, `attrs['feature_names']`,
            `attrs['explainer_type']`, `attrs['model_class']`, `attrs['task']`,
            `attrs['class_index']`, `attrs['class_names']`, `attrs['output_space']`
            (SHAP 값의 단위) 가 함께 담긴다.

    Raises:
        ValueError: cum_ratio 가 (0, 1] 밖이거나 class_index 가 클래스 범위 밖인 경우.
        TypeError: 파이프라인에 리샘플러(SMOTE 등)가 들어 있어 전처리를 통과시킬 수 없는 경우.
    """
    import shap

    # --- 1) 파라미터 검증 ---
    if not 0.0 < cum_ratio <= 1.0:
        raise ValueError(f"cum_ratio 는 (0.0, 1.0] 범위여야 합니다: {cum_ratio}")

    # --- 2) 언랩 — 최종 모델과 모델공간 입력 준비 ---
    model, x_df, base_est, search_params = _shap_prepare(estimator, x)
    model_class = type(model).__name__

    if is_classifier(model):
        task = 'classification'
    else:
        task = 'regression'

    class_names = list(getattr(model, 'classes_', []))
    n_classes = len(class_names)

    # 분류면 설명할 클래스를 정한다 (기본은 마지막 = 이진에서 양성 클래스)
    if task == 'classification' and n_classes:
        if class_index is None:
            class_index = n_classes - 1
        elif not 0 <= class_index < n_classes:
            raise ValueError(
                f"class_index 는 0..{n_classes - 1} 범위여야 합니다 (classes_={class_names}).")

    # --- 3) 행 샘플링과 배경표본 ---
    # SHAP 은 행 수에 비례해 느려지므로 재현 가능한 샘플링으로 계산량을 줄인다
    rng = np.random.RandomState(random_state)

    if max_samples is not None and len(x_df) > max_samples:
        pick = rng.choice(len(x_df), size=max_samples, replace=False)
        pick.sort()
        x_explain = x_df.iloc[pick]
    else:
        x_explain = x_df

    # 배경분포 — Linear·Kernel explainer 가 '기준이 되는 평균 예측' 을 잡는 데 쓴다
    if len(x_df) > background_samples:
        bg_pick = rng.choice(len(x_df), size=background_samples, replace=False)
        bg_pick.sort()
        background = x_df.iloc[bg_pick]
    else:
        background = x_df

    # --- 4) explainer 생성과 SHAP 값 계산 ---
    explainer, explainer_type, fallback_reason = _shap_make_explainer(
        model, model_class, background)

    if explainer_type == 'KernelExplainer':
        raw = explainer.shap_values(x_explain, silent=not verbose)
    else:
        raw = explainer.shap_values(x_explain)

    expected_value = getattr(explainer, 'expected_value', 0.0)

    shap_2d, base_value, used_class, single_output = _shap_select_class(
        raw, expected_value, task, class_index, n_classes)
    shap_2d = np.asarray(shap_2d, dtype=float)

    # SHAP 값의 단위(확률·로그오즈·예측값)를 실제 예측과 대조해 판정한다.
    # 클래스를 뒤집기 전의 원본으로 판정해야 모델 출력과 그대로 맞아떨어진다
    output_space = _shap_output_space(model, task, raw, expected_value, x_explain)

    # 단일 출력 이진 분류에서 음성 클래스를 요청했으면 부호를 뒤집어 준다.
    # (XGBoost·LightGBM 등은 클래스축이 없어 그냥 두면 class_index 가 무시된다)
    if single_output and task == 'classification' and class_index != used_class:
        if n_classes > 2:
            raise ValueError(
                f"'{model_class}' 의 SHAP 출력에 클래스축이 없어 class_index={class_index} 를 "
                f"선택할 수 없습니다 (클래스 {n_classes}개). class_index 를 비워 두세요.")

        shap_2d, base_value = _shap_flip_class(shap_2d, base_value, output_space)
        used_class = class_index

    # --- 5) 요약표 작성 (영향력·방향·안정성) ---
    feat_names = list(x_explain.columns)
    shap_df = DataFrame(shap_2d, columns=feat_names, index=x_explain.index)

    mean_abs = shap_df.abs().mean().values      # 영향력 크기 (부호 무관)
    mean_s = shap_df.mean().values              # 평균 기여 (부호 = 방향)
    std_s = shap_df.std().values                # 기여의 흔들림
    direction = np.sign(mean_s)

    summary = DataFrame({
        'mean_abs_shap': mean_abs,
        'mean_shap': mean_s,
        'std_shap': std_s,
        'direction': np.where(direction > 0, '증가', np.where(direction < 0, '감소', '중립')),
        # 변동계수 — 평균 기여보다 흔들림이 크면 비선형·상호작용을 의심한다
        'cv': std_s / (mean_abs + 1e-9),
    }, index=feat_names)

    summary['stability'] = np.where(summary['cv'] < 1, '안정적', '비선형/불안정')
    summary.index.name = 'Feature'
    summary = summary.sort_values('mean_abs_shap', ascending=False)

    # mean|SHAP| 를 합 1 로 정규화한 비율과 내림차순 누적 비율
    total_abs = float(summary['mean_abs_shap'].sum())

    if total_abs > 0:
        summary['ratio'] = summary['mean_abs_shap'] / total_abs
    else:
        summary['ratio'] = 0.0

    summary['cum_ratio'] = summary['ratio'].cumsum()

    # 컬럼 순서 정리 — 영향력 → 비율 → 누적 비율 → 해석용 컬럼
    summary = summary[['mean_abs_shap', 'ratio', 'cum_ratio',
                       'mean_shap', 'std_shap', 'direction', 'cv', 'stability']]

    # 후속 Dependence·Waterfall 이 재계산 없이 쓰는 원시 데이터를 attrs 에 담는다.
    # explainer 객체 자체는 담지 않는다 — 복사가 안 되는 객체라 슬라이스할 때 깨진다.
    summary.attrs['shap_values'] = shap_2d
    summary.attrs['shap_df'] = shap_df
    summary.attrs['data'] = x_explain
    summary.attrs['expected_value'] = base_value
    summary.attrs['feature_names'] = feat_names
    summary.attrs['explainer_type'] = explainer_type
    summary.attrs['model_class'] = model_class
    summary.attrs['task'] = task
    summary.attrs['class_index'] = used_class
    summary.attrs['class_names'] = class_names
    summary.attrs['output_space'] = output_space

    if search_params is not None:
        summary.attrs['search_params'] = search_params

    # --- 6) 시각화 (bar·beeswarm·cumulative 는 각각 독립 토글) ---
    n_plots = int(bar) + int(beeswarm) + int(cumulative)

    if task == 'classification' and class_names:
        cls_tag = f' · class={class_names[used_class]}'
    else:
        cls_tag = ''

    # 단위는 모델마다 달라 그래프에도 함께 적어 준다 (확률 0.1 과 로그오즈 0.1 은 다르다)
    unit_tag = _shap_unit_tag(task, output_space)

    # 그래프가 한 장이면 경로를 그대로, 여러 장이면 종류 꼬리표를 붙여 저장한다
    if n_plots > 1:
        bar_path = _shap_tag_path(save_path, 'bar')
        beeswarm_path = _shap_tag_path(save_path, 'beeswarm')
        cumulative_path = _shap_tag_path(save_path, 'cumulative')
    else:
        bar_path = save_path
        beeswarm_path = save_path
        cumulative_path = save_path

    # ① Bar — 영향력 순위 막대 + 누적 비율
    if bar:
        _shap_bar_plot(summary, model_class, cls_tag, unit_tag, cum_ratio, max_display,
                       palette, title, xlabel, ylabel, width, height, bar_path)

    # ② Beeswarm — 변수별 SHAP 값의 방향과 분포를 점으로 표시
    if beeswarm:
        if title is None:
            bee_title = f'SHAP Summary (Beeswarm): {model_class}{cls_tag}'
        else:
            bee_title = title

        my_plot.init(title=bee_title, width=width, height=height)
        shap.summary_plot(shap_2d, x_explain, plot_type='dot',
                          max_display=max_display, show=False, plot_size=None)

        # 축 라벨은 shap 이 직접 붙이므로 그린 뒤에 덮어쓴다
        if xlabel is None:
            plt.xlabel(f'SHAP value{unit_tag}')
        else:
            plt.xlabel(xlabel)

        my_plot.show(save_path=beeswarm_path)

    # ③ Cumulative — 파레토 차트 (막대=비율, 선=누적 비율)
    if cumulative:
        _shap_pareto_plot(summary, model_class, cls_tag, unit_tag, max_display,
                          palette, title, xlabel, ylabel, width, height, cumulative_path)

    # --- 7) 분석 요약 출력 ---
    if verbose:
        print('\n' + '=' * 78)
        print(f'◆ SHAP Analysis: {model_class}  ({explainer_type}, {task})')
        print('=' * 78)

        if explainer_type == 'KernelExplainer':
            print(f'   ⚠ {model_class} 는 전용 explainer 를 쓸 수 없어 KernelExplainer 를 사용합니다.')

            if fallback_reason:
                print(f'     사유: {fallback_reason}')

            print(f'     계산이 느릴 수 있어 max_samples({max_samples})·'
                  f'background_samples({background_samples}) 로 표본을 제한했습니다.')

        if task == 'classification' and class_names:
            print(f'   ▷ 설명 대상 클래스: {class_names[used_class]} (index={used_class})')

            if single_output:
                print(f'     이 모델은 출력이 하나라 SHAP 값이 양성 클래스 기준으로 나옵니다'
                      f"{' — 부호를 뒤집어 반대 클래스로 변환했습니다.' if used_class != n_classes - 1 else '.'}")

        if search_params is not None:
            print(f'   ▷ 탐색 객체의 best_params: {search_params}')

        print(f'   설명 표본 {len(x_explain)}행 × 변수 {len(feat_names)}개 · '
              f'base value={base_value:.4f} · SHAP 단위={output_space}')

        # 단위를 알면 base value 를 사람이 읽는 값으로 한 번 더 풀어 준다
        if output_space == '로그오즈' and n_classes == 2:
            print(f'   → base value 를 확률로 바꾸면 '
                  f'{1.0 / (1.0 + np.exp(-np.clip(base_value, -500, 500))):.2%} '
                  f'(SHAP 값도 확률이 아니라 로그오즈 증감입니다)')
        elif output_space == '알 수 없음':
            print(f'   ⚠ SHAP 값의 단위를 모델 출력과 대조해 확정하지 못했습니다 — '
                  f'절대값 해석은 피하고 순위·부호 위주로 읽으세요.')

        print('=' * 78)

        for feature, row in summary.head(5).iterrows():
            print(f"   {feature:<22} mean|SHAP|={row['mean_abs_shap']:.4f}  "
                  f"방향={row['direction']}  {row['stability']}")

        print('=' * 78 + '\n')

    return summary


# --------------------------------------------------------
# mean|SHAP| 기준으로 주변수를 자동 선정
# --------------------------------------------------------
def _shap_main_features(shap_values, cum_ratio, top_k):
    """Dependence Plot 의 주축이 될 변수를 mean|SHAP| 기준으로 고른다.

    feature_importance 의 채택 규칙과 같다 — 합 1 로 정규화 후 내림차순 누적 비율이
    cum_ratio 에 처음 도달하는 지점까지 채택한다. top_k 를 주면 상위 top_k 개를 쓴다.

    Args:
        shap_values (ndarray): (n, f) SHAP 값 배열
        cum_ratio (float): 채택할 누적 비율
        top_k (int): 지정하면 누적 비율 대신 상위 top_k 개 채택

    Returns:
        tuple: (선정된 변수 인덱스 ndarray, 선정 기준 설명 str)
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    order_all = np.argsort(mean_abs)[::-1]
    n = len(order_all)

    if top_k is not None:
        k = max(1, min(int(top_k), n))
        return order_all[:k], f'top_k={top_k}'

    total = mean_abs.sum()

    # 모든 변수의 기여가 0 이면 비율을 계산할 수 없어 최상위 1개만 쓴다
    if total <= 0:
        return order_all[:1], 'cum_ratio(폴백: 기여 0)'

    cum = np.cumsum(mean_abs[order_all] / total)
    k = int(np.searchsorted(cum, cum_ratio)) + 1
    k = max(1, min(k, n))

    return order_all[:k], f'cum_ratio={cum_ratio:.0%}'


# --------------------------------------------------------
# 주변수마다 상호작용이 강한 짝을 자동 계산
# --------------------------------------------------------
def _shap_auto_pairs(shap_values, data, cum_ratio, top_k):
    """자동 선정한 주변수 각각에 대해 상호작용이 가장 강한 변수를 짝지어 반환한다.

    Args:
        shap_values (ndarray): (n, f) SHAP 값 배열
        data (DataFrame): 모델공간 입력
        cum_ratio (float): 주변수 선정에 쓸 누적 비율
        top_k (int): 지정하면 누적 비율 대신 상위 top_k 개 채택

    Returns:
        tuple: (변수쌍 list[dict], 선정 기준 설명 str)
    """
    import shap

    cols = list(data.columns)
    order, select_mode = _shap_main_features(shap_values, cum_ratio, top_k)

    pairs = []
    for fi in order:
        # 해당 변수와 상호작용이 큰 변수를 중요도 순으로 돌려준다
        inter = shap.utils.approximate_interactions(int(fi), shap_values, data)

        if len(inter):
            partner = cols[int(inter[0])]
        else:
            partner = cols[int(fi)]

        pairs.append({'feature': cols[int(fi)], 'interaction_feature': partner})

    return pairs, select_mode


# --------------------------------------------------------
# 변수값과 SHAP 값의 관계를 그리는 Dependence Plot
# --------------------------------------------------------
def shap_dependence_plot(result, features=None, cum_ratio=0.9, top_k=None,
                         interaction_index='auto', plot=True,
                         title=None, xlabel=None, ylabel=None,
                         width=1280, height=640, save_path=None, verbose=True):
    """shap_analysis 결과로 Dependence Plot 을 그린다 (의미 있는 변수쌍 자동 계산).

    한 변수의 값(x축) 대비 그 변수의 SHAP 값(y축)을, 상호작용이 강한 다른 변수의
    색으로 표시해 비선형·상호작용 패턴을 드러낸다.

    features 를 주지 않으면 주변수를 자동 선정한다 — mean|SHAP| 누적 비율이 cum_ratio 에
    도달하는 상위 변수까지, 즉 '영향력의 90% 를 설명하는 핵심 변수' 만 그린다.

    Args:
        result (DataFrame): shap_analysis 가 반환한 결과표 (attrs 를 사용).
        features (list): 주변수 이름 리스트 (기본값: None → 자동 선정).
        cum_ratio (float): 자동 선정 시 채택할 누적 비율 (기본값: 0.9).
        top_k (int): 지정하면 누적 비율 대신 상위 top_k 개를 주변수로 채택 (기본값: None).
        interaction_index (str): 색으로 쓸 상호작용 변수 (기본값: 'auto' → 변수별 자동).
        plot (bool): 결과를 시각화할지 여부 (기본값: True).
        title (str): 그래프 제목 (기본값: None).
        xlabel (str): x축 라벨 (기본값: None).
        ylabel (str): y축 라벨 (기본값: None).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 640).
        save_path (str): 그래프 저장 경로 (기본값: None). 여러 장이면 변수명이 붙는다.
        verbose (bool): 선정된 변수쌍을 출력할지 여부 (기본값: True).

    Returns:
        list: 사용된 변수쌍 [{'feature', 'interaction_feature'}, ...].

    Raises:
        ValueError: cum_ratio 가 (0, 1] 밖이거나 없는 변수명을 준 경우.
    """
    import shap

    if not 0.0 < cum_ratio <= 1.0:
        raise ValueError(f"cum_ratio 는 (0.0, 1.0] 범위여야 합니다: {cum_ratio}")

    shap_values = result.attrs['shap_values']
    data = result.attrs['data']

    # y축(SHAP 값)의 단위는 모델마다 달라 제목에 함께 적는다
    unit_tag = _shap_unit_tag(result.attrs.get('task'),
                              result.attrs.get('output_space', '알 수 없음'))

    # --- 1) 변수쌍 결정 — 자동 선정 또는 사용자가 준 주변수 ---
    if features is None:
        pairs, select_mode = _shap_auto_pairs(shap_values, data, cum_ratio, top_k)
    else:
        select_mode = f'features={features}'
        pairs = []

        for f in features:
            if f not in data.columns:
                raise ValueError(f"'{f}' 는 모델공간 변수에 없습니다. 가능: {list(data.columns)}")

            if interaction_index == 'auto':
                inter = shap.utils.approximate_interactions(
                    data.columns.get_loc(f), shap_values, data)

                if len(inter):
                    partner = data.columns[int(inter[0])]
                else:
                    partner = f
            else:
                partner = interaction_index

            pairs.append({'feature': f, 'interaction_feature': partner})

    if verbose:
        print(f'◆ SHAP Dependence — 변수쌍(주변수 × 상호작용변수) · 선정기준={select_mode}')
        for p in pairs:
            print(f"   {p['feature']}  ×  {p['interaction_feature']}")

    # --- 2) 시각화 — 변수쌍마다 그래프를 한 장씩 그린다 ---
    if plot and pairs:
        multi = len(pairs) > 1

        for p in pairs:
            if interaction_index == 'auto':
                inter = p['interaction_feature']
            else:
                inter = interaction_index

            if title is None:
                plot_title = (f"SHAP Dependence: {p['feature']}  ×  "
                              f"{p['interaction_feature']}{unit_tag}")
            else:
                plot_title = title

            fig, ax = my_plot.init(title=plot_title, width=width, height=height)
            shap.dependence_plot(p['feature'], shap_values, data,
                                 interaction_index=inter, ax=ax, show=False)

            # 축 라벨은 shap 이 직접 붙이므로 사용자가 준 값만 덮어쓴다
            if xlabel is not None:
                ax.set_xlabel(xlabel)

            if ylabel is not None:
                ax.set_ylabel(ylabel)

            # 여러 장이면 주변수명을 꼬리표로 붙여 서로 덮어쓰지 않게 한다
            if multi:
                path = _shap_tag_path(save_path, p['feature'])
            else:
                path = save_path

            my_plot.show(save_path=path)

    return pairs


# --------------------------------------------------------
# 예측값 분위수에서 대표 관측치를 선정
# --------------------------------------------------------
def _shap_representative_indices(shap_values, base_value, n):
    """예측값(= base value + SHAP 합) 분포의 균등 분위수에서 대표 관측치를 고른다.

    예측값을 오름차순으로 세운 뒤 0~100% 를 n 등분한 위치의 행을 뽑는다
    (n=5 → 최저·하위·중앙·상위·최고). 결정적이라 재현 가능하다.

    Args:
        shap_values (ndarray): (n, f) SHAP 값 배열
        base_value (float): base value (평균 예측)
        n (int): 선정할 관측치 수

    Returns:
        list: [{'index'(행 위치), 'pred'(예측값), 'quantile'(0~1)}, ...] 예측 오름차순.
    """
    # SHAP 의 가산성 — 행별 예측 = base value + 그 행의 SHAP 합
    pred = np.asarray(base_value, dtype=float) + np.asarray(shap_values, dtype=float).sum(axis=1)
    order = np.argsort(pred)
    total = len(order)
    n = max(1, min(int(n), total))

    # 0 ~ 마지막 위치를 n 등분한 분위 위치 (반올림 후 중복 제거)
    positions = np.unique(np.round(np.linspace(0, total - 1, n)).astype(int))

    picks = []
    for p in positions:
        if total > 1:
            quantile = float(p / (total - 1))
        else:
            quantile = 0.0

        picks.append({'index': int(order[p]),
                      'pred': float(pred[order[p]]),
                      'quantile': quantile})

    return picks


# --------------------------------------------------------
# 개별 관측치의 기여를 분해하는 Waterfall Plot
# --------------------------------------------------------
def shap_waterfall_plot(result, index=None, n=5, max_display=12, plot=True,
                        title=None, width=1280, height=640, save_path=None, verbose=True):
    """shap_analysis 결과로 관측치별 Waterfall Plot 을 그린다 (개별 사례 기여 분해).

    base value(평균 예측)에서 시작해 변수별 SHAP 기여를 위·아래로 쌓아 그 관측치의
    최종 예측에 도달하는 과정을 보여준다 (기여가 큰 변수부터).

    index 를 주지 않으면 예측값 분포의 균등 분위수에서 대표 n개(기본 5: 최저·하위·
    중앙·상위·최고)를 자동으로 뽑아 각각 그린다.

    Args:
        result (DataFrame): shap_analysis 가 반환한 결과표 (attrs 를 사용).
        index (int or list): 설명할 관측치의 행 위치 (기본값: None → 분위수 자동 선정).
        n (int): 자동 선정할 대표 관측치 수 (기본값: 5). index 를 주면 무시된다.
        max_display (int): 표시할 최대 변수 개수 (기본값: 12). 나머지는 others 로 합산.
        plot (bool): 결과를 시각화할지 여부 (기본값: True).
        title (str): 그래프 제목 (기본값: None).
        width (int): 그래프 너비 (기본값: 1280).
        height (int): 그래프 높이 (기본값: 640).
        save_path (str): 그래프 저장 경로 (기본값: None). 여러 장이면 행 위치가 붙는다.
        verbose (bool): 선정된 관측치를 출력할지 여부 (기본값: True).

    Returns:
        list: 설명에 사용된 관측치 행(Series) 리스트.
    """
    import shap

    shap_values = result.attrs['shap_values']
    data = result.attrs['data']
    base_value = result.attrs['expected_value']
    feat_names = result.attrs['feature_names']

    # --- 1) 설명할 관측치 결정 — 자동(분위수) / 단일 정수 / 정수 리스트 ---
    if index is None:
        picks = _shap_representative_indices(shap_values, base_value, n)
        mode = f'auto: 예측값 분위수 {len(picks)}개'
    elif isinstance(index, (list, tuple, np.ndarray)):
        picks = []
        for i in index:
            picks.append({'index': int(i), 'pred': None, 'quantile': None})
        mode = f'지정 {len(picks)}개'
    else:
        picks = [{'index': int(index), 'pred': None, 'quantile': None}]
        mode = '지정 1개'

    class_names = result.attrs.get('class_names')
    class_index = result.attrs.get('class_index')
    output_space = result.attrs.get('output_space', '알 수 없음')

    if class_names and class_index is not None:
        cls_tag = f' · class={class_names[class_index]}'
    else:
        cls_tag = ''

    # base value 와 pred 는 모델에 따라 확률일 수도 로그오즈일 수도 있어 단위를 함께 적는다
    unit_tag = _shap_unit_tag(result.attrs.get('task'), output_space)

    if verbose:
        print(f'◆ SHAP Waterfall — 선정={mode}{cls_tag}, '
              f'base value={base_value:.4f} (단위: {output_space})')

    # --- 2) 관측치마다 Waterfall 한 장씩 ---
    multi = len(picks) > 1
    rows = []

    for p in picks:
        i = p['index']
        row = data.iloc[i]

        # 자동 선정일 때만 예측값·분위 꼬리표가 붙는다
        if p['pred'] is not None:
            info = f" — pred≈{p['pred']:.3g}{unit_tag} (분위 {p['quantile']:.0%})"
        else:
            info = ''

        if verbose:
            print(f'   · 위치={i} (라벨={data.index[i]}){info}')

        rows.append(row)

        if not plot:
            continue

        # shap 이 요구하는 설명 객체 — 기여도·기준값·실제 변수값을 함께 담는다
        expl = shap.Explanation(
            values=np.asarray(shap_values[i], dtype=float),
            base_values=float(base_value),
            data=row.values,
            feature_names=list(feat_names),
        )

        if title is None:
            plot_title = f'SHAP Waterfall: obs#{i}{info}{cls_tag}'
        else:
            plot_title = title

        my_plot.init(width=width, height=height)
        shap.plots.waterfall(expl, max_display=max_display, show=False)

        # shap 이 그리면서 캔버스 크기를 바꾸므로 제목은 그린 뒤에 붙인다
        plt.title(plot_title, fontsize=18, fontweight=500, pad=25)

        # 여러 장이면 관측치 위치를 꼬리표로 붙여 서로 덮어쓰지 않게 한다
        if multi:
            path = _shap_tag_path(save_path, f'obs{i}')
        else:
            path = save_path

        my_plot.show(save_path=path)

    return rows
