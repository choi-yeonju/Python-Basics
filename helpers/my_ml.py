# --------------------------------------------------------
# save_model / load_model : 학습된 모델의 직렬화와 경로 처리
# --------------------------------------------------------
import joblib
from pathlib import Path

# --------------------------------------------------------
# fit_pipeline : 전처리 단계를 쌓아 모델까지 연결
# --------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer                # 결측치 대체
from sklearn.preprocessing import OneHotEncoder         # 더미변수 인코딩
from sklearn.compose import ColumnTransformer           # 연속형·명목형 분기 처리
from sklearn.decomposition import PCA                   # 차원 축소

from . import RANDOM_STATE                              # 재현성을 위한 랜덤시드
from . import my_prep                                   # 스케일러 목록(SCALERS)
from .my_vif_selector import VIFSelector                # 다중공선성 제거
from .my_outlier_clipper import OutlierClipper          # 이상치 경계값 대체

# --------------------------------------------------------
# reg_score : 회귀 성능 지표 계산
# --------------------------------------------------------
import numpy as np
from pandas import DataFrame
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_squared_log_error, mean_absolute_percentage_error
)

# --------------------------------------------------------
# reg_compare_models : 모델별 지표 1행을 세로로 결합
# --------------------------------------------------------
from pandas import concat

# --------------------------------------------------------
# reg_overfit : 교차검증 점수와 학습곡선
# --------------------------------------------------------
from sklearn.model_selection import cross_validate, learning_curve as sk_learning_curve

from . import my_plot                                   # 학습곡선 그래프 틀(init·show)


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

    권장 조합: 선형·SGD 계열 → scale·vif·drop_first / 거리·SVM 계열 → 추가로 pca /
    트리·부스팅 계열 → 없음 / CatBoost → encode=False.

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
    model_name = name if name else type(model).__name__

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
    # 목록에 담는 순서가 곧 처리 순서다. 결측치를 먼저 메워야 이후 단계가 NaN 을 만나지 않고,
    # 이상치를 먼저 잘라야 정규화의 기준값(평균·표준편차)이 극단값에 끌려가지 않는다.
    # 다중공선성 제거를 정규화보다 앞에 두어 my_ols.fit_pipeline() 과 순서를 맞춘다
    # (VIF 는 컬럼별 아핀 변환에 불변이라 결과는 같고, 버려질 변수를 스케일링하지 않게 된다).
    # 차원 축소는 스케일에 민감하므로 반드시 정규화 뒤에 온다
    numeric_steps = []
    numeric_report = []

    if continuous and impute:
        numeric_steps.append(('imputer', SimpleImputer(strategy=numeric_impute)))
        numeric_report.append(f'결측치 대체({numeric_impute})')

    if continuous and outlier:
        numeric_steps.append(('outlier_clipper', OutlierClipper(method=outlier_method)))
        numeric_report.append(f'이상치 대체({outlier_method})')

    if continuous and vif:
        numeric_steps.append(('vif_selector', VIFSelector(threshold=vif_threshold)))
        numeric_report.append(f'다중공선성 제거(VIF >= {vif_threshold})')

    if continuous and scale:
        # 스케일러 이름은 my_prep.scaling() 과 같은 표기를 받는다 ('StandardScaler' -> 'standard')
        scale_name = scale_method.lower().replace('scaler', '').strip()

        # 오타를 냈을 때 KeyError 대신 사용 가능한 이름을 알려준다
        if scale_name not in my_prep.SCALERS:
            raise ValueError(f"지원하지 않는 스케일러입니다: '{scale_method}' "
                             f"(사용 가능: {list(my_prep.SCALERS.keys())})")

        numeric_steps.append(('scaler', my_prep.SCALERS[scale_name]()))
        numeric_report.append(f'정규화({scale_name})')

    if continuous and pca:
        numeric_steps.append(('pca', PCA(n_components=pca_variance, random_state=RANDOM_STATE)))
        numeric_report.append(f'차원 축소(설명분산 {pca_variance})')

    # --- 5) 명목형 전처리 단계 구성 ---
    categorical_steps = []
    categorical_report = []

    if nominal_cols and impute:
        categorical_steps.append(('imputer', SimpleImputer(strategy=categorical_impute)))
        categorical_report.append(f'결측치 대체({categorical_impute})')

    if nominal_cols and encode:
        # handle_unknown='ignore' 는 훈련 때 없던 범주가 검증 데이터에 나타나도
        # 예외 대신 전부 0인 더미로 처리해, 예측이 중간에 끊기지 않게 한다
        categorical_steps.append(('onehot', OneHotEncoder(
            drop='first' if drop_first else None,
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

    # y_test 를 1차원 배열로 통일
    if isinstance(y_test, DataFrame):
        y_test_array = y_test.values.ravel()
    else:
        y_test_array = np.asarray(y_test).ravel()

    # --- 기본 지표 ---
    r2 = r2_score(y_test_array, y_pred)
    mae = mean_absolute_error(y_test_array, y_pred)
    mse = mean_squared_error(y_test_array, y_pred)
    rmse = np.sqrt(mse)

    # --- RMSLE: 로그를 취하므로 음수가 하나라도 있으면 계산 불가 ---
    if np.any(y_test_array < 0) or np.any(y_pred < 0):
        rmsle = np.nan
    else:
        rmsle = np.sqrt(mean_squared_log_error(y_test_array, y_pred))

    # --- MAPE·MPE: 실제값으로 나누므로 0 이 하나라도 있으면 계산 불가 ---
    if np.any(y_test_array == 0):
        mape = np.nan
        mpe = np.nan
    else:
        # 사이킷런의 MAPE 는 비율을 반환하므로 백분율로 변환 (MPE 와 단위 일치)
        mape = mean_absolute_percentage_error(y_test_array, y_pred) * 100
        mpe = np.mean((y_test_array - y_pred) / y_test_array) * 100

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
        'MPE':   {'better': 'closer_to_zero', 'flaw_type': 'abs_excess', 'threshold': 5.0},
    }

    # --- 2) 파라미터 검증 ---
    if isinstance(aux, str):
        aux = [aux]     # 보조 지표를 문자열 하나로 준 경우도 허용

    if primary not in metric_specs:
        raise ValueError(f"지원하지 않는 주 지표입니다: '{primary}' "
                         f"(사용 가능: {sorted(metric_specs)})")

    for m in aux:
        if m not in metric_specs:
            raise ValueError(f"지원하지 않는 보조 지표입니다: '{m}' "
                             f"(사용 가능: {sorted(metric_specs)})")

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
        close_mask = primary_col >= best_primary * 0.95
        band_str = f'{primary} ≥ {best_primary * 0.95:.3f}'
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
    # neg_* scorer 는 '클수록 좋게' 부호가 뒤집혀 있어 되돌리고, RMSLE 는 제곱근,
    # MAPE 는 비율을 백분율로 바꿔 단위를 맞춘다.
    # MPE 는 0에 가까울수록 좋은 지표라 '훈련보다 얼마나 나쁜가'를 정의할 수 없어 제외한다.
    metric_specs = {
        'R2':    ('higher', 'r2',                                 lambda s: s),
        'MAE':   ('lower',  'neg_mean_absolute_error',            lambda s: -s),
        'MSE':   ('lower',  'neg_mean_squared_error',             lambda s: -s),
        'RMSE':  ('lower',  'neg_root_mean_squared_error',        lambda s: -s),
        'RMSLE': ('lower',  'neg_mean_squared_log_error',         lambda s: np.sqrt(-s)),
        'MAPE':  ('lower',  'neg_mean_absolute_percentage_error', lambda s: -s * 100),
    }

    # --- 2) 파라미터 검증 ---
    if isinstance(metrics, str):
        metrics = [metrics]     # 지표를 문자열 하나로 준 경우도 허용

    for m in metrics:
        if m not in metric_specs:
            raise ValueError(f"지원하지 않는 지표입니다: '{m}' "
                             f"(사용 가능: {sorted(metric_specs)})")

    # --- 3) 탐색 객체면 최적 모델을 꺼내기 ---
    # 탐색 객체 그대로 교차검증하면 폴드마다 탐색을 다시 도는 중첩 CV 가 되어 버린다.
    base_est = getattr(estimator, 'best_estimator_', estimator)
    search_params = getattr(estimator, 'best_params_', None)

    # --- 4) Train / Test 점수 ---
    train_scores = reg_score(base_est, x_train, y_train)
    test_scores = reg_score(base_est, x_test, y_test)
    classname = train_scores.index[0]

    # --- 5) CV 점수 (out-of-fold) — 판정의 일반화 기준 ---
    # 과소적합은 R2 로 보므로 metrics 에 없어도 R2 는 항상 함께 계산한다.
    y_cv = y_train.values.ravel() if isinstance(y_train, DataFrame) else np.asarray(y_train).ravel()
    wanted = list(dict.fromkeys(metrics + ['R2']))
    out = cross_validate(base_est, x_train, y_cv, cv=cv,
                         scoring={m: metric_specs[m][1] for m in wanted},
                         n_jobs=-1, params=fit_params, error_score=np.nan)
    cv_scores = {m: metric_specs[m][2](out[f'test_{m}']) for m in wanted}   # 부호·단위 보정

    # --- 6) 과소적합 선판정 (모델 수준) ---
    # 훈련 성능이 낮으면 격차 분석 자체가 무의미하므로 모든 지표 행에 우선 적용한다.
    # 스케일에 좌우되지 않는 R2 로만 본다 (RMSE 등은 절대 임계치를 정할 수 없다).
    train_r2 = train_scores['R2'].iloc[0]
    cv_r2 = float(np.mean(cv_scores['R2']))
    underfit = train_r2 < underfit_threshold

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
        print(f'     · train R2={train_r2:.4f} (과소적합 기준 < {underfit_threshold}) · '
              f'CV R2={cv_r2:.4f}')
        print('=' * 78 + '\n')

    return result
