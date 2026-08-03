# 나의 분석 모듈 레퍼런스 가이드

`import my_모듈` 형태로 불러와 쓰는 개인 패키지 6개 파일에 대한 정리본.
분석할 때마다 "이 함수 뭐였지?" 싶을 때 펼쳐보는 용도.

---

## 0. 전체 구조 (`__init__.py`)

패키지를 임포트하면 아래 6개 서브모듈이 한번에 로드되고, 한글 폰트(`../helpers/fonts`)와
`matplotlib` 기본 스타일(해상도 200dpi, 마이너스 기호 깨짐 방지 등)이 자동으로 세팅됨.

| 파일 | 역할 |
|---|---|
| `my_qtcheck` | 데이터 품질 점검 (타입, 결측치, 중복, 기술통계) |
| `my_prep` | 데이터 전처리 (라벨링, 더미, 로그변환, 이상치, 스케일링, VIF) |
| `my_stats` | 통계적 가설검정 (t검정, 분산분석, 상관, 카이제곱) |
| `my_ols` | 선형회귀 (적합·진단·보고) |
| `my_logit` | 로지스틱회귀 (적합·진단·보고) |
| `my_plot` | 시각화 (seaborn/matplotlib 래퍼) |

**의존 관계**: `my_stats`, `my_ols`, `my_logit`는 내부적으로 `my_plot`을 호출해서 결과를 그림.
`my_stats`는 `my_prep`도 참조함. 즉 `my_plot` → `my_stats`/`my_ols`/`my_logit` 순서로 쌓인 구조.

---

## 1. `my_qtcheck.py` — 데이터 품질 점검

분석 맨 처음, EDA 단계에서 데이터 상태를 훑어보는 함수들.

| 함수 | 역할 | 주요 파라미터 | 리턴 |
|---|---|---|---|
| `set_type(data, as_int=[], as_float=[], as_string=[], as_category=[], as_datetime=[])` | 지정한 컬럼들을 원하는 타입으로 한번에 변환 + `df.info()` 출력 | 타입별 컬럼명 리스트 5종 | 타입 변경된 DataFrame |
| `get_number_column_names(data)` | 숫자형 컬럼 이름만 리스트로 추출 | - | 컬럼명 리스트 |
| `get_categorical_column_names(data)` | `category` 타입 컬럼 이름만 리스트로 추출 | - | 컬럼명 리스트 |
| `check_duplicates(data, drop=True)` | 중복 행 개수 출력, `drop=True`면 제거까지 | `drop` | 중복 제거된 DataFrame |
| `check_missing_values(data)` | 컬럼별 결측치 개수·비율표 생성 | - | 결측 개수/비율 DataFrame |
| `categorical_summary(data, columns=None, value_counts=True, save_path=None)` | 범주형 컬럼 기술통계 + (옵션) `value_counts()` 출력, Excel 저장 | `columns`(None이면 전체 범주형), `save_path` | 기술통계 DataFrame |
| `numerical_summary(data, columns=None, save_path=None)` | **숫자형 컬럼 종합 진단표** — 아래 참고 | `columns`, `save_path` | 진단 DataFrame |

**`numerical_summary`가 하는 일 (제일 자주 쓸 함수)**
`describe()` 결과에 컬럼을 계속 덧붙여서 한 번에:
- `rel_diff`, `rdiff_flag` : 평균-중앙값 상대 차이율로 분포 치우침 여부(`similar`/`diff`/`large_diff`) 판단
- `iqr`, `upper_bound`, `lower_bound` : IQR 기반 이상치 경계
- `upper/lower/outliers`, `*_ratio` : 이상치 개수와 비율
- `skew`, `skew_interpret` : 왜도와 해석(`left tail`/`right tail`/`symmetric`)
- `kurt`, `kurt_interpret` : 첨도와 해석(`platykurtic`/`leptokurtic`/`mesokurtic`)
- `log_need` : 왜도·첨도·최솟값을 종합해 `log`/`log1p`/`reverse_log1p`/`none` 중 추천 변환 방식까지 제시

→ EDA에서 이거 하나 돌리면 "이 컬럼 로그변환 필요한가, 이상치 몇 개인가"가 표 하나로 정리됨.

---

## 2. `my_prep.py` — 데이터 전처리

`numerical_summary`나 EDA로 파악한 문제를 실제로 고치는 단계.

| 함수 | 역할 | 주요 파라미터 | 리턴 |
|---|---|---|---|
| `labeling(df, columns, save_path=None, verbose=True)` | 범주형 컬럼을 0부터 시작하는 정수로 라벨 인코딩 | `save_path`(인코더 pkl 저장) | 라벨링된 DataFrame |
| `dummies(df, columns, drop_first=True, verbose=True)` | 원-핫 인코딩(더미변수 생성). `drop_first=True`면 다중공선성(더미변수 함정) 방지용으로 첫 범주 제거 | `drop_first` | 더미 인코딩된 DataFrame |
| `long2wide(df, hue, values, dropna=True)` | long format → wide format 변환 (`hue` 값들이 새 컬럼이 됨) | `hue`, `values` | wide DataFrame |
| `log_transform` / `log_transfrom`(오타로 중복 존재) `(df, log_columns=None, log1p_columns=None, reflect_columns=None, verbose=True)` | 로그 계열 변환 일괄 적용 — `log`(0 없는 우측꼬리) / `log1p`(0 포함 우측꼬리) / `reflect`(좌측꼬리는 반사 후 log1p) | 변환 방식별 컬럼 리스트 | 변환된 DataFrame |
| `inverse_log_transform(df, log_columns=None, log1p_columns=None, reflect_columns=None, verbose=True)` | 위 로그변환을 원래 단위로 역변환 | `reflect_columns`는 `{컬럼명: 변환당시 최댓값}` 딕셔너리로 전달 | 역변환된 DataFrame |
| `replace_outlier(df, columns=None, method='bound', value=None, verbose=True)` | 이상치를 지정 방식으로 대체 | `method`: `'bound'`(경계값)/`'median'`/`'mean'`/`'value'` | 이상치 대체된 DataFrame |
| `scaling(df, columns=None, method='standard', save_path=None, verbose=True)` | 스케일링 | `method`: `'standard'`/`'minmax'`/`'robust'`/`'maxabs'` | 스케일링된 DataFrame |
| `reduce_vif(df, columns=None, threshold=10.0, verbose=True)` | VIF가 threshold 이하가 될 때까지 VIF 최대 변수를 반복 제거 (다중공선성 해소) | `threshold` | 변수 일부 제거된 DataFrame |

> ⚠️ `log_transform`과 `log_transfrom`은 이름이 오타로 두 개 존재 — 같은 역할이니 헷갈리지 않게 하나로 쓰는 걸 추천.

---

## 3. `my_stats.py` — 통계적 가설검정

**공통 패턴**: 대부분의 검정 함수가 "①가정 확인(정규성/등분산성) → ②그에 맞는 검정 자동 분기 → ③(옵션)시각화"를 한 번에 처리함. 직접 `normaltest`, `levene` 등을 골라 쓸 필요 없이 함수 하나로 끝냄.

| 함수 | 역할 | 자동 분기 로직 |
|---|---|---|
| `ci(data, column=None, clevel=0.95)` | 모평균 신뢰구간 계산 | - |
| `test_assumptions(data, columns=None, alpha=0.05, center="median")` | 정규성(normaltest) + (컬럼 2개 이상이면) 등분산성 검정 | 모두 정규 → Bartlett, 하나라도 비정규 → Levene |
| `test_1sample(data, column, popmean=0, alpha=0.05)` | 일표본 평균 검정 (양측/좌측/우측 3방향 한번에) | 정규 → 일표본 t검정, 비정규 → Wilcoxon 부호순위 |
| `test_paired(data, before, after, alpha=0.05, plot=True, ...)` | 대응표본(전/후) 차이 검정 | 차이값 정규 → 대응표본 t검정, 비정규 → Wilcoxon |
| `test_independent(data, group1, group2, alpha=0.05, plot=True, ...)` | 두 독립집단 평균 비교 | 둘 다 정규 → (등분산이면 Student, 아니면 Welch) t검정, 하나라도 비정규 → Mann-Whitney U |
| `anova_oneway(data, y, between, alpha=0.05)` | 일원분산분석 | 등분산 위반 시 Welch-ANOVA |
| `posthoc_oneway(data, y, between, alpha=0.05, plot=True, summary=True, ...)` | 일원분산분석 사후검정 | - |
| `anova_twoway(data, y, between, alpha=0.05, order=None, plot=True, ...)` | 이원분산분석 (상호작용 플롯 포함) | 등분산 위반 시 Type-II ANOVA |
| `posthoc_twoway(data, y, between, alpha=0.05, plot=True, summary=True, ...)` | 이원분산분석 사후검정 | 등분산 충족 → Tukey HSD, 위반 → Games-Howell |
| `correlation(data, x, y, alpha=0.05, plot=True, ...)` | 두 연속형 변수 상관분석 (산점도+회귀선 포함) | - |
| `multi_correlation(data, columns=None, alpha=0.05, plot=True, diag_kind="kde", reg=False, ...)` | 여러 변수 쌍 상관분석 일괄 + 산점도 행렬 | - |
| `compute_vif(df, columns=None)` | 각 변수 VIF 계산 | - |
| `correlation_summary(corr_df, strength="Strong")` | `multi_correlation` 결과를 변수별로 집계해 다중공선성 의심 변수를 정렬해서 보여줌 | - |
| `chi2_goodness_of_fit(data, column, expected=None, order=None, alpha=0.05, plot=True, ...)` | 적합도 검정 (관측 vs 기대빈도) + Cohen's w 효과크기 | 기대빈도 가정 미충족 시 `recommend`에 "category merge" 안내 |
| `chi2_independence(data, x, y, alpha=0.05, plot=True, orient='v', ...)` | 두 범주형 변수 독립성 검정 + Cramer's V | 기대빈도 가정 위반 + 2×2표 → Fisher's exact test |
| `chi2_homogeneity(data, group, category, alpha=0.05, plot=True, ...)` | 동질성 검정 (`chi2_independence`와 로직 동일, 해석만 다름) | 위와 동일 |

> `_chi2_crosstab`은 위 두 카이제곱 함수의 내부 공통 구현부(private)라 직접 호출할 일은 없음.

---

## 4. `my_ols.py` — 선형회귀

| 함수 | 역할 |
|---|---|
| `fit_model(data, y, summary=False)` | OLS로 선형회귀 적합. `y` 제외 나머지 컬럼 전부 독립변수로 사용, 절편 자동 추가 |
| `predict(fit, new_data)` | 새 데이터로 예측값(`pred` 컬럼) 산출 |
| `test_linear(fit, alpha=0.05, plot=True, ...)` | 선형성 검정 (Ramsey RESET test) — 적합값-잔차 산점도 |
| `test_normal(fit, alpha=0.05, plot=True, ...)` | 잔차 정규성 검정 2종 + Q-Q플롯 |
| `test_equalvar(fit, alpha=0.05)` | 잔차 등분산성 검정 |
| `test_independent(fit)` | 잔차 독립성 검정 (Durbin-Watson, 시계열 아니면 무시해도 무방) |
| `report_fitness(fit, log_y=False, log_x=None, ...)` | 모형 적합도를 문장으로 서술 (로그변환 여부 반영해서 실제 적합 모형 그대로 표기) |
| `report_variables(fit, data, hc3=False)` | 독립변수별 회귀계수 보고표 (B, 표준오차, 베타, t, p, 공차, VIF) |
| `report_variables_text(fit, log_y=False, ..., hc3=False)` | 위 보고표를 해석 문장(markdown 불릿)으로 변환 |
| `plot_beta(fit, data, ...)` | 표준화 회귀계수(베타)를 가로 막대그래프로 시각화 |
| `auto_ols(data, y, report=True, log_y=False, ..., backward=False, alpha=0.05)` | **적합 → 보고서 → 가정검정까지 원샷 실행** |
| `fit_pipeline(data, y, nominal_cols=None, *, labeling=?, encode=?, log=?, outlier=?, vif=?, ...)` | 전처리 플래그(라벨링/더미/로그/이상치/VIF) 지정해서 전처리부터 적합까지 한번에 |
| `compare_models(fits, metric='RMSE', sub_metric='변수수', tolerance=0.05, ...)` | 여러 모델을 성능지표로 비교·정렬, 최고 모델 반환 (log1p/반사변환 모델은 원본 척도로 환산해서 비교) |
| `report_model(fit, title=True, plot=True)` | `fit_pipeline`/`auto_ols` 결과 객체 하나로 성능보고+가정검정 전체를 한번에 출력 |

**추천 워크플로**: 처음엔 `auto_ols` 또는 `fit_pipeline`으로 빠르게 돌려보고, 여러 후보 모델은 `compare_models`로 비교 → 최종 모델은 `report_model`로 깔끔하게 정리.

---

## 5. `my_logit.py` — 로지스틱회귀

`my_ols.py`와 함수 구성이 대칭적임 (같은 이름의 함수는 로짓 버전으로 이해하면 됨).

| 함수 | 역할 |
|---|---|
| `fit_model(data, y, summary=False)` | Logit으로 이항 로지스틱 회귀 적합 (`y`는 0/1이어야 함) |
| `predict(fit, new_data, threshold=0.5)` | 예측 확률(`proba`)과 임계값 기준 분류(`pred`) 반환 |
| `plot_sigmoid(fit, data, x, threshold=0.5, ...)` | 특정 독립변수 `x`에 대한 사건발생확률 S자 곡선 시각화 |
| `report_fitness(fit)` | 모형 적합도 보고 문장 생성 |
| `report_variables(fit, data)` | 독립변수별 계수·오즈비(OR)·95% 신뢰구간·VIF 보고표 (｜B｜ 내림차순 정렬) |
| `report_variables_text(fit, data=None, alpha=0.05)` | 오즈비를 "오즈가 몇% 증가/감소" 식 해석 문장으로 변환 |
| `plot_odds(fit, data, ...)` | 오즈비를 가로 막대그래프로 시각화 (OR=1 기준선, 증가/감소 색 구분) |
| `auto_logit(data, y, report=True, plot=True, threshold=0.5, backward=False, alpha=0.05, ...)` | **적합 → (옵션)후진소거법 변수선택 → 보고 → 시각화까지 원샷 실행** |

---

## 6. `my_plot.py` — 시각화 (seaborn/matplotlib 래퍼)

`my_stats`/`my_ols`/`my_logit`이 내부에서 호출하지만, 단독으로 EDA 그래프 그릴 때도 바로 쓸 수 있음.
대부분 공통 파라미터: `data, x, y, hue, palette, title, xlabel, ylabel, width, height, save_path`.

| 함수 | 역할 |
|---|---|
| `init(width=1280, height=640, rows=1, cols=1, ...)` | Figure/Axes 생성 (서브플롯 구성 포함) |
| `show(save_path=None)` | 그래프 화면 출력 (+ 저장) |
| `lineplot(...)` | 선 그래프 |
| `kdeplot(..., meanline=?, clevel=?)` | 커널밀도추정 그래프 (평균선, 신뢰구간 옵션) |
| `histplot(..., bins="auto", kde=?)` | 히스토그램 (+ KDE 겹쳐 그리기) |
| `boxplot(..., orient=?)` | 상자그림 |
| `violinPlot(..., orient=?)` | 바이올린 플롯 |
| `heatmap(..., annot=True, fmt="0.2f")` | 히트맵 (상관행렬 등) |
| `barplot(..., estimator=np.mean)` | 막대그래프 (대표값 집계) |
| `countplot(...)` | 범주별 빈도 막대그래프 |
| `pieplot(x, labels, ..., donutchart=?)` | 파이/도넛 차트 |
| `stackplot(..., aggfunc=np.sum, ratio=?)` | 누적 막대그래프 (비율 표시 옵션) — `chi2_independence` 등에서 사용 |
| `scatterplot(..., size=100, alpha=1)` | 산점도 |
| `plot_hull(data, x, y, hue, palette, ax)` | 산점도 위에 군집별 ConvexHull 외곽선 (보조함수) |
| `lmplot(...)` | 산점도+회귀선 (facet 지원: `col`/`row`) |
| `pairplot(..., diag_kind="kde", reg=?)` | 산점도 행렬 — `multi_correlation`에서 사용 |
| `pointplot(..., errorbar=?, dodge=?)` | 점그래프 (분산분석 상호작용 플롯 등에 활용) |

> `_draw_ci`는 `kdeplot` 내부에서 신뢰구간을 그리는 보조함수(private)라 직접 호출하지 않음.

---

## 전형적인 분석 흐름 (참고용 지도)

```
1. my_qtcheck.set_type / check_missing_values / check_duplicates   → 데이터 상태 점검
2. my_qtcheck.numerical_summary / categorical_summary               → 이상치·왜도·로그필요여부 파악
3. my_prep.replace_outlier / log_transform / dummies / scaling      → 문제 있는 부분 전처리
4. my_stats.test_* / anova_* / correlation / chi2_*                 → 가설검정
5. my_ols.auto_ols / my_logit.auto_logit  (또는 fit_pipeline)       → 회귀모델링
6. my_ols.compare_models / report_model                             → 모델 비교·최종 보고
```
