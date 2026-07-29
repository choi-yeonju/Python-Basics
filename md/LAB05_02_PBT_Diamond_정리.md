# LAB-05. 데이터 품질 검사(Data Quality Check) 기본기 정리

> 이 문서는 `diamonds` 데이터셋으로 진행한 데이터 품질 검사 실습(`my_qtcheck.py`)을 바탕으로,
> **앞으로 어떤 데이터를 분석하든 항상 같은 순서로 밟아야 하는 "사전 점검 루틴"** 을 정리한 문서다.
> 새 프로젝트를 시작할 때마다 이 문서를 체크리스트처럼 펼쳐놓고 그대로 따라가면 된다.

---

## 0. 왜 "순서"가 중요한가?

품질 검사는 각 단계가 **다음 단계의 전제조건**이 되는 구조다. 순서를 바꾸면 뒤 단계 결과가 틀어진다.

```
① 자료형 점검/변환
   ↓ (범주형/숫자형이 제대로 구분되어야)
② 중복 데이터 점검
   ↓ (중복을 먼저 걷어내야 통계량이 왜곡되지 않음)
③ 결측치 점검
   ↓ (결측치 유무를 알아야 이후 처리 방침 결정 가능)
④ 품질 검사 결과 저장 (원본 스냅샷)
   ↓
⑤ 기술 통계량 확인 (범주형 → 숫자형)
```

예를 들어 `cut`, `color`, `clarity` 컬럼을 `category`로 바꾸기 전에 `numerical_summary()`를 돌리면,
문자열 컬럼이 숫자형으로 안 잡히거나 반대로 범주형 요약에서 빠지는 등 **함수가 컬럼을 엉뚱하게 분류**한다.
즉 ①번(자료형 변환)이 선행되지 않으면 ②~⑤ 전부 신뢰할 수 없다.

---

## 1. 전체 워크플로우 (diamonds 실습 기준)

| 단계 | 코드 | 하는 일 |
|---|---|---|
| 01 준비작업 | `load_data("diamonds")` | 원본 데이터 로드, `head()`로 눈으로 확인 |
| 02 자료형 점검 | `origin.info()` → `set_type(...)` | dtype 확인 후 필요한 컬럼을 int/float/str/category/datetime으로 변환 |
| 03 중복 점검 | `check_duplicates(df1)` | 행 단위 중복 개수 확인, 필요시 제거 |
| 04 결측치 점검 | `check_missing_values(df2)` | 컬럼별 결측 개수·비율 확인 |
| 05 결과 저장 | `df2.to_excel(...)` | 품질검사 끝난 "클린 데이터"를 스냅샷으로 저장 |
| 06 기술 통계량 | `categorical_summary(...)`, `numerical_summary(...)` | 범주형/숫자형 각각 요약 통계 + 저장 |

포인트: **02~05는 "정제(clean)" 단계, 06은 "이해(understand)" 단계**다.
정제가 끝나지 않은 데이터로 기술통계를 내면 의미가 없기 때문에 항상 이 순서를 지킨다.

---

## 2. 단계별 상세 개념

### ① 자료형 점검 — `set_type()`

**핵심 개념: pandas는 데이터를 불러올 때 dtype을 "추측"할 뿐, 분석 의도를 모른다.**
문자열처럼 보여도 값의 종류가 제한적이면(`cut`: Fair/Good/... 5개뿐) `category`로 바꿔줘야
메모리도 아끼고, 이후 `describe(include="category")` 같은 범주형 전용 통계도 쓸 수 있다.

```python
df1 = my_qtcheck.set_type(origin, as_category=['cut', 'color', 'clarity'])
```

- 내부적으로 `df = data.copy()` 후 변경 → **원본 `origin`은 절대 건드리지 않는다.**
  (연주님이 이미 알고 계신 pandas 원칙: `astype`은 in-place가 아니라 새 객체를 반환 → 반드시 재할당해서 받아야 함)
- 함수 마지막에 `df.info()`를 자동 출력 → 변환이 의도대로 됐는지 즉시 눈으로 확인하는 습관을 강제함.

| 파라미터 | 용도 |
|---|---|
| `as_int` / `as_float` | 수치 연산할 컬럼 |
| `as_string` | ID처럼 연산 의미 없는 문자열 |
| `as_category` | 값의 종류가 제한된 범주형 (성별, 등급 등) |
| `as_datetime` | 날짜/시간 |

> 💡 **재사용 팁**: 새 데이터셋을 받으면 제일 먼저 `origin.info()`로 dtype을 훑고,
> "이건 카테고리, 이건 날짜"라고 판단되는 컬럼을 리스트로 모아 `set_type()` 한 번에 넘긴다.

---

### ② 중복 점검 — `check_duplicates()`

```python
df2 = my_qtcheck.check_duplicates(df1)
```

- `duplicated()`로 완전히 동일한 행을 찾아 개수를 세고, `drop=True`(기본값)면 `drop_duplicates()`로 제거.
- **왜 중복을 먼저 제거해야 하나?** 중복 행이 남아있으면 평균·분산·개수 기반 통계(예: `value_counts`, `mean`)가
  실제보다 특정 값 쪽으로 과대 반영되어 왜곡된다. 통계를 내기 전에 반드시 걷어내는 이유.

---

### ③ 결측치 점검 — `check_missing_values()`

```python
my_qtcheck.check_missing_values(df2)
```

- 컬럼별 `isna().sum()`(개수)과 `/ len(data) * 100`(비율)을 데이터프레임으로 반환.
- 이 실습에서는 "결측치 없음"으로 끝났지만, 실제로는 이 결과를 보고
  **삭제할지 / 대체(imputation)할지 / 그대로 둘지**를 결정하는 분기점이 된다.
- 비율까지 같이 보는 이유: 개수만 보면 "50개 결측"이 큰지 작은지 판단이 안 되지만,
  "전체의 0.3%"라고 하면 바로 감이 온다.

---

### ④ 품질 검사 결과 저장

```python
df2.to_excel("diamonds_qtcheck.xlsx", index=False)
```

- 자료형 변환 + 중복 제거 + (결측 처리)까지 끝난 **"클린 데이터"의 스냅샷**을 별도 파일로 남기는 단계.
- 이후 통계 분석이나 모델링에서 매번 원본부터 다시 정제할 필요 없이 이 파일을 불러오면 됨 (재현성 확보).

---

### ⑤-1 범주형 기술 통계 — `categorical_summary()`

```python
category_desc = my_qtcheck.categorical_summary(df2, save_path="diamonds_category_summary.xlsx")
```

- `columns`를 안 주면 `get_categorical_column_names()`로 **category 타입 컬럼을 자동 탐색** → ①에서 `set_type`을 안 했으면 여기서 아무것도 안 잡힌다는 뜻 (순서가 중요한 실제 이유).
- `describe(include="category")` → 범주형 변수의 `count`, `unique`, `top`(최빈값), `freq`(최빈값 빈도) 반환.
- `value_counts=True`면 컬럼별로 각 범주의 등장 횟수까지 상세 출력.
- `save_path`가 있으면:
  - 요약표는 시트 "Summary"에,
  - 컬럼별 `value_counts`는 **컬럼명을 시트명으로 삼아 각각 추가(append)** 저장.
  - `ExcelWriter(mode='a', engine='openpyxl')` → 기존 파일에 시트를 덧붙이는 방식. (`mode='a'`를 안 쓰면 매번 파일이 통째로 덮어써짐 — 주의할 개념)

---

### ⑤-2 숫자형 기술 통계 — `numerical_summary()`

가장 정보량이 많은 함수. 단순 `describe()`를 넘어서 **분포 진단 + 이상치 탐지 + 변환 추천**까지 자동화.

```python
desc_df = my_qtcheck.numerical_summary(df2, save_path="diamonds_numerical_summary.xlsx")
```

단계별로 뜯어보면:

1. **기본 기술통계**: `describe().T` → 행=컬럼, 열=통계량으로 전치(가독성용).
2. **평균-중앙값 상대 차이율** `rel_diff = |mean - median| / median`
   - 평균과 중앙값이 크게 다르면 분포가 한쪽으로 치우쳤다는 신호.
   - `< 0.1` → `similar`(대칭에 가까움), `< 0.5` → `diff`, 그 이상 → `large_diff`
3. **IQR 기반 이상치 경계**
   - `IQR = Q3 - Q1`
   - 상한 `upper_bound = Q3 + 1.5*IQR`, 하한 `lower_bound = Q1 - 1.5*IQR`
   - 통계학 표준 이상치 판별 규칙 (박스플롯의 "수염" 밖 = 이상치).
4. **이상치 개수/비율 집계**: 상한 초과, 하한 미만, 전체 이상치 각각 개수·비율 계산.
5. **왜도(skewness) 진단**
   - `skew < -0.5` → `left tail`(왼쪽 꼬리, 음의 왜도), `skew > 0.5` → `right tail`(오른쪽 꼬리), 그 외 `symmetric`
   - 왜도는 "분포가 어느 방향으로 길게 늘어졌는가"를 나타냄.
6. **첨도(kurtosis) 진단**
   - `kurt < 0` → `platykurtic`(납작한 분포), `kurt > 0` → `leptokurtic`(뾰족한 분포), `kurt == 0` → `mesokurtic`(정규분포에 가까움)
   - pandas의 `kurt()`는 **초과 첨도(excess kurtosis)** 기준 (정규분포=0).
7. **로그 변환 필요성 자동 판정** (`judge_log_transform`)

   | 조건 | 판정 | 의미 |
   |---|---|---|
   | `skew >= 1` | `log1p` | 강한 오른쪽 꼬리 → 로그변환 권장 |
   | `0.5 < skew < 1` and `kurt > 0` | `log1p` | 오른쪽 꼬리 + 뾰족함 → 로그변환 권장 |
   | `skew <= -1` | `reverse_log1p` | 강한 왼쪽 꼬리 |
   | `-1 < skew < -0.5` and `kurt > 0` | `reverse_log1p` | 왼쪽 꼬리 + 뾰족함 |
   | 그 외 | `none` | 변환 불필요 (대칭에 가까움) |

   → 회귀분석/모델링 전에 "이 변수는 로그를 씌워야 하나?"를 매번 직접 판단하지 않고
   함수가 1차 스크리닝을 해주는 것. (최종 판단은 도메인 지식+시각화로 재확인 권장)

---

## 3. 오늘 배운 것 중 앞으로 계속 써먹을 "핵심 원칙" 3가지

1. **순서를 지킨다**: 자료형 변환 → 중복 → 결측치 → (저장) → 기술통계.
   순서를 어기면 함수가 컬럼을 잘못 인식하거나 통계가 왜곡됨.
2. **원본은 보존한다**: 모든 함수가 `data.copy()`로 시작 → 원본 df는 항상 그대로 남아있고,
   변환된 결과는 새 변수(`df1`, `df2`, ...)에 순차적으로 담아 **단계별 추적이 가능**하게 만든다.
   (pandas 핵심 원칙: `astype`, `drop`, `rename` 등은 원본을 바꾸지 않고 새 객체를 반환 → 반드시 재할당)
3. **함수는 재사용을 전제로 만든다**: `my_qtcheck.py`는 diamonds 전용이 아니라
   **어떤 데이터셋이 오든 그대로 import해서 쓰는 범용 품질검사 모듈**이다.
   새 프로젝트 시작할 때마다:
   ```python
   from helpers import my_qtcheck

   df1 = my_qtcheck.set_type(origin, as_category=[...])
   df2 = my_qtcheck.check_duplicates(df1)
   my_qtcheck.check_missing_values(df2)
   df2.to_excel("xxx_qtcheck.xlsx", index=False)
   my_qtcheck.categorical_summary(df2, save_path="xxx_category_summary.xlsx")
   my_qtcheck.numerical_summary(df2, save_path="xxx_numerical_summary.xlsx")
   ```
   이 6줄이 **모든 프로젝트의 시작점(템플릿)** 이 된다.

---

## 4. 다음에 새 데이터 받으면 쓸 체크리스트

- [ ] `origin.info()`로 dtype 확인
- [ ] category/datetime으로 바꿔야 할 컬럼 리스트업 → `set_type()`
- [ ] `check_duplicates()`로 중복 제거
- [ ] `check_missing_values()`로 결측 확인 → 처리 방침 결정(삭제/대체/보류)
- [ ] 클린 데이터 `to_excel()`로 스냅샷 저장
- [ ] `categorical_summary()`로 범주형 분포 확인 (최빈값, 유니크 개수)
- [ ] `numerical_summary()`로 숫자형 분포 확인 → 이상치 비율, 왜도/첨도, 로그변환 필요 여부 체크
- [ ] 이상치·왜도가 큰 컬럼은 시각화(boxplot/histogram)로 한 번 더 눈으로 검증

---

## 5. 헷갈렸던 부분 / 다시 볼 것 (필요시 채워넣기)

- [ ]
- [ ]

