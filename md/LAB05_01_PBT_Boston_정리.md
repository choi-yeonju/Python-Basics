# [LAB 05 - PBT] Boston Housing 데이터 품질 점검

> 직접 코드로 풀어서 작성한 버전 (= my_qtcheck 함수들이 내부적으로 하는 일)

---

## 🗺️ 전체 흐름 한눈에 보기

```
데이터 로드
  ↓
① 자료형 점검/변환     → CHAS를 category로
  ↓
② 중복 점검            → duplicated() / drop_duplicates()
  ↓
③ 형식·표기·단위 점검   → value_counts(), min/max 구간 확인  ← 이번 LAB에서 새로 추가된 단계
  ↓
④ 결측치 점검          → isna().sum() / 비율 계산
  ↓
⑤ 품질점검 결과 저장    → to_excel()
  ↓
[새 노트북에서 이어서 시작]
  ↓
⑥ 기술 통계량 표 생성   → describe()
  ↓
⑦ 중심 수준 파악        → 평균-중앙값 상대 차이율
  ↓
⑧ 이상치 신호 감지      → IQR 기반 경계값
  ↓
⑨ 비대칭 신호 확인      → 왜도/첨도 → 로그변환 판단
```

> 💡 지난번 정리한 `my_qtcheck.py` 의 `numerical_summary()` 함수가 바로 ⑥~⑨번을 한 번에 처리해주는 함수야. 이번엔 그 내부를 한 줄씩 직접 짜본 거.

---

## ① 자료형 점검 및 변환

```python
origin.info()                              # 전체 컬럼 타입 확인

df1 = origin.copy()                        # 원본 보존 (항상 copy 먼저!)
df1["CHAS"] = df1["CHAS"].astype("category")
df1.info()                                 # 변환 확인
```

**왜 CHAS만 category로 바꾸나?**

- `CHAS`는 숫자(0/1)로 저장되어 있지만 실제로는 **"강 경계 여부"** 라는 범주형 의미를 가진 변수임
- 숫자처럼 보여도 의미상 범주형이면 `category`로 바꿔야 `describe(include='category')` 등에서 명목형으로 제대로 잡힘

> ⚠️ 포인트: `.astype()`은 원본을 바꾸지 않으므로 반드시 `df1["CHAS"] = ...` 형태로 재할당해야 함

---

## ② 데이터 중복 점검

```python
dup = df1.duplicated()   # 각 행이 중복인지 True/False
dup.sum()                 # 중복 행 개수

df2 = df1.drop_duplicates()   # 중복 제거 (새 변수로 받기)
df2.duplicated().sum()        # 제거 후 재확인 (0이어야 정상)
```

- `duplicated()` : 첫 번째로 나온 행은 False, 그 이후 **반복된** 행만 True
- 제거 후 다시 `duplicated().sum()`으로 검증하는 습관이 중요 (검증 없이 넘어가면 실수 발견 못함)

---

## ③ 형식 · 표기 · 단위 점검 ⭐ (이번에 새로 배운 단계)

> 지금까지는 "타입이 맞나, 중복이 있나"만 봤다면, 이 단계는 **"값 자체가 말이 되는가"** 를 보는 단계

### 1) 명목형 변수 → 값의 종류 확인

```python
df2['CHAS'].value_counts()
```

- 명목형 변수는 분포(어떤 값들이 몇 개씩 있는지)를 보면 단위/형식 이상을 발견할 수 있음
- 예: CHAS가 0, 1만 있어야 하는데 2나 -1 같은 값이 섞여있으면 데이터 이상

### 2) 연속형 변수 → 필드명 추출

```python
fields = df2.select_dtypes(include="number").columns.to_list()
print(fields)
```

- `select_dtypes(include="number")` : int/float 타입 컬럼만 골라냄
- `.columns.to_list()` : 컬럼 이름만 리스트로 추출

### 3) 연속형 변수 → 최소/최대값 구간 확인

```python
minmax = []

for field in fields:
    min_value = df2[field].min()
    max_value = df2[field].max()
    minmax.append({"min": min_value, "max": max_value})

minmax_df = DataFrame(minmax, index=fields)
minmax_df
```

**핵심 아이디어:**

- 각 변수의 min/max를 구해서, **데이터 명세서(설명서)에 적힌 정상 범위와 비교**
- 예: 비율(%) 변수인데 max가 150이 나오면 → 데이터 입력 오류 의심
- 예: 위도/경도인데 범위를 벗어나면 → 잘못된 좌표

> 💡 `for문 + append`로 딕셔너리 리스트를 만들고, 마지막에 `DataFrame()`으로 한번에 변환하는 패턴은 자주 쓰이니 익혀두면 좋음

---

## ④ 결측치 점검

```python
na_count = df2.isna().sum()       # 컬럼별 결측치 개수
na_count

rows, cols = df2.shape             # 행, 열 개수 확인
print(f"rows: {rows}, cols: {cols}")

na_ratio = na_count / rows         # 결측치 비율
na_ratio
```

**판단 기준 (이번 자료에서 제시된 룰):**

- 결측치 비율이 **5% 미만**이면 → 전체 분석에 미치는 영향 미미 → 삭제해도 괜찮음
- 5% 이상이면 → 삭제보다는 대체(imputation) 등을 고려해야 함

---

## ⑤ 품질 점검 결과 저장

```python
df2.to_excel("boston_qtcheck.xlsx", index=False)
```

- `index=False` : 판다스 인덱스 번호는 저장하지 않음 (안 그러면 불필요한 열이 추가됨)
- 이렇게 저장해두면 다음 분석(기술통계, 시각화 등)은 **새 노트북에서 이 파일을 불러와서** 이어가는 게 권장 방식

```python
# 실제 권장 방식
origin_qt = read_excel("boston_qtcheck.xlsx")

# 수업 자료라서 라이브러리로 대체 로드 (실무에선 위 방식 사용)
origin_qt = load_data("boston_qtcheck")
```

> ⚠️ **중요**: 엑셀로 저장 후 다시 불러오면 `category` 타입 정보는 사라짐(엑셀이 타입을 기억 못함). 그래서 불러올 때마다 `astype('category')`를 다시 해줘야 함

```python
df3 = origin_qt.copy()
df3['CHAS'] = df3['CHAS'].astype('category')
df3.info()
```

---

## ⑥ 기술 통계량 표 생성

### 연속형 변수

```python
desc_df = df3.describe().T
desc_df
```

- `describe()` 는 기본적으로 **숫자형 컬럼만** 대상으로 함 (category는 자동 제외됨)
- `.T` : 행/열을 뒤집어서 변수가 행으로 오게 함 → 변수 많을 때 보기 편함

> 💡 위도(LAT)/경도(LON) 같은 컬럼은 숫자형이어도 "연속형 변수"로서 의미있는 통계가 아니므로 분석에서 제외 고려 필요 (좌표는 평균을 내도 의미 없음)

### 명목형 변수

```python
cate_desc_df = df3.describe(include='category').T
cate_desc_df
```

| 통계량 | 의미                |
| ------ | ------------------- |
| count  | 결측 아닌 데이터 수 |
| unique | 값의 종류 수        |
| top    | 최빈값              |
| freq   | 최빈값의 빈도수     |

### 명목형 변수 — 값별 개수와 비율

```python
cate_fields = df3.select_dtypes(include='category').columns

for field in cate_fields:
    vcount = df3[field].value_counts()       # 종류별 개수
    percent = vcount / df3.shape[0]            # 전체 대비 비율

    df = DataFrame({'count': vcount, 'percent': percent})
    display(df)
```

- `value_counts()`만으로는 비율을 알 수 없어서, 직접 나눠서 `percent` 컬럼을 추가해줌

---

## ⑦ 중심 수준 파악 — 평균과 중앙값 비교

```python
desc_df['rel_diff'] = abs(desc_df['mean'] - desc_df['50%']) / desc_df['50%']

conditions = [desc_df['rel_diff'] < 0.1, desc_df['rel_diff'] < 0.5]
choices = ['similar', 'diff']
desc_df['rdiff_flag'] = np.select(conditions, choices, default='large_diff')
```

**왜 평균과 중앙값을 비교하나?**

- 평균은 이상치에 민감하고, 중앙값은 이상치에 둔감함
- 둘이 비슷하면 → 분포가 대체로 대칭적 (정상)
- 둘 차이가 크면 → 이상치나 비대칭 분포 가능성 있음 → 추가 점검 필요 신호

```
rel_diff < 0.1   → similar      (평균 ≈ 중앙값, 정상적 분포)
rel_diff < 0.5   → diff         (어느정도 차이, 주의)
그 외             → large_diff   (큰 차이, 이상치/왜곡 의심)
```

**`np.select()` 사용법 핵심:**

```python
np.select(조건_리스트, 결과_리스트, default=기본값)
```

- 조건 리스트를 위에서부터 순서대로 검사
- 가장 먼저 True인 조건의 결과를 채택
- 어느 조건도 안 맞으면 `default` 값 사용

---

## ⑧ 이상치 신호 감지 — IQR 방식

### 1) IQR 및 경계값 계산

```python
desc_df['iqr'] = desc_df['75%'] - desc_df['25%']
desc_df['upper_bound'] = desc_df['75%'] + 1.5 * desc_df['iqr']
desc_df['lower_bound'] = desc_df['25%'] - 1.5 * desc_df['iqr']
```

```
IQR (사분위 범위) = Q3(75%) - Q1(25%)

정상 범위:  [Q1 - 1.5×IQR,  Q3 + 1.5×IQR]
            ↑ 이 범위를 벗어나면 통계적으로 "이상치"로 간주
```

### 2) 명목형 변수 제외 (이상치는 연속형 개념이므로)

```python
cate_fields = df3.select_dtypes(include='category').columns
df4 = df3.drop(columns=cate_fields)
df4.head()
```

> 💡 `drop(columns=...)` 도 기본은 원본 비보존 → 새 변수(`df4`)로 받아야 함

### 3) 상한/하한 이상치 탐지

```python
# 상한 이상치
desc_df['upper_outliers'] = (df4 > desc_df['upper_bound']).sum()
desc_df['upper_outliers_ratio'] = desc_df['upper_outliers'] / df4.shape[0]

# 하한 이상치
desc_df['lower_outliers'] = (df4 < desc_df['lower_bound']).sum()
desc_df['lower_outliers_ratio'] = desc_df['lower_outliers'] / df4.shape[0]
```

**동작 원리:**

- `df4 > desc_df['upper_bound']` → 비교 시 컬럼명 기준으로 자동 매칭되어 True/False DataFrame 생성
- `.sum()` → 컬럼별로 True(=1)의 개수를 합산 → 이상치 개수

### 4) 통합 집계

```python
desc_df['outliers'] = desc_df['upper_outliers'] + desc_df['lower_outliers']
desc_df['outliers_ratio'] = desc_df['outliers'] / df3.shape[0]
```

---

## ⑨ 비대칭 신호 확인 — 왜도 / 첨도

### 1) 왜도(Skewness)

```python
desc_df['skew'] = df4.skew()

conditions_skew = [(desc_df['skew'] < -0.5), (desc_df['skew'] > 0.5)]
choices_skew = ['left tail', 'right tail']
desc_df['skew_interpret'] = np.select(conditions_skew, choices_skew, default='symmetric')
```

```
왜도 < -0.5  →  left tail   (왼쪽으로 긴 꼬리, 데이터는 오른쪽에 몰림)
왜도 > +0.5  →  right tail  (오른쪽으로 긴 꼬리, 데이터는 왼쪽에 몰림)
그 외        →  symmetric   (좌우 대칭에 가까움)
```

### 2) 첨도(Kurtosis)

```python
desc_df['kurt'] = df4.kurt()

conditions_kurt = [(desc_df['kurt'] < 0), (desc_df['kurt'] > 0)]
choices_kurt = ['platykurtic', 'leptokurtic']
desc_df['kurt_interpret'] = np.select(conditions_kurt, choices_kurt, default='mesokurtic')
```

```
첨도 < 0   →  platykurtic   (납작한 분포, 분산이 넓게 퍼짐)
첨도 > 0   →  leptokurtic   (뾰족한 분포, 평균 근처에 밀집 + 꼬리가 두꺼움)
첨도 = 0   →  mesokurtic    (정규분포와 유사한 뾰족함)
```

> pandas의 `kurt()`는 "초과 첨도(excess kurtosis)" 기준 → 정규분포의 첨도를 0으로 맞춰서 계산함

### 3) 로그 변환 필요성 판정 함수

```python
def judge_log_transform(skew, kurt):
    if skew >= 1:
        return "log1p"                # 강한 우측 꼬리
    elif skew > 0.5 and kurt > 0:
        return "log1p"                # 우측 꼬리 + 뾰족함
    elif skew <= -1:
        return "reverse_log1p"        # 강한 좌측 꼬리
    elif skew < -0.5 and kurt > 0:
        return "reverse_log1p"        # 좌측 꼬리 + 뾰족함
    else:
        return "none"

desc_df['log_need'] = desc_df.apply(
    lambda row: judge_log_transform(row['skew'], row['kurt']), axis=1
)
```

**판정 로직 정리표**

| skew 조건        | kurt 조건 | 결과          | 의미                                 |
| ---------------- | --------- | ------------- | ------------------------------------ |
| skew ≥ 1        | 무관      | log1p         | 강한 우측 꼬리 → 로그변환 강력 추천 |
| 0.5 < skew < 1   | kurt > 0  | log1p         | 우측 꼬리 + 뾰족 → 로그변환 권장    |
| skew ≤ -1       | 무관      | reverse_log1p | 강한 좌측 꼬리                       |
| -1 < skew < -0.5 | kurt > 0  | reverse_log1p | 좌측 꼬리 + 뾰족                     |
| 그 외            | -         | none          | 변환 불필요                          |

**`apply(axis=1)` 동작 원리:**

- `axis=1` → 행 단위로 함수를 적용 (각 변수/row마다 skew, kurt 값을 꺼내 함수에 넣음)
- `axis=0`(기본값)이면 열 단위 적용이므로 헷갈리지 않게 주의

---

## ⑩ 최종 저장

```python
desc_df.to_excel("boston_qtcheck_desc.xlsx")
```

---

## 🔑 이번 LAB에서 새로 배운 핵심 3가지

1. **③ 형식·표기·단위 점검** 단계가 추가됨

   - 명목형: `value_counts()`로 값 종류 확인
   - 연속형: min/max로 정상 범위 벗어나는지 확인
   - → 지난 LAB(diamonds)에는 없던, **"값이 말이 되는가"** 를 검증하는 단계
2. **엑셀로 저장 후 다시 불러올 때 category 타입이 사라진다**

   - 매번 다시 `.astype('category')` 해줘야 함
   - 함수화된 버전(`my_qtcheck.set_type`)을 쓰면 이 과정을 한 줄로 줄일 수 있는 이유가 바로 이것
3. **위도/경도처럼 숫자형이어도 통계적 의미가 없는 컬럼**은 기술통계·이상치 분석에서 제외를 고려해야 함

   - 타입은 number지만 "연속형 변수"로서의 의미가 없는 경우를 구분하는 게 중요

---

## 📌 함수화 버전과 비교

| 단계                    | PBT(직접 코드)                                | 함수화 버전                                    |
| ----------------------- | --------------------------------------------- | ---------------------------------------------- |
| 타입 변환               | `df["X"] = df["X"].astype("category")`      | `my_qtcheck.set_type(df, as_category=[...])` |
| 중복 제거               | `duplicated()` → `drop_duplicates()`     | `my_qtcheck.check_duplicates(df)`            |
| 결측치                  | `isna().sum()` / 비율 직접 계산             | `my_qtcheck.check_missing_values(df)`        |
| 범주형 통계             | `describe()` + `value_counts()` 직접 반복 | `my_qtcheck.categorical_summary(df)`         |
| 수치형 통계+이상치+왜도 | 한 줄씩 직접 계산 (이번 자료 ⑥~⑨)           | `my_qtcheck.numerical_summary(df)` 한 줄     |

> 결론: 함수화 버전은 **이번에 직접 짠 코드를 그대로 재사용 가능하게 묶어놓은 것**. 원리를 알고 쓰는 것과 모르고 쓰는 것의 차이가 크니, 이번처럼 직접 풀어서 한 번 짜보는 연습이 중요함.

