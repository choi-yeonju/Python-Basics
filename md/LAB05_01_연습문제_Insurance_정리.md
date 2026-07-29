# [LAB 05 - 연습문제] 보험료(Insurance) 데이터 품질 점검
> Boston 데이터셋과 동일한 템플릿 적용 + 실제 인사이트 도출까지

---

## 🗺️ 전체 흐름 (Boston과 동일한 템플릿)

```
데이터 로드
  ↓
① 자료형 점검/변환     → sex, smoker, region → category
  ↓
② 중복 점검            → 중복 1건 발견 → 제거
  ↓
③ 형식·표기·단위 점검   → 라벨링 필요 항목 발견
  ↓
④ 결측치 점검          → 0건
  ↓
⑤ 품질점검 결과 저장
  ↓
⑥ 기술 통계량 표 생성
  ↓
⑦ 중심 수준 파악
  ↓
⑧ 이상치 신호 감지     → Charges에서 다량 발견
  ↓
⑨ 비대칭 신호 확인     → Charges 로그변환 필요 판정
  ↓
⑩ 인사이트 정리
```

> 💡 이 템플릿은 Boston LAB에서 짠 코드와 **완전히 동일한 구조**야. 데이터셋만 바뀌었을 뿐 흐름은 그대로니까, "이 단계에서 뭘 하는지"는 지난 정리 파일을 참고하면 되고, 여기서는 **insurance 데이터에서 실제로 어떤 값이 나왔는지, 그걸 어떻게 해석했는지**에 집중할게.

---

## ① 자료형 점검 및 변환

```python
df1 = origin.copy()
df1["sex"] = df1["sex"].astype("category")
df1["smoker"] = df1["smoker"].astype("category")
df1["region"] = df1["region"].astype("category")
```

- 명목형 변수 3개(`sex`, `smoker`, `region`)를 한 번에 category로 변환
- Boston은 CHAS 하나였는데, 여기는 **명목형 변수가 여러 개인 경우** → 컬럼마다 반복 처리

> 💡 이런 반복 작업이 바로 `my_qtcheck.set_type(df, as_category=['sex','smoker','region'])` 처럼 리스트로 한 번에 처리하는 함수가 왜 필요한지 보여주는 부분

---

## ② 데이터 중복 점검

```python
dup = df1.duplicated()
dup.sum()   # → 1건 발견!

df2 = df1.drop_duplicates()
df2.duplicated().sum()   # → 0 (제거 확인)
```

> ⚠️ **Boston에서는 중복이 없었지만, 이번엔 실제로 1건이 발견됨**
> → "이번 데이터엔 중복 없겠지"라고 넘기지 않고 매번 검사하는 게 중요한 이유가 바로 이런 경우 때문

---

## ③ 형식·표기·단위 점검

### 명목형 변수 단위 확인

```python
fields = df2.select_dtypes(include=["category"]).columns

for field in fields:
    display(df2[field].value_counts())
```

**결과 해석:**
> "데이터의 종류에는 문제가 없으나, 라벨링 처리는 필요하다"

- 즉 `sex`(male/female), `smoker`(yes/no), `region`(4개 지역) 자체는 이상한 값 없이 깨끗함
- 하지만 분석/모델링 단계에서는 **숫자로 인코딩(라벨링)** 이 필요하다는 뜻 → 품질에는 문제 없지만 다음 단계를 위한 메모

### 연속형 변수 min/max 확인

```python
fields = df2.select_dtypes(include="number").columns.to_list()

minmax = []
for field in fields:
    min_value = df2[field].min()
    max_value = df2[field].max()
    minmax.append({"min": min_value, "max": max_value})

minmax_df = DataFrame(minmax, index=fields)
```

**결과 해석:**
> "데이터 명세의 내용에서 크게 벗어나는 범위는 없는 것으로 판단됨"

- age, bmi, children, charges 모두 정상 범위 내 → 추가 조치 불필요

---

## ④ 결측치 점검

```python
na_count = df2.isna().sum()
rows, cols = df2.shape
na_ratio = na_count / rows
```

**결과:** 결측치 0건 → 가장 깔끔한 케이스 (별도 처리 불필요)

---

## ⑤ 품질 점검 결과 저장

```python
df2.to_excel("insurance_qtcheck.xlsx", index=False)
```

이후 새 작업에서 다시 불러와서 진행:

```python
origin_qt = load_data("insurance_qtcheck")

df3 = origin_qt.copy()
df3["sex"] = df3["sex"].astype("category")
df3["smoker"] = df3["smoker"].astype("category")
df3["region"] = df3["region"].astype("category")
```

> ⚠️ 역시 엑셀에서 다시 불러오면 category 타입이 풀리므로 재변환 필수 (Boston과 동일 패턴)

---

## ⑥ 기술 통계량 표 생성

```python
desc_df = df3.describe().T              # 연속형
cate_desc_df = df3.describe(include='category').T   # 명목형

# 명목형 값별 개수/비율
cate_fields = df3.select_dtypes(include='category').columns
for field in cate_fields:
    vcount = df3[field].value_counts()
    percent = vcount / df3.shape[0]
    df = DataFrame({'count': vcount, 'percent': percent})
    display(df)
```

---

## ⑦ 중심 수준 파악 — 평균 vs 중앙값

```python
desc_df['rel_diff'] = abs(desc_df['mean'] - desc_df['50%']) / desc_df['50%']

conditions = [desc_df['rel_diff'] < 0.1, desc_df['rel_diff'] < 0.5]
choices = ['similar', 'diff']
desc_df['rdiff_flag'] = np.select(conditions, choices, default='large_diff')
```

**이번 자료에 정리된 판정 기준표 (지난번보다 더 명확하게 정리됨):**

| 구간 | 의미 |
|------|------|
| 0 ~ 0.1 | 평균과 중앙값이 거의 비슷함 |
| 0.1 ~ 0.5 | 약간 차이 있음 |
| 0.5 이상 | 차이 큼, 왜도·극단값 의심 |

**실제 결과:**
- `charges`(의료비) → rel_diff **0.415** → "diff" 구간, 왜도 의심 신호
- `age`, `bmi` → rel_diff < 0.01 → "similar", 매우 대칭적

---

## ⑧ 이상치 신호 감지 — IQR 방식

```python
desc_df['iqr'] = desc_df['75%'] - desc_df['25%']
desc_df['upper_bound'] = desc_df['75%'] + 1.5 * desc_df['iqr']
desc_df['lower_bound'] = desc_df['25%'] - 1.5 * desc_df['iqr']

cate_fields = df3.select_dtypes(include='category').columns
df4 = df3.drop(columns=cate_fields)   # 명목형 제외

desc_df['upper_outliers'] = (df4 > desc_df['upper_bound']).sum()
desc_df['upper_outliers_ratio'] = desc_df['upper_outliers'] / df4.shape[0]

desc_df['lower_outliers'] = (df4 < desc_df['lower_bound']).sum()
desc_df['lower_outliers_ratio'] = desc_df['lower_outliers'] / df4.shape[0]

desc_df['outliers'] = desc_df['upper_outliers'] + desc_df['lower_outliers']
desc_df['outliers_ratio'] = desc_df['outliers'] / df4.shape[0]
```

**실제 결과:**
- `charges` → 상한 이상치 **139개 (10.4%)** → 상당히 많음
- `age`, `bmi`, `children` → 이상치 거의 없거나 미미

> 💡 Boston 때보다 **이상치 비율이 큰 변수가 실제로 나온 케이스**라, "정상 범위 안에서도 분포 모양이 한쪽으로 쏠릴 수 있다"는 걸 직접 확인한 사례

---

## ⑨ 비대칭 신호 확인 — 왜도/첨도

```python
desc_df['skew'] = df4.skew()
conditions_skew = [(desc_df['skew'] < -0.5), (desc_df['skew'] > 0.5)]
choices_skew = ['left tail', 'right tail']
desc_df['skew_interpret'] = np.select(conditions_skew, choices_skew, default='symmetric')

desc_df['kurt'] = df4.kurt()
conditions_kurt = [(desc_df['kurt'] < 0), (desc_df['kurt'] > 0)]
choices_kurt = ['platykurtic', 'leptokurtic']
desc_df['kurt_interpret'] = np.select(conditions_kurt, choices_kurt, default='mesokurtic')

def judge_log_transform(skew, kurt):
    if skew >= 1:
        return "log1p"
    elif skew > 0.5 and kurt > 0:
        return "log1p"
    elif skew <= -1:
        return "reverse_log1p"
    elif skew < -0.5 and kurt > 0:
        return "reverse_log1p"
    else:
        return "none"

desc_df['log_need'] = desc_df.apply(lambda row: judge_log_transform(row['skew'], row['kurt']), axis=1)
```

**실제 결과:**
- `charges` → 왜도 **1.515** → `skew >= 1` 조건 충족 → **log1p 필수**
- `children` → 왜도 **0.937** → 역시 `skew >= 1`은 아니지만 우측 꼬리 → log 변환 권장
- `age`, `bmi` → 대칭에 가까움 → 변환 불필요

---

## ⑩ 최종 저장

```python
desc_df.to_excel("insurance_qtcheck_desc.xlsx", index=True)
```

---

## 💡 도출된 인사이트 정리

### 1. Charges (의료보험 청구 비용) — 가장 주의가 필요한 변수
| 지표 | 값 | 해석 |
|------|-----|------|
| 평균 | 13,279 | |
| 중앙값 | 9,386 | 평균이 중앙값보다 훨씬 큼 |
| rel_diff | 0.415 | "diff" 구간 (큰 차이) |
| 왜도 | 1.515 | 강한 우측 꼬리 |
| 상한 이상치 | 139개 (10.4%) | 고액 청구 건이 다수 |
| 처리 방향 | **log1p 필수** | |

→ 소수의 고액 의료비 청구 건이 평균을 끌어올리는 전형적인 패턴. 분석/모델링 전에 로그 변환이 거의 필수.

### 2. Children (부양 자녀 수)
- 왜도 0.937로 우측 꼬리
- 이상치는 없지만 분포 정상화를 위해 로그 변환 권장

### 3. Age, BMI — 건강한 변수
- 평균-중앙값 거의 일치 (rel_diff < 0.01)
- 대칭적 분포, 이상치 거의 없음 → 추가 처리 불필요

### 4. 데이터 품질 총평
- 결측치 0건
- 완전 중복 1건 발견 → 제거 (1,338 → 1,337행)
- 전반적으로 양호하나, **charges 변수 하나가 핵심 처리 대상**

---

## 🔑 Boston 케이스와 비교했을 때 새로 확인한 것

| 항목 | Boston | Insurance |
|------|--------|-----------|
| 중복 데이터 | 없음 | **1건 발견 → 실제 제거 경험** |
| 명목형 변수 개수 | 1개 (CHAS) | **3개 (sex, smoker, region) → 반복 처리 경험** |
| 단위 점검 결과 | 정상 | 정상이지만 **"라벨링 필요"라는 후속 작업 메모 발견** |
| 강한 이상치/왜도 변수 | 뚜렷하지 않음 | **charges에서 명확하게 발견 → 실전 로그변환 케이스 학습** |

→ 이번 insurance 데이터는 Boston보다 **"실제로 문제가 있는 변수를 발견하고 조치를 결정하는" 전 과정을 한 번에 경험**한 케이스라서, 템플릿이 단순 형식이 아니라 실제로 작동하는 걸 확인한 의미가 큼.
