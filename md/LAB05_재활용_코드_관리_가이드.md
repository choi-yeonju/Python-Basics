# 재활용 코드 모듈 관리 가이드 (`my_qtcheck.py`, `my_plot.py`)

> 지금부터 이 두 파일은 "실습 결과물"이 아니라 **연주 개인 유틸리티 라이브러리**임.
> 매 프로젝트마다 새로 짜는 게 아니라, 여기에 계속 쌓아가면서 갖다 쓰는 자산으로 관리하는 게 목표.

---

## 0. 지금 두 모듈에 뭐가 들어있는지 (현재 카탈로그)

### `my_qtcheck.py` — 품질검사/기술통계
| 함수 | 역할 |
|---|---|
| `set_type()` | 컬럼 타입 일괄 변경 (int/float/str/category/datetime) |
| `get_number_column_names()` | 숫자형 컬럼명 리스트 |
| `get_categorical_column_names()` | 범주형 컬럼명 리스트 |
| `check_duplicates()` | 중복 행 검사 + 제거 |
| `check_missing_values()` | 결측치 개수/비율 |
| `categorical_summary()` | 범주형 기술통계 + value_counts + 엑셀 저장 |
| `numerical_summary()` | 연속형 기술통계 + IQR 이상치 + 왜도/첨도 + 로그변환 판정 + 엑셀 저장 |

### `my_plot.py` — 시각화
| 함수 | 역할 |
|---|---|
| `init()` / `show()` | 그래프 캔버스 초기화/출력 공통 처리 (한글 폰트 자동 설정 포함) |
| `lineplot()` | 선 그래프 |
| `kdeplot()` | 커널밀도 (평균선 옵션 포함) |
| `histplot()` | 히스토그램 |
| `boxplot()` / `violinPlot()` | 분포 시각화 |
| `heatmap()` | 히트맵 (상관행렬 등) |
| `barplot()` / `countplot()` | 막대그래프 |

두 모듈 다 **"매개변수로 넘기고, `data.copy()`로 원본 안 건드리고, 마지막에 결과/그래프 반환·출력"**하는 동일한 설계 패턴을 따르고 있음. 이 패턴 자체가 앞으로 함수 추가할 때 지켜야 할 컨벤션임.

---

## 1. 폴더 구조 — 어디에 두고 어떻게 부를 것인가

지금 `Python-Basics` 레포(`C:\py_temp\15_LAB`) 기준으로 이렇게 잡는 걸 권장:

```
C:\py_temp\15_LAB\
├── helpers\                  ← 패키지 폴더 (반드시 __init__.py 있어야 import 가능)
│   ├── __init__.py           ← 비어있어도 됨, 이게 있어야 "패키지"로 인식됨
│   ├── my_qtcheck.py
│   ├── my_plot.py
│   └── my_stat.py            ← (나중에 Test/Model 파트 만들면 여기 추가 예정)
├── yeonju\                   ← 연주 개인 실습 파일들
│   ├── [yeonju]_LAB05_boston.ipynb
│   └── ...
└── data\                     ← 원본 데이터셋
```

**핵심 포인트:**
- `helpers` 안에 `__init__.py`가 없으면 파이썬이 그냥 폴더로 보고 패키지로 인식 못 해서 `import` 에러남. (지금 겪었던 module/path 이슈가 대체로 이거 아니면 아래 2번 문제일 확률 높음)
- 새 프로젝트(Olist 등) 폴더를 완전히 새로 팔 거면, `helpers` 폴더 자체를 통째로 복사해가거나, 아래 4번(패키지화)처럼 아예 어디서든 불러쓸 수 있게 만드는 걸 고려.

---

## 2. import 경로 문제 — 왜 자꾸 깨지는지

노트북 위치와 `helpers` 위치의 상대 경로가 안 맞으면 `ModuleNotFoundError`가 남. 두 가지 해결 방식:

### 방법 A — sys.path에 상위 경로 추가 (지금처럼 실습 폴더 구조 유지할 때)
```python
import sys
sys.path.append('..')   # 또는 절대경로: sys.path.append(r'C:\py_temp\15_LAB')

from helpers import my_qtcheck, my_plot
```
- 노트북이 `yeonju/` 안에 있고 `helpers/`가 한 단계 위에 있는 구조라면 `'..'`로 충분
- 폴더 구조 바뀌면 이 경로도 같이 깨지니, **레포 구조를 함부로 옮기지 않는 게 관리 원칙**

### 방법 B — 환경변수로 고정 경로 등록 (더 안정적, 여러 프로젝트에서 공유할 때)
`PYTHONPATH` 환경변수에 `helpers`의 부모 폴더를 등록해두면 `sys.path.append` 없이 어디서든 import 가능. Windows에서:
```
시스템 환경변수 편집 → PYTHONPATH 추가 → C:\py_temp\15_LAB
```
이러면 Olist 프로젝트 폴더가 따로 있어도 `from helpers import my_qtcheck`가 바로 먹힘. **여러 프로젝트에서 계속 재사용할 계획이면 이 방식을 추천.**

---

## 3. Git 관리 — 어떻게 버전 관리할 것인가

지금 겪었던 `.git` 손상 문제 재발 방지 + 라이브러리로서 버전 추적하는 방법:

### 커밋 단위 원칙
- **`helpers/` 안의 함수 수정은 반드시 별도 커밋으로 분리.** 실습 노트북 작업이랑 같이 묶어서 커밋하면 나중에 "언제 이 함수가 왜 바뀌었는지" 추적 안 됨.
```bash
git add helpers/my_qtcheck.py
git commit -m "qtcheck: numerical_summary에 kurt 조건 오타 수정 (reverse_log1p 앞 공백)"
```
- 커밋 메시지에 **어떤 함수를, 왜 바꿨는지** 남기기. "수정함" 같은 메시지는 6개월 후 아무 도움 안 됨.

### 브랜치는 지금 단계에선 오버스펙
- 지금처럼 혼자 쓰는 유틸 모듈 단계에서는 `main` 브랜치에 바로 커밋해도 무방. 다만 **큰 구조 변경**(예: `my_qtcheck.py`를 여러 파일로 쪼갠다든지)을 시도할 땐 임시 브랜치 하나 파서 작업 후 합치는 걸 권장:
```bash
git checkout -b refactor-qtcheck
# 작업
git checkout main
git merge refactor-qtcheck
```

### 레포 손상 방지 습관
- 이전에 겪은 `.git` 손상 복구 절차(`Remove-Item -Recurse -Force .git` → `git init` → ...)는 최후 수단. 평소엔:
  - 큰 파일(엑셀 결과물, 데이터셋 원본)은 `.gitignore`에 넣어서 레포에 안 올리기 — 용량 문제가 손상의 흔한 원인
  - 작업 중간중간 `git status`로 상태 확인하는 습관

`.gitignore` 예시:
```
*.xlsx
*.csv
data/
__pycache__/
.ipynb_checkpoints/
```
→ 데이터/결과 엑셀은 Google Drive 백업으로, 코드(`helpers/`, 노트북)만 GitHub으로 — 지금 쓰는 "GitHub=코드, Drive=파일" 구분과도 정확히 맞음.

---

## 4. 함수 추가/수정 워크플로우 — "실습에서 검증 → 모듈에 편입"

지금까지 해온 패턴이 사실 이상적인 워크플로우였음. 앞으로도 이 순서 유지:

```
1) 새 데이터셋 실습하면서 필요한 로직을 노트북에 직접 풀어서 짬 (PBT 방식)
   ↓
2) 여러 데이터셋에 반복 적용해보면서 "매번 똑같이 쓰는 부분"인지 검증
   ↓
3) 검증되면 helpers/ 모듈에 함수로 편입 (파라미터화해서 범용성 확보)
   ↓
4) 모듈에 넣은 뒤 최소 1개 이상의 다른 데이터셋으로 재검증 (버그 없는지)
   ↓
5) 커밋 (메시지에 어떤 함수 추가/변경했는지 명시)
```

**아무 로직이나 바로 모듈에 넣지 말 것.** 한 데이터셋에서만 통했던 로직을 성급하게 일반화하면, 다음 프로젝트에서 조용히 틀린 결과를 낼 수 있음. (예: `numerical_summary`가 특정 데이터에서만 맞는 임계값을 하드코딩하고 있었다면, 다른 데이터셋에 그대로 쓰다가 잘못된 로그변환 판정을 낼 수 있음)

---

## 5. 지금 코드에서 눈에 띄는 것들 — 관리 차원에서 짚어두면 좋을 부분

코드 자체 리뷰라기보단, "라이브러리로 계속 굴릴 때 문제가 될 수 있는 지점"만 짚음:

1. **`numerical_summary()`의 `judge_log_transform` 내부에 오타 하나 있음**
   ```python
   elif skew < -0.5 and kurt > 0 :
       return " reverse_log1p"   # ← 맨 앞에 공백 하나 들어가 있음 ("reverse_log1p"가 아니라 " reverse_log1p")
   ```
   지금은 `log_need` 컬럼 값으로만 쓰이니 눈으로 보는 데는 문제 없지만, 나중에 이 값으로 `if desc_df['log_need'] == 'reverse_log1p'` 같은 조건문을 짜면 공백 때문에 매칭 실패함. **모듈에 들어간 코드는 이런 사소한 오타가 오래 숨어있다가 나중에 터지는 게 제일 골치 아픈 버그 유형**이니, 지금 고쳐두는 게 좋음.

2. **`my_plot.py`의 `countplot()`이 `barplot()`을 그대로 복붙한 형태**
   ```python
   def countplot(...):
       ...
       sb.barplot(data=data, x=x, y=y, hue=hue, order=order, palette=palette)  # sb.countplot이 아니라 sb.barplot 호출
   ```
   함수명은 countplot인데 내부에서 seaborn의 `countplot`이 아니라 `barplot`을 호출하고 있음. 지금 정리 문서에서도 짚었던 "복붙 후 이름 안 바꾸는 습관"이 라이브러리 코드에도 들어간 사례. 이런 건 실습 노트북에서면 넘어가도 되지만, **재사용 모듈에 들어가면 이름과 동작이 안 맞는 함수가 계속 잘못 호출되는 근본 원인**이 됨.

3. **`my_qtcheck.py`의 `numerical_summary`가 `data[columns]`만 보고 명목형을 자동 제외하는 구조**
   - `columns` 인자를 안 넘기면 `get_number_column_names()`로 숫자형만 골라오니 문제 없는데, 만약 나중에 숫자로 인코딩된 범주형(예: 0/1로 인코딩된 CHAS 같은 걸 category로 안 바꾼 상태)이 섞여 들어오면 조용히 이상치/왜도 계산 대상에 포함됨. → **함수 자체는 안전한데, "쓰는 사람이 ①단계(자료형 변환)를 먼저 했는지"에 전적으로 의존**하는 구조. 함수 docstring에 "반드시 set_type으로 범주형 처리 후 사용" 같은 전제조건을 한 줄 명시해두면 나중에 실수 줄일 수 있음.

4. **`categorical_summary`의 `save_path` 저장 로직 — 매 컬럼마다 파일을 열고 닫음**
   ```python
   for col in columns:
       ...
       with ExcelWriter(save_path, mode='a', engine='openpyxl') as excel_writer:
           cdf.to_excel(excel_writer, sheet_name=col, index=True)
   ```
   컬럼 개수만큼 파일 open/close가 반복됨. 지금 규모(컬럼 몇 개)에선 문제없지만, 나중에 컬럼이 수십 개인 테이블(Olist처럼 넓은 테이블)에 쓰면 느려질 수 있음. 아직 급한 문제는 아니고, "나중에 느리다 싶으면 여기부터 의심"으로 메모만 해두면 됨.

---

## 6. 문서화 습관 — docstring은 이미 잘 하고 있음, 유지할 것

두 모듈 다 Args/Returns를 꼬박꼬박 적어놓은 게 좋은 습관임. 앞으로 함수 추가할 때도 동일 포맷 유지:
```python
def 함수명(파라미터...):
    """
    한 줄 요약

    Args:
        파라미터명 (타입): 설명

    Returns:
        타입: 설명
    """
```
추가로 고려하면 좋을 것 — **"주의사항" 섹션을 docstring에 넣기.** 예를 들어 `numerical_summary`에는:
```python
"""
...
주의:
    - 호출 전 반드시 set_type()으로 범주형 컬럼을 category로 변환해둘 것
    - 위경도처럼 숫자형이지만 통계적으로 의미 없는 컬럼은 columns 인자로 제외 권장
"""
```
이런 식으로 "함수가 뭘 가정하고 있는지"까지 적어두면, 몇 달 뒤 다시 볼 때 훨씬 빨리 이해됨.

---

## 7. 앞으로 확장할 때 파일 분리 기준

지금은 `my_qtcheck.py` 하나에 품질검사+통계가 다 들어있는데, Test/Model 파트(가설검정, 회귀진단, VIF 등)를 추가하게 되면:

```
helpers/
├── my_qtcheck.py    ← 지금 그대로 (품질검사 + 기술통계)
├── my_plot.py        ← 지금 그대로 (시각화)
└── my_stat.py         ← 신규: 가설검정(t-test/ANOVA/chi-square), 회귀진단(VIF/Q-Q/Durbin-Watson) 등
```
한 파일에 계속 몰아넣지 말고 **역할 단위로 파일을 분리**하는 게 나중에 특정 함수 찾기도 쉽고, `import`할 때도 필요한 것만 골라 쓸 수 있어서 좋음.
```python
from helpers import my_qtcheck, my_plot, my_stat
```

---

## 8. 체크리스트 — 새 함수를 모듈에 추가하기 전에

- [ ] 최소 2개 이상의 다른 데이터셋에서 검증됐는가 (한 데이터셋 전용 로직 아닌지)
- [ ] `data.copy()`로 원본 보존 패턴을 따르는가
- [ ] docstring에 Args/Returns/주의사항이 다 있는가
- [ ] 함수명과 실제 동작이 일치하는가 (예: countplot인데 barplot 호출하는 식의 불일치 없는지)
- [ ] 이 함수가 다른 함수(예: `set_type`)의 사전 처리에 의존한다면 그 전제조건이 명시돼 있는가
- [ ] 커밋 메시지에 어떤 함수를 왜 추가/수정했는지 남겼는가

---

## 한 줄 요약

`my_qtcheck.py`, `my_plot.py`는 이제 "실습 파일"이 아니라 **계속 자라나는 개인 라이브러리**. 관리 원칙은 세 가지: **(1) 폴더 구조·import 경로를 고정해서 어디서든 불러쓸 수 있게, (2) 노트북에서 검증된 로직만 골라서 모듈에 편입, (3) 함수 추가/수정은 항상 별도 커밋 + docstring 갱신.** 이 세 가지만 지키면 Olist든 다음 프로젝트든 매번 새로 짜는 게 아니라 계속 불려서 쓰는 구조가 됨.
