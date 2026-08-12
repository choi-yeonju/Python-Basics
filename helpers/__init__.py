import os 
import glob as gl
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from pathlib import Path 

# ------------------------------------------------
# 전역 상수 
# ------------------------------------------------
# 무작위성이 개입하는 모든 기능(PCA, 군집, 데이터 분할 등)의 재현성을 위한 랜덤시드 
# 하위 모듈에서 'from . import RANDOM_STATE'로 참조하므로 모듈 임포트보다 먼저 정의한다. 
RANDOM_STATE = 3217

# -----------
# 내보낼 모듈 임포트 
# ----------- 
from . import my_qtcheck    # 데이터 품질 점검 관련 함수 모듈
from . import my_plot       # 시각화 관련 함수 모듈 
from . import my_stats      # 통계 분석 관련 함수 모듈
from . import my_prep       # 데이터 전처리 관련 함수 모듈
from . import my_ols        # 선형회귀 관련 함수 모듈
from . import my_logit      # 로지스틱 회귀 관련 함수 모듈 
from . import my_ts         # 시계열 분석 관련 함수 모듈
from . import my_cluster    # 군집분석 관련 함수 모듈 

# ------------------------
# 한글 폰트 설정 
# ------------------------
fpath = "../helpers/fonts" # 한글을 지원하는 폰트 파일의 경로 
font_files = gl.glob(os.path.join(fpath,"*.ttf")) # 폰트 파일 검색

for f in font_files: 
    fm.fontManager.addfont(f) # 폰트 등록
    fprop = fm.FontProperties(fname=f) #폰트의 속성을 읽어옴 
    fname = fprop.get_name()
    plt.rcParams['font.family'] = fname

#----------------------
# 그래프 기본 설정 
# ----------------------------
my_dpi = 200                                # 이미지 선명도(100~300)
plt.rcParams['font.size'] = 12              # 기본 폰트 이미지 
plt.rcParams['axes.unicode_minus'] = False  # 그래프에 마이너스 깨짐 방지 
plt.rcParams['figure.dpi'] = my_dpi         # 그래프의 dpi  설정
plt.rcParams['savefig.dpi'] = my_dpi        # 저장되는 그래프의 dpi 설정
plt.rcParams['lines.linewidth'] = 2         # 그래프 선 굵기 설정 
plt.rcParams['axes.axisbelow'] =True        # 그래프의 축과 격자선을 뒤에 배치 
