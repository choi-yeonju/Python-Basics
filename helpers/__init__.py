import os 
import glob as gl
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt



# -----------
# 내보낼 모듈 임포트 
# ----------- 
from . import my_qtcheck 
from . import my_plot

# ------------------------
# 한글 폰트 설정 
# ------------------------
fpath = "./helpers/fonts" # 한글을 지원하는 폰트 파일의 경로 
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
