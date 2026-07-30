import numpy as np
import seaborn as sb
from pandas import DataFrame
from statsmodels.api import add_constant, Logit 

from . import my_plot 
import helpers.my_logit as my_logit

def fit_model(data, y, summary=False):
    """
    statsmodels 의 Logit을 이용해 이항 로지스틱 회귀 모델을 적합한다.

    종속변수 'y'를 제외한 나머지 모든 컬럼을 독립변수로 사용하며, 
    절편(상수항)을 자동으로 추가한 뒤 최대우도추정(MLE)으로 회귀계수를 추정한다.
    종속변수는 0/1 두 값만 가지는 이분형이어야 한다.

    Args:
        data : 독립변수와 종속변수를 모두 포함하는 데이터 프레임 
        y : 종속변수로 사용할 컬럼명. 'data'에 반드시 존재해야 하며 0/1의 이분형이어야 한다.
        summary : True로 설정하면 적합된 모델의 요약 통계량을 출력한다.
                Defaults fo False 
    
    Returns:
        적합이 완료된 로지스틱 회귀분석 결과 객체 
    """

    x = data.drop(columns=[y]) #독립변수 데이터 프레임 생성 
    y_series = data[y] # 종속변수 시리즈 생성 
    x_input = add_constant(x) # 독립변수에 절편(상수항) 추가 
    model = Logit(y_series, x_input) # Logit 모델 객체 생성 
    fit = model.fit(disp=0) # 모델 적합. disp=0 -> 수렴 메시지 출력 안함 

    if summary: 
        print(fit.summary()) # 적합된 모델의 요약 통계량 출력 여부 확인

    return fit # 적합된 모델 객체(분석 결과) 반환

def predict(fit, new_data, threshold=0.5):
    """
    적합된 로지스틱 모델로 새로운 데이터의 예측 확률과 예측 범주를 계산한다.

    로지스틱 회귀의 예측값은 '1'(사건 발생)일 확률이므로, 임계값(threshold)을 초과하면 1, 
    그렇지 않으면 0으로 분류한다.

    Args: 
        fit : 'fit_model' 함수로 적합된 회귀분석 결과 객체 
        new_data : 예측에 사용할 새로운 데이터 프레임. 독립변수 컬럼만 포함해야 한다.
        threshold (float) : 확률을 0/1로 분류하는 임계값( 기본값: 0.5 )

    Returns : 
        DataFrame : 예측 확률('proba')과 예측값('pred')을 담은 데이터 프레임 
    """

    # 새로운 데이터에 절편(상수항) 추가 
    new_data_with_const = add_constant(new_data)

    # 사건 발생(=1) 확률 예측 
    proba = fit.predict(new_data_with_const)

    # 예측 확률과 임계값 기준 예측 범주를 DataFrame으로 반환 
    return DataFrame({
        "proba" : proba,    # 1이 될 확률 
        "pred" : (proba > threshold).astype(int)  # 예측값
    })

def plot_sigmoid(fit, data, x, threshold=0.5, palette=None, title=None, 
                 xlabel=None, ylabel=None, width=1280, height=640, 
                 save_path=None):
    """
    독립변수에 따른 사건 발생 확률의 S자 곡선 ( 시그모이드)을 그린다.

    Args: 
        fit: 'fit_model' 함수로 적합된 회귀분석 결과 객체
        data: 회귀분석에 사용한 원본 데이터 프레임( 독립변수와 종속변수를 모두 포함 )
        x (str) : 곡선의 x축으로 사용할 독립변수명 
        threshold (float) : 확률을 0/1로 분류하는 임계값 (기본값 : 0.5)
        palette (str) : 그래프 색상에 사용할 팔레트 이름( 기본값 : None )
        title (str) : 그래프 제목 (기본값: None)
        xlabel (str) : x축 레이블 (기본값: None -> 독립변수명)
        ylabel (str) : y축 레이블 (기본값: None -> "P(종속변수=1)")
        width (int) : 캔버스 가로 픽셀 (기본값:1280)
        height (int) : 캔버스 세로 픽셀 (기본값:640)
        save_path (int) : 이미지 저장 경로(기본값: None)
    """
    # --- 1) 그릴 종속변수 결정 ---
    yname = fit.model.endog_names 

    # --- 2) 곡선을 그릴 x값 격자 생성 ---
    # 관측된 x의 최솟값~최댓값을 200등분해 촘촘한 곡선을 만든다.
    grid = np.linspace(data[x].min(), data[x].max(), 200)

    # 곡선 계산용 입력 데이터 
    curve_input = DataFrame({x: grid})

    # 예측에 사용할 수 있도록 상수항을 추가한 뒤 사건 발생(=1) 확률 계산 
    proba = fit.predict(add_constant(curve_input))

    # --- 3) 그래프 초기화 ---
    # 팔레트가 지정되었다면 첫번째 색상을 선 색상으로 사용하고, 
    # 지정되지 않았다면 기본 파랑색을 사용한다. 
    line_color = sb.color_palette(palette)[0] if palette else "#328CC1"

    # 그래프 초기화 및 축 레이블 설정 
    fig, ax = my_plot.init(width=width, height=height, title=title, 
                           xlabel=xlabel if xlabel else x, 
                           ylabel=ylabel if ylabel else f"P({yname}=1)")
    
    # --- 4) 실제 관측치(0/1) 산점도---
    # 같은 높이(0 또는 1)에 점이 겹쳐 보이므로 반투명하게 처리한다. 
    my_plot.scatterplot(data=data, x=x, y=yname, color="#888888", 
                        alpha=0.4, palette=None, ax=ax)
    
    # --- 5) 예측 확률의 S자 곡선 ---
    my_plot.lineplot(x=grid, y=proba, color=line_color, ax=ax)

    # --- 6) 임계값 가로선과 분류 경계 세로선---
    a = fit.params[x] # 독립변수 x에 대한 기울기 
    b = fit.params['const']  # 절편 
    boundary = -b/a # 분류 경계 = -절편/기울기 

    # 분류 경계에 대한 세로 선과 텍스트 
    ax.axvline(x=boundary, color="red", linestyle="--", alpha=0.7)
    ax.text(x=boundary, y=threshold, s=f"분류 경계: {boundary:.2f}", 
            color="red", va="bottom", ha="left")

    # 임계값에 대한 가로 선 
    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.7)

    # 확률은 0~1 범위이므로 여백을 조금 두고 축을 고정한다. 
    ax.set_ylim(-0.1, 1.1)

    # --- 7) 그래프 표시 ---
    my_plot.show(save_path=save_path)