from . import RANDOM_STATE
from . import my_prep
from . import my_plot
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import seaborn as sb
import numpy as np
from kneed import KneeLocator

def kmeans(data, k, columns=None, scaling='standard', cluster_name='그룹번호',
           random_state=RANDOM_STATE, n_init=10, verbose=True, plot=True, x=None, y=None,
           title=None, palette='tab10', size=100, edgecolor='#ffffff', linewidth=1.5,
           alpha=1, outline=True, center=True, center_marker='X', center_size=150,
           center_color='#ff0000', center_edgecolor='#000000', center_linewidth=1.5,
           equal_scale=False, width=1280, height=640, save_path=None, ax=None):
    """데이터를 k개의 군집으로 나누고, 군집 결과와 중심점을 시각화하는 함수 

    K-Means는 데이터 사이의 '거리'로 그룹을 나누므로, 변수마다 값의 범위가 다르면
    범위가 큰 변수 하나가 거리를 지배하게 된다. 그래서 스케일링을 기본으로 수행한다.

    Args (기본값은 위의 함수 정의 참고):
        data, k: 군집화할 데이터프레임과 나눌 군집의 개수
        columns, cluster_name, random_state: 사용할 컬럼(None이면 수치형 전체),
            군집 번호를 저장할 컬럼명, 중심점의 초기 위치를 결정하는 랜덤시드
        scaling: 스케일러 이름('standard'/'minmax'/'robust'/'maxabs', None이면 원본 값)
        verbose: 스케일링 전후의 값의 범위를 출력할지 여부
        plot, x, y, title: 시각화 여부, 산점도의 x·y축 컬럼명(None이면 대상 컬럼의 앞 두 개),
            그래프 제목(None이면 군집 개수를 포함하여 자동 생성)
        palette, size, edgecolor, linewidth, alpha: 데이터 포인트의 색상 팔레트,
            마커 크기, 테두리 색상, 테두리 두께, 투명도
        outline: 군집의 외곽선(ConvexHull)을 표시할지 여부
        center: 모델이 찾은 중심점을 표시할지 여부
        center_marker, center_size, center_color, center_edgecolor, center_linewidth:
            중심점의 마커 모양, 크기, 색상, 테두리 색상, 테두리 두께
        equal_scale: x축의 범위를 y축과 동일하게 맞출지 여부
            (모델이 실제로 본 거리 관계를 그대로 확인할 때 사용한다)
        width, height, save_path, ax: 캔버스 가로·세로 픽셀, 저장 경로,
            그래프를 그릴 Axes 객체(None이면 새로 생성)

    Returns:
        tuple: (estimator, df, center_df) - 학습이 완료된 모델,
            군집 번호 컬럼이 추가된 데이터(스케일링 적용 후),
            각 군집의 중심점 좌표(컬럼명은 대상 컬럼과 동일)    
    
    """
    # --- 1) 군집화에 사용할 컬럼 결정 ---
    # 지정이 없으면 수치형 컬럼만 자동 선택 (문자열 컬럼은 거리 계산이 불가능하다)
    if columns is None :
        columns = list(data.select_dtypes(include='number').columns)

    # --- 2) 스케일링 적용 ---
    # 변수마다 값의 범위가 다르면 범위가 큰 변수 하나만 보고 그룹을 나눈 셈이 되므로 
    # 거리 계산 전에 두 변수의 눈금을 같은 기준으로 맞춘다 
    if scaling:
        df = my_prep.scaling(data[columns], method=scaling, verbose=verbose)
    else:
        df = data[columns].copy()

    # --- 3) 모델 생성 및 학습 (중심점을 찾는 과정) ---
    estimator = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    estimator.fit(df)

    # --- 4) 각 데이터가 몇 번 그룹인지 예측하여 컬럼으로 추가 ---
    df[cluster_name] = estimator.predict(df)

    # --- 5) 모델이 찾은 중심점을 데이터프레임으로 구성 ---
    center_df = DataFrame(estimator.cluster_centers_, columns=columns)

    # --- 6) 군집 결과 시각화 ---
    if plot:
        # 축으로 사용할 컬럼 결정 (지정이 없으면 대상 컬럼의 앞에서 두 개)
        if x is None:      x = columns[0]
        if y is None:      y = columns[1]

        # 제목을 지정하지 않은 경우 군집 개수를 포함한 제목을 자동으로 생성
        if title is None:  title = f'K-Means 군집 결과 (k={k})'

        # 그래프 초기화 (ax를 전달받은 경우에는 그 위에 겹쳐 그린다)
        fig = None
        if ax is None:
            fig, ax = my_plot.init(width=width, height=height, title=title,
                                    xlabel=x, ylabel=y)

        # 군집별 산점도 (outline=True이면 각 군집의 외곽선까지 표시)
        my_plot.scatterplot(data=df, x=x, y=y, hue=cluster_name,
                            palette=palette, size=size, edgecolor=edgecolor,
                            linewidth=linewidth, alpha=alpha, outline=outline, ax=ax)

        # 모델이 찾은 중심점을 덧그리기 (중심점의 x좌표에 대한 y좌표 산점도)
        if center:
            my_plot.scatterplot(data=center_df, x=x, y=y,
                                marker=center_marker, size=center_size,
                                color=center_color, edgecolor=center_edgecolor,
                                linewidth=center_linewidth, outline=False, ax=ax)

        # x축의 범위를 y축과 동일하게 맞춰 실제 거리 관계를 확인
        if equal_scale:
            ax.set_xlim(ax.get_ylim())

        # 그래프 표시 (ax를 전달받은 경우에는 호출한 쪽에서 표시한다)
        if fig is not None:
            my_plot.show(save_path=save_path)

    # --- 7) 모델, 군집 결과, 중심점 좌표 반환 ---
    return estimator, df, center_df

# ===================================================================
# 실루엣 시각화 — 군집이 잘 나뉘었는지를 데이터 하나하나 단위로 확인한다
# 원본: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# ===================================================================
def visualize_silhouette(estimator, data, columns=None, cluster_name='그룹번호',
                         x=None, y=None, title=None, palette='tab10',
                         size=50, edgecolor='#ffffff', linewidth=1.5, alpha=1, outline=True,
                         center=True, center_marker='X', center_size=150,
                         center_color='#ff0000', center_edgecolor='#000000', center_linewidth=1.5,
                         width=800, height=580, save_path=None):
    """군집별 실루엣 계수 막대(왼쪽)와 군집 산점도(오른쪽)를 나란히 그리는 함수

    실루엣 계수는 "내가 속한 군집의 이웃들과는 얼마나 가깝고, 가장 가까운 다른 군집과는
    얼마나 먼가"를 데이터 하나마다 -1~1 사이의 값으로 나타낸 것이다. 평균값 하나만 보면
    "군집 하나가 통째로 망가진 경우"와 "모든 군집이 고만고만한 경우"를 구분할 수 없으므로,
    군집별 분포를 막대로 펼쳐서 함께 확인한다.

    Args (기본값은 위의 함수 정의 참고):
        estimator: 학습이 끝난 군집 모델 (KMeans, AgglomerativeClustering 등)
        data (DataFrame): 군집에 사용한 데이터
            (kmeans 함수의 반환값처럼 군집 번호 컬럼이 포함되어 있어도 된다)
        columns, cluster_name: 실루엣 계산에 사용할 컬럼(None이면 군집 번호 컬럼을 제외한
            수치형 전체), 데이터에 포함된 군집 번호 컬럼의 이름
        x, y, title: 산점도의 x·y축 컬럼명(None이면 대상 컬럼의 앞 두 개),
            그래프 상단 제목(None이면 군집 개수와 실루엣 스코어로 자동 생성)
        palette, size, edgecolor, linewidth, alpha, outline: 산점도의 색상 팔레트,
            마커 크기, 테두리 색상, 테두리 두께, 투명도, 외곽선 표시 여부
            (왼쪽 막대의 색상도 같은 팔레트를 사용하므로 두 그래프의 군집 색이 일치한다)
        center, center_marker, center_size, center_color, center_edgecolor, center_linewidth:
            중심점 표시 여부와 마커 모양·크기·색상·테두리 색상·테두리 두께
            (중심점을 제공하는 모델에서만 표시된다)
        width, height, save_path: 그래프 한 칸의 가로·세로 픽셀, 저장 경로
    """
    labels = estimator.labels_

    # --- 1) 실루엣 계산에 사용할 컬럼 결정 ---
    # 군집 번호 컬럼이 섞여 있으면 그 자체가 거리 계산에 반영되므로 반드시 제외한다
    if columns is None:
        columns = [c for c in data.select_dtypes(include='number').columns if c != cluster_name]

    df = data[columns]

    # --- 2) 노이즈(-1)를 제외한 실제 군집 번호 확인 ---
    # 실루엣 계수는 "다른 군집과의 거리"가 있어야 정의되므로 군집이 2개는 되어야 한다
    cluster_ids = sorted([c for c in set(labels) if c != -1])
    k = len(cluster_ids)

    if k < 2:
        print("군집이 2개 미만이므로 실루엣 계수를 계산할 수 없습니다.")
        return

    # --- 3) 전체 평균과 데이터별 실루엣 계수 계산 ---
    sil_avg = silhouette_score(X=df, labels=labels)
    sil_values = silhouette_samples(X=df, labels=labels)

    if title is None:
        title = f'군집수={k}, 실루엣 스코어={sil_avg:.3f}'

    fig, (ax1, ax2) = my_plot.init(width=width, height=height, rows=1, cols=2, title=title)

    # 전체 제목이 각 그래프의 소제목과 겹치지 않도록 위쪽으로 밀어낸다
    fig.suptitle(title, fontsize=24, fontweight=500, y=1.0)

    # --- 4) 왼쪽: 군집별 실루엣 계수 막대 ---
    # 오른쪽 산점도와 같은 팔레트를 사용해 두 그래프의 군집 색을 일치시킨다
    colors = sb.color_palette(palette, k)

    ax1.set_title('군집별 실루엣 계수', fontsize=16, pad=10)
    ax1.set_xlabel('실루엣 계수')
    ax1.set_ylabel('군집 번호')
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(df) + (k + 1) * 10])
    ax1.set_yticks([])

    y_lower = 10

    for i, c in enumerate(cluster_ids):
        # 해당 군집에 속한 데이터의 실루엣 계수를 오름차순으로 정렬 (칼날 모양의 막대가 된다)
        ith_values = np.sort(sil_values[labels == c])
        y_upper = y_lower + len(ith_values)

        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_values,
                          facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * len(ith_values), str(c))

        # 다음 군집의 막대를 10칸 띄워서 그리기 위한 시작 위치
        y_lower = y_upper + 10

    # 전체 평균 실루엣 계수를 빨간 점선으로 표시 (이 선을 넘지 못하는 군집이 문제 군집)
    ax1.axvline(x=sil_avg, color='red', linestyle='--')

    # --- 5) 오른쪽: 군집 산점도 ---
    if x is None:       x = columns[0]
    if y is None:       y = columns[1]

    vdf = df.copy()
    vdf[cluster_name] = labels

    my_plot.scatterplot(data=vdf, x=x, y=y, hue=cluster_name, palette=palette,
                        size=size, edgecolor=edgecolor, linewidth=linewidth,
                        alpha=alpha, outline=outline, ax=ax2)

    ax2.set_title('군집 산점도', fontsize=16, pad=10)
    ax2.set_xlabel(x)
    ax2.set_ylabel(y)

    # 중심점을 제공하는 모델(KMeans 등)이라면 중심점도 함께 표시
    centers = getattr(estimator, 'cluster_centers_', None)

    if center and centers is not None:
        center_df = DataFrame(centers, columns=columns)
        my_plot.scatterplot(data=center_df, x=x, y=y,
                            marker=center_marker, size=center_size,
                            color=center_color, edgecolor=center_edgecolor,
                            linewidth=center_linewidth, outline=False, ax=ax2)

    # 범례를 그래프 바깥 오른쪽 위에 고정한다
    # (자동 배치에 맡기면 데이터가 없는 자리를 찾아다니느라 K마다 위치가 달라진다)
    handles, texts = ax2.get_legend_handles_labels()

    if handles:
        ax2.legend(handles, texts, title=cluster_name,
                   loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # --- 6) 그래프 표시 ---
    # 전체 제목이 들어갈 위쪽 공간(7%)을 미리 비워 둔 뒤 표시한다
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    my_plot.show(save_path=save_path)

def _prepare_k_search(data, klist, columns, scaling, verbose): 
    """최적 k 탐색 함수들이 공통으로 수행하는 컬럼 선택, 스케일링, k 목록 정리
    
    여러 k를 같은 데이터로 비교해야 하므로 스케일링은 반복문 밖에서 한 번만 수행한다.

    Args: 
        data, klist: 군집화할 데이터, 확인할 k목록(None이면 2~10)
        columns, scaling, verbose: 사용할 컬럼(None이면 수치형 전체), 
            스케일러 이름(None이면 원본 값), 스케일링 전후의 값의 범위 출력 여부
    
    Returns:
        tuple: (df, klist) - 스케일링이 끝난 데이터, 2 이상만 남긴 k 목록

    """
    # 대상 컬럼이 없다면 숫자형태의 컬럼만 추려낸다.
    # --> 거리기반 알고리즘이므로 문자열 형태는 처리하지 못한다.
    if columns is None:
        columns = list(data.select_dtypes(include='number').columns)

    # 스케일링을 k마다 반복하면 매번 같은 계산을 다시 하는 셈이므로 여기서 한 번만 처리한다 
    if scaling:
        df = my_prep.scaling(data[columns], method=scaling, verbose=verbose)
    else:
        df = data[columns].copy()

    # 실루엣 계수는 "다른 군집과의 거리"가 있어야 정의되므로 k는 2부터 확인한다
    klist = list(range(2, 11)) if klist is None else [k for k in klist if k >=2]    

    # 스케일링 후의 데이터와 2 이상만 남긴 k 목록을 반환한다 
    return df, klist 

def best_k_elbow(data, klist=None, columns=None, scaling='standard',
                  sensitivity=0.01, random_state=RANDOM_STATE, n_init=10, verbose=True,
                  plot=True, title=None, color='#1f77b4', marker='o', linestyle=':',
                  best_color='#ff0000', width=1280, height=640, save_path=None, ax=None):
    """이너셔의 감소폭이 꺾이는 지점(엘보우 포인트)을 찾아 최적의 k를 추정하는 함수 

    이너셔(군집 중심까지 거리의 제곱합)는 k가 커지면 항상 줄어들기 때문에 가장 작은 값을 고르는 것은 의미가 없다. 
    대신 "k를 하나 더 늘려도 이제 별 이득이 없어지는" 지점을 찾는다. 
    꺾이는 지점은 KneeLocator가 계산하며, 이너셔는 뭉침만 보는 지표이므로 
    이 결과는 최종 답이 아니라 후보다. 

    Args (기본값은 위의 함수 정의 참고):
        data, klist: 군집화할 데이터, 확인할 k 목록(None이면 2~10)
        columns, scaling, random_state: 사용할 컬럼(None이면 수치형 전체),
            스케일러 이름(None이면 원본 값), 중심점의 초기 위치를 결정하는 랜덤시드 
        n_init: 시작 위치를 바꿔가며 시도할 횟수(k끼리 공정하게 비교하려면 1회로는 부족하다)
        sensitivity: KneeLocator의 민감도(S). 작을수록 작은 꺾임에도 반응한다 
        verbose, plot, title: 계산 결과 출력 여부, 시각화 여부 
                                그래프 제목(None이면 자동 생성)
        color, marker, linestyle, best_color: 이너셔 선의 색상, 마커 모양, 선스타일, 
                                              엘보우 포인트를 표시할 세로선의 색상
        width, height, save_path, ax: 캔버스 가로 세로 픽셀, 저장 경로, 
                                    그래프를 그릴 Axes 객체(None이면 새로 생성)
    
    Returns: 
        tuple: (best_k, result_df) - 엘보우 포인트, k별 이너셔와 감소량이 담긴 데이터 프레임 
    """
    # --- 0) 공통 준비 작업 ---
    df, klist = _prepare_k_search(data, klist, columns, scaling, verbose)

    # --- 1) k를 늘려 가며 이너셔 수집 ---
    inertia = []

    for k in klist: 
        estimator = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        estimator.fit(df)
        inertia.append(estimator.inertia_)

    # --- 2) k가 1 늘어날 때마다 이너셔가 얼마나 줄었는지 계산 ---
    # np.diff()는 바로 앞 값과의 차이를 구해준다 (이너셔는 줄어들므로 음수 -> 절댓값 변환)
    drop = np.abs(np.diff(inertia))

    # --- 3) 결과 정리 ---
    # 감소량은 "앞의 k에서 넘어올 때"의 값이므로 첫 행은 비워 둔다 
    result_df = DataFrame({
        'k': klist,
        '이너셔': np.round(inertia, 4), 
        '감소량': np.round(np.insert(drop, 0,np.nan), 4)
    })

    # --- 4) 엘보우 포인트 찾기 ---
    # convex(아래로 볼록) + decreasing(우하향)은 이너셔 곡선의 모양이다
    kl = KneeLocator(klist, inertia, curve='convex', 
                    direction='decreasing', S=sensitivity)
    best_k = kl.elbow 

    # 곡선이 거의 직선이면 꺾이는 지점이 없어 None이 나온다 
    if best_k is None:
        print("곡선이 완만해 꺾이는 지점을 찾지 못했습니다"
                "sensitivity를 낮추거나 klist의 범위를 넓혀 다시 시도해 보세요")
        return None, result_df 

    if verbose: 
        print(f"[엘보우] 최적의 k = {best_k}")
    # --- 5) 시각화 --- 
    if plot: 
        if title is None: title = '엘보우 포인트'

        fig = None
        if ax is None:
            fig, ax = my_plot.init(width=width, height=height, title=title, 
                                   xlabel='군집 개수(k)', ylabel='이너셔')  

        my_plot.lineplot(x=klist, y=inertia, color=color, marker=marker, 
                         linestyle=linestyle, ax=ax)    

        # 엘보우 포인트를 세로선으로 표시 
        ax.axvline(x=best_k, color=best_color, linestyle=':')
        ax.text(best_k + 0.1, max(inertia)*0.9, 'Best k = {best_k}', color=best_color)

        if fig is not None:
            my_plot.show(save_path=save_path)

    return best_k, result_df 

def best_k_silhouette(data, klist=None, columns=None, scaling='standard', 
                      random_state=RANDOM_STATE, n_init=10, verbose=True, plot=True,
                      plot_each=False, title=None, palette='tab10', 
                      width=800, height=580, save_path=None):
    """실루엣 막대의 면적과 두께를 함께 평가해 최적의 k를 찾는 함수 

    Args (기본값은 위의 함수 정의 참고):
        data, klist: 군집화할 데이터, 확인할 k 목록( None이면 2~10 )
        columns, scaling, random_state: 사용할 컬럼(None이면 수치형 전체),
                            스케일러 이름(None이면 원본 값), 중심점의 초기 위치를 결정하는 랜덤시드 
        n_init: 시작 위치를 바꿔 가며 시도할 횟수 (k끼리 공정하게 비교하려면 1회로는 부족하다)
        verbose, plot, plot_each: 계산 결과 출력 여부,
            선택된 k의 실루엣 막대, 산점도(visualize_shilhousette)를 그릴지 여부, 
            선택된 k뿐 아니라 모든 k에 대해 같은 그림을 그릴지 여부 
        title, palette: 그래프 제목(None이면 군집 개수와 스코어로 자동 생성), 
                산점도와 막대에 함께 쓰이는 색상 팔레트 
        width, heigth, save_path: 그래프 한 칸의 가로 세로 픽셀, 저장 경로
    
    Returns: 
        tuple: (best_k, result_df) - 균형지수가 가장 높은 k, 
            k별 스코어, 전체면적, 두께비, 최소상대면적, 균형지수가 담긴 데이터 프레임 
    """
    # -- 0) 공통 준비 작업 --
    df, klist = _prepare_k_search(data, klist, columns, scaling, verbose)

    # -- 1) k를 늘려 가며 실루엣 지표 수집 --
    items = [] 

    # 아래에서 선택된 k의 그림을 그릴 때 다시 학습시키지 않도록 모델을 보관해둔다 
    estimators = {}

    for k in klist:
        # 1-1) 모델 학습 
        estimator = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        estimator.fit(df)
        estimators[k] = estimator
        cluster = estimator.labels_

        # 1-2) 실루엣 스코어 계산 (군집 전체의 평균)
        # 데이터 하나하나의 실루엣 계수 (= 띠 하나의 가로 길이)
        s_samples = silhouette_samples(X=df, labels=cluster)

        # 1-3) 군집별 면적과 두께 계산 
        # 군집별 면적(계수의 합)과 두께(데이터 개수)
        area = np.array([s_samples[cluster == c].sum() for c in range(k)])
        size = np.array([(cluster == c).sum() for c in range(k)])

        score = s_samples.mean() # 계수의 평균 = 실루엣 스코어 
        total_area = area.sum() # 계수의 합 = 데이터 개수 x 스코어 
        thickness = size.max() / size.min() 

        # 면적의 합이 0 이하라면 군집이 사실상 무너진 상태이므로 균형을 0으로 처리한다 
        min_area = (area.min() / total_area) * k if total_area > 0 else 0.0 

        # 1-4) 균형지수 계산 (스코어 x 최소대상면적 / 두께비)
        items.append({
            'k': k, 
            '스코어': round(score, 4), 
            '전체면적': round(total_area, 4),
            '두께비': round(thickness, 4),
            '최소상대면적': round(min_area, 4),
            '균형지수': round(score * min_area / thickness, 4), 
        })

        # 1-5) k마다 실루엣 막대와 산점도를 확인 (눈으로 면적 두께를 함께 보기 위한 옵션) 
        # k마다 실루엣 막대와 산점도를 확인 ( 눈으로 면적 두께를 함께 보기 위한 옵션 )
        if plot_each:
            visualize_silhouette(estimator, df, palette=palette, 
                                 width=width, height=height)
            
    # -- 2) 균형지수가 가장 높은 k 선택 -- 
    result_df = DataFrame(items)
    best = result_df.loc[result_df['균형지수'].idxmax()] 
    best_k = int(best['k'])

    if verbose: 
        print(f"[실루엣] 균형지수 최대 -> 최적의 k = {best_k}"
              f"(스코어 {best['스코어']}, 두께비 {best['두께비']}, 균형지수 {best['균형지수']})")   
        
    # -- 3) 시각화 -- 
    # 선택된 k가 왜 뽑혔는지 지표 숫자보다 막대의 면적과 두께를 직접 보는 편이 빠르다 
    # (plot_each로 이미 모든 k를 그렸다면 같은 그림을 두 번 그리지 않는다)
    if plot and not plot_each:
        visualize_silhouette(estimators[best_k], df, title=title, palette=palette,
                                 width=width, height=height, save_path=save_path)

    return best_k, result_df

def best_k(data, klist=None, columns=None, scaling='standard', 
           sensitivity=0.01, random_state=RANDOM_STATE, n_init=10, verbose=True, 
           plot=True, plot_each=False, title=None, width=1280, height=640):
    """엘보우 포인트와 실루엣 균형지수를 함께 확인해 최종 k를 결정하는 함수 

    Args (기본값은 위의 함수 정의 참고):
            data, klist: 군집화할 데이터, 확인할 k 목록(None이면 2~10)
            columns, scaling, random_state: 사용할 컬럼(None이면 수치형 전체),
                스케일러 이름(None이면 원본 값), 중심점의 초기 위치를 결정하는 랜덤시드
            n_init: 시작 위치를 바꿔 가며 시도할 횟수 (k끼리 공정하게 비교하려면 1회로는 부족하다)
            sensitivity: KneeLocator의 민감도(S). 작을수록 작은 꺾임에도 반응한다
            verbose, plot, plot_each, title: 판단 과정 출력 여부, 시각화 여부,
                k마다 실루엣 막대·산점도를 그릴지 여부, 그래프 제목(None이면 자동 생성)
            width, height: 캔버스 한 칸의 가로·세로 픽셀

        Returns:
            tuple: (best_k, result_df) - 최종 선택한 k,
                k별 이너셔·감소량·실루엣 지표를 한 표로 합친 데이터프레임
        """
    # --- 1) 스케일링을 한 번만 수행한 뒤 두 함수에 같은 데이터를 남긴다 ---
    df, klist = _prepare_k_search(data, klist, columns, scaling, verbose)

    # --- 2) 두 지표를 각각 계산 ---
    elbow_k, elbow_df = best_k_elbow(df, klist=klist, scaling=None,
                                    sensitivity=sensitivity,
                                    random_state=random_state,
                                    n_init=n_init, verbose=False, plot=plot,
                                    width=width, height=height)

    sil_k, sil_df = best_k_silhouette(df, klist=klist, scaling=None, 
                                      random_state=random_state, n_init=n_init, 
                                      verbose=False, plot=plot, plot_each=plot_each, 
                                      width=width, height=height)    
    
    # --- 3) 두 결과를 한 표로 합치기 ---
    result_df = elbow_df.merge(sil_df, on='k')

    # --- 4) 최종 판단 ---
    final_k = sil_k # 엘보우는 뭉침만 보므로 후보 제시용, 최종 결정은 분리와 균형까지 보는 실루엣이 맡는다 

    if verbose: 
        # 두 지표가 가리키는 k와 지표값을 함께 출력해 비교한다 
        sil_row = sil_df[sil_df['k'] == sil_k].iloc[0]  
        elbow_row = sil_df[sil_df['k'] == elbow_k]

        print(f"[1단계] 엘보우 포인트: k = {elbow_k} (뭉침만 확인 -> 후보)")
        print(f"[2단계] 실루엣 균형지수: k = {sil_k} (뭉침 + 분리 + 크기 균형)")
        print('-' * 60)

        # 두 지표가 같은 k를 가리키는지 여부에 따라 최종 선택을 안내한다 
        if elbow_k == sil_k : 
            print("두 지표가 같은 k를 가리키므로 그대로 확정합니다")
        else:
            print("두 지표가 다른 k를 가리킵니다")

            if not elbow_row.empty:
                print(f" . k = {elbow_k} : 균형지수 {elbow_row.iloc[0]['균형지수']}"
                      f"(스코어 {elbow_row.iloc[0]['스코어']}, 두께비 {elbow_row.iloc[0]['두께비']})")   

            print(f" . k = {sil_k} : 균형지수 {sil_row['균형지수']}"
                  f"(스코어 {sil_row['스코어']}, 두께비 {sil_row['두께비']})")

        print('-' * 60)
        print(f" >>> 최종선택 : k = {final_k}")
        print("    (지표는 후보를 좁혀줄 뿐이므로, 목적과 도메인 지식으로 한번 더 확인할 것)")

    return final_k, result_df 



