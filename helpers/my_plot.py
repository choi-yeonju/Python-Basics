from matplotlib import pyplot as plt 

# ----------------------------------
def init(width=1280, height=640, rows=1, cols=1, title=None, xlabel=None, ylabel=None, grid=True):
    """
    그래프의 크기와 dpi를 설정하여 fig와 ax 객체를 반환하는 함수 

    Parameters: 
        - width : 그래프의 가로 크기 ( 픽셀 단위)
        - height : 그래프의 세로 크기 ( 픽셀 단위) 
        - rows : 그래프의 행 수 
        - cols : 그래프의 열 수 
        - title : 그래프의 제목 ( 기본값 : None) 
        - xlabel : x축 레이블 ( 기본값 : None) 
        - ylabel : y축 레이블 (기본값 : None) 
        - grid : 그래프에 그리드를 표시할지 여부 (기본값 : True)

    Returns : 
        - fig : 생성된 Figure 객체 
        - ax : 생성된 Axes 객체 또는 Axes 배열 
    """
    my_figsize = (width / 100, height/100) 
    fig, ax = plt.subplots(rows, cols, figsize=my_figsize)
    ax.grid(grid, alpha=0.5) 

    if title: 
        ax.set_title(title, fontsize=24, fontweight=500, pad=15)
    if xlabel: 
        ax.set_xlabel(xlabel, fontsize=16, fontweight=400, labelpad=5)
    if ylabel: 
        ax.set_ylabel(ylabel, fontsize=16, fontweight=400, labelpad=5)
    return fig, ax

def show(save_path=None): 
    """
    그래프를 화면에 표시하는 함수 

    Parameters : 
        - grig : 그리드를 표시할지 여부 (기본값 = True) 
        - save_path : 그래프를 저장할 파일 경로, None이면 저장하지 않음. 
    """       
    if save_path: 
        plt.savefig(save_path)

    plt.tight_layout()
    plt.show()
    plt.close()