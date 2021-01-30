import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import random
from IPython.display import HTML
from matplotlib.animation import PillowWriter
from PIL import Image, ImageDraw
import xlrd
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy import signal
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN


# 目的関数
# def objective_function(position):
#       return np.sum(position * position)
fig = plt.figure(figsize=(10,6), constrained_layout=True)
gs = fig.add_gridspec(2,3, width_ratios=[1,1,2], height_ratios=[1,2])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
##ax3 = fig.add_subplot(gs[1,:2], projection='3d')
ax4 = fig.add_subplot(gs[1,:])

ims = []

# ファイルから各要素を読み込み
s = []

print('HOW MANY frequencies(range:1-4):')
num = int(input())

print('f1:')
f1 = input()
inputfiles = f1+'khz.xlsx'
list_s = pd.read_excel(inputfiles, sheet_name=0)
s = list_s.values
list_r = pd.read_excel(inputfiles, sheet_name=1)
R1 = list_r.values
list_l = pd.read_excel(inputfiles, sheet_name=2)
L1 = list_l.values

if num >= 2:
    print('f2:')
    f2 = input()
    inputfiles2 = f2+'khz.xlsx'
    list_rr = pd.read_excel(inputfiles2, sheet_name=1)
    R2 = list_rr.values
    list_ll = pd.read_excel(inputfiles2, sheet_name=2)
    L2 = list_ll.values

    if num >= 3:
        print('f3:')
        f3 = input()
        inputfiles3 = f3+'khz.xlsx'
        list_rrr = pd.read_excel(inputfiles3, sheet_name=1)
        R3 = list_rrr.values
        list_lll = pd.read_excel(inputfiles3, sheet_name=2)
        L3 = list_lll.values

        if num >= 4:
            print('f4:')
            f4 = input()
            inputfiles4 = f4+'khz.xlsx'
            list_rrrr = pd.read_excel(inputfiles4, sheet_name=1)
            R4 = list_rrrr.values
            list_llll = pd.read_excel(inputfiles4, sheet_name=2)
            L4 = list_llll.values

"""""""""""""""""""""""""""""""""""""""""""""""
    # besselfilter
"""""""""""""""""""""""""""""""""""""""""""""""
def lowpassfilter(y):
    """
    結論としては、A/D変換前のアナログ信号処理のような
    波形を重視する用途ではベッセルが適し、
    オーディオ信号のような
    スペクトルを重視する用途では
    バタワースが優れていることになります。

    https://www.orixrentec.jp/helpful_info/detail.html?id=43
    """
    n = 999
    dt = 0.5 # サンプリング間隔
    f = 500
    fn = 500/(2*dt) # ナイキスト周波数
    t = np.linspace(1,n,n)*dt-dt
    y = np.array(y)

    fp = 3
    fs = 13
    gpass = 2
    gstop = 50
    Wp = fp/fn
    Ws = fs/fn

    # ベッセルフィルタ
    N = 4
    b5, a5 = signal.bessel(N,Ws,"high")
    y1 = signal.filtfilt(b5,a5,y,axis=0)
    return(y1)


X1 = lowpassfilter(R1)
Y1 = lowpassfilter(L1)
if num >= 2:
    X2 = lowpassfilter(R2)
    Y2 = lowpassfilter(L2)
    if num >= 3:
        X3 = lowpassfilter(R3)
        Y3 = lowpassfilter(L3)
        if num >= 4:
            X4 = lowpassfilter(R4)
            Y4 = lowpassfilter(L4)

"""""""""""""""""""""""""""""""""""""""""""""""
  ↓目的関数に粒子位置を代入し計算値をスカラで返す↓
"""""""""""""""""""""""""""""""""""""""""""""""
def objective_function(positions):
    # MY FUNCTION
    # f = np.sum((s - (aX + bY))**2)
    # 空の配列を作成
    b1 = np.empty((999,1))
    c1 = np.empty((999,1))
    b2 = np.empty((999,1))
    c2 = np.empty((999,1))
    b3 = np.empty((999,1))
    c3 = np.empty((999,1))
    b4 = np.empty((999,1))
    c4 = np.empty((999,1))
    d = np.empty((999,1))

    t1 = s
    # positionを計算式に入れるため行方向に繰り返し
    posi_tile = np.tile(positions, (len(s), 1))
    # 行ごとに配列の要素どうしを掛け算
    for i in range(len(s)):
        b1[i, 0] = np.multiply(X1[i], posi_tile[i, 0])
        c1[i, 0] = np.multiply(Y1[i], posi_tile[i, 1])
        if num >= 2:
            b2[i, 0] = np.multiply(X2[i], posi_tile[i, 2])
            c2[i, 0] = np.multiply(Y2[i], posi_tile[i, 3])
            if num >= 3:
                b3[i, 0] = np.multiply(X3[i], posi_tile[i, 4])
                c3[i, 0] = np.multiply(Y3[i], posi_tile[i, 5])
                if num >= 4:
                    b4[i, 0] = np.multiply(X4[i], posi_tile[i, 6])
                    c4[i, 0] = np.multiply(Y4[i], posi_tile[i, 7])

        # d = (aX + bY)を計算
        d[i] = b1[i, 0] + c1[i, 0]
        if num >= 2:
            d[i] = d[i] + b2[i, 0] + c2[i, 0]
            if num >= 3:
                d[i] = d[i] + b3[i, 0] + c3[i, 0]
                if num >= 4:
                    d[i] = d[i] + b4[i, 0] + c4[i, 0]
    t2 = d
    # 空の配列を作成
    mse = np.empty((1, 1))
    # 二乗誤差を計算
    mse = mean_squared_error(t1, t2)
    return mse

# 各粒子の位置更新
def update_positions(positions, velocities):
    positions += velocities
    return positions


# 各粒子の速度更新
def update_velocities(positions, velocities, personal_best_positions, global_best_particle_position, w=0.5, ro_max=0.14):
    rc1 = random.uniform(0, ro_max)
    rc2 = random.uniform(0, ro_max)

    velocities = velocities * w + rc1 * (personal_best_positions - positions) + rc2 * (global_best_particle_position - positions)

    return velocities

# 最適解を適用した時の再現欠陥波形の作成.描画
def result_signal(global_best_particle_position):

    # global positionを計算式に入れるため行方向に繰り返し
    best_posi_tile = np.tile(global_best_particle_position, (len(s),1))

    b1_best = np.empty((999,1))
    c1_best = np.empty((999,1))
    b2_best = np.zeros((999,1))
    c2_best = np.zeros((999,1))
    b3_best = np.zeros((999,1))
    c3_best = np.zeros((999,1))
    b4_best = np.zeros((999,1))
    c4_best = np.zeros((999,1))
    best_signal = np.empty((999,1))

    for i in range(len(s)):
        b1_best[i, 0] = np.multiply(X1[i], best_posi_tile[i, 0])
        c1_best[i, 0] = np.multiply(Y1[i], best_posi_tile[i, 1])
        if num >= 2:
            b2_best[i, 0] = np.multiply(X2[i], best_posi_tile[i, 2])
            c2_best[i, 0] = np.multiply(Y2[i], best_posi_tile[i, 3])
            if num >= 3:
                b3_best[i, 0] = np.multiply(X3[i], best_posi_tile[i, 4])
                c3_best[i, 0] = np.multiply(Y3[i], best_posi_tile[i, 5])
                if num >= 4:
                    b4_best[i, 0] = np.multiply(X4[i], best_posi_tile[i, 6])
                    c4_best[i, 0] = np.multiply(Y4[i], best_posi_tile[i, 7])
        # d = (aX + bY)を計算
        best_signal[i] = b1_best[i, 0] + c1_best[i, 0] + b2_best[i, 0] + c2_best[i, 0] + b3_best[i, 0] + c3_best[i, 0] + b4_best[i, 0] + c4_best[i, 0]

    best_signal = np.array(best_signal)
    n = 999
    dt = 0.5 # サンプリング間隔
    t = np.linspace(1,n,n)*dt-dt
    ax4.plot(t, s, linestyle='dotted', label='s')
    ax4.plot(t, best_signal, linestyle='solid', label='best signal')
    ax4.legend()

    return best_signal

def main():
    print("Particles: ")
    number_of_particles = int(input())

    dimensions = int(num * 2)
    print("Dimensions: ", dimensions)

    print("LimitTimes: ")
    limit_times = int(input())

    xy_min, xy_max = -5, 5

    # グラフの初期化
    #axes, mesh_XYZ = init_plot(xy_min, xy_max)

    # 各粒子の位置
    positions = np.array([[random.uniform(xy_min, xy_max) for _ in range(dimensions)] for _ in range(number_of_particles)])

    # 各粒子の速度
    velocities = np.zeros(positions.shape)

    # 各粒子ごとのパーソナルベスト位置
    personal_best_positions = np.copy(positions)

    # 各粒子ごとのパーソナルベストの値
    personal_best_scores = np.apply_along_axis(objective_function, 1, personal_best_positions)

    # グローバルベストの粒子ID
    global_best_particle_id = np.argmin(personal_best_scores)

    # グローバルベスト位置
    global_best_particle_position = personal_best_positions[global_best_particle_id]


    # 規定回数
    for T in range(limit_times):

        # 速度更新
        velocities = update_velocities(positions, velocities, personal_best_positions, global_best_particle_position)

        # 位置更新
        positions = update_positions(positions, velocities)

        # パーソナルベストの更新
        for i in range(number_of_particles):
            score = objective_function(positions[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]

        # グローバルベストの更新
        global_best_particle_id = np.argmin(personal_best_scores)
        global_best_particle_position = personal_best_positions[global_best_particle_id]

    print("global_best_particle_position:", global_best_particle_position)
    print("global_best_scores:", objective_function(global_best_particle_position))
    ##Writer = animation.writers['pillow']
    ##writer = Writer(metadata=dict(artist='Me'), bitrate=1800)

    ##ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)

    n = 999
    dt = 0.5 # サンプリング間隔
    t = np.linspace(1,n,n)*dt-dt

    # 入力信号描画
    # ax1 X成分プロット
    ax1.plot(t,X1,label=f1+'khz')
    if num >= 2:
        ax1.plot(t,X2,label=f2+'khz')
        if num >= 3:
            ax1.plot(t,X3,label=f3+'khz')
            if num >= 4:
                ax1.plot(t,X4,label=f4+'khz')
    ax1.legend()
    ax1.set_ylabel('X')


    # ax2 Y成分プロット
    ax2.plot(t,Y1,label=f1+'khz')
    if num >= 2:
        ax2.plot(t,Y2,label=f2+'khz')
        if num >= 3:
            ax2.plot(t,Y3,label=f3+'khz')
            if num >= 4:
                ax2.plot(t,Y4,label=f4+'khz')
    ax2.legend()
    ax2.set_ylabel('Y')
    ax4.set_ylabel("result signal")
    np.set_printoptions(precision=4)
    plt.gcf().text(0.6, 0.95, 'the number of frequencies:'+str(num), size=10)
    if num == 1:
        poyo = f1+'khz'
    elif num == 2:
        poyo = f1+'khz+'+f2+'khz'
    elif num == 3:
        poyo = f1+'khz+'+f2+'khz+'+f3+'khz'
    elif num == 4:
        poyo = f1+'khz+'+f2+'khz+'+f3+'khz+'+f4+'khz'
    else:
        print('Error: the number of f is not correct')
    plt.gcf().text(0.6, 0.9, poyo, size=10)
    plt.gcf().text(0.6, 0.85, 'global best particle position:', size=10)
    plt.gcf().text(0.6, 0.8,  global_best_particle_position, size=10)
    plt.gcf().text(0.6, 0.75, 'min(objective_function):', size=10)
    plt.gcf().text(0.6, 0.7, objective_function(global_best_particle_position), size=10)

    ResultAction = result_signal(global_best_particle_position)

    print("image_name: ")
    imgname = input()
    fig.savefig(imgname+'.png')
    ##ani.save(imgname+'.gif', writer=writer)
    #ani = animation.ArtistAnimation(fig,ims,  blit=True)
    #ani.save('popopo.gif', writer="pillow", dpi=100)
    plt.show()

if __name__ == '__main__':
    main()
