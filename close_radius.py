import sympy.geometry as sg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def CrossPoint(center, radius, segment1, segment2):
    """
    円と線分の交点を算出する関数

    @param center 円の中心座標
    @param radius 円の半径
    @param segment1 線分の一点目座標
    @param segment2 線分の二点目座標

    """
    circle = sg.Circle(center, radius)
    segment = sg.Segment(segment1, segment2)
    return sg.intersection(circle, segment)

def PlotCrossPoint(center, radius, crosspoint):
    """
    交点と元の円を描画する関数

    @param center 円の中心座標
    @param radius 円の半径
    @param **crosspoints 交点(**kwargs)
    """
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)

    ax.add_patch(plt.Circle(xy=center, radius=radius, ec='darkblue',linestyle=':', fill=False, ))
    for i,crosspoint in enumerate(crosspoint):
        plt.plot(crosspoint.x, crosspoint.y, color='crimson',marker='.')
        plt.text(crosspoint.x, crosspoint.y+0.1, 'p{0}'.format(i),size=10,horizontalalignment='center',verticalalignment='bottom')
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    plt.show()
