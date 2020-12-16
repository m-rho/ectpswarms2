import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
from close_radius import CrossPoint
from close_radius import PlotCrossPoint

class Particle(object):
    """粒子群最適化法の粒子を表すクラス"""

    def __init__(self, f, position, velocity, rs=None, close_rs=None):
        """
        コンストラクタ
        @param f 最適化を行う関数
        @param position 粒子の初期位置
        @param velocity 粒子の初期速度
        @param rs 粒子が移動可能な最大半径
        @param close_rs 粒子が移動可能な位置最大半径

        """
        self.__f = f
        self.__my_best_position = position
        self.__my_best_score = f(position)
        self.__my_position = position
        self.__my_velocity = velocity
        #self.__maxs = maxs
        #self.__mins = mins
        self.__rs = rs
        self.__close_rs = close_rs
        self.__WEIGHT_V = 0.8
        self.__WEIGHT_ME = 1.0
        self.__WEIGHT_US = 1.0

    @property
    def score(self):
        return self.__my_best_score

    @property
    def position(self):
        return np.array(self.__my_best_position)

    @property
    def close(self):
        return self.__close_rs

    def move(self, best_position,position_i=None):
        """
        粒子を移動させるメソッド
        @param best_position 関数の最小値を与える粒子の位置
        """
        position_i = self.__my_position
        # 粒子の位置を更新する
        self.__my_position += self.__my_velocity
        # 範囲外に出た粒子は範囲内に収める
        """
        if (self.__maxs is not None):
            max_out_of_range_index = self.__my_position > self.__maxs
            self.__my_position[max_out_of_range_index] = self.__maxs[max_out_of_range_index]
        if (self.__mins is not None):
            min_out_of_range_index = self.__my_position < self.__mins
            self.__my_position[min_out_of_range_index] = self.__mins[min_out_of_range_index]
        """

        if (self.__rs is not None):
            if abs(self.__my_position) > self.__rs:
                r_out_of_range_index = self.__my_position
            segment2 = position_i
            segment1 = position_i - self.__my_velocity
            center = (0,0) ; radius = 1
            self.__close_rs = CrossPoint(center, radius, segment1, segment2)
            self.__my_position[r_out_of_range_index] = self.__close_rs[r_out_of_range_index]


        # 評価値を計算する
        score = self.__f(self.__my_position)
        # 最良解を更新する
        if (score < self.__my_best_score):
            self.__my_best_position = self.__my_position
            self.__my_best_score    = score
        # 速度を更新する
        item1     = self.__WEIGHT_V*self.__my_velocity
        random_me = np.random.rand(self.__my_position.size)
        item2     = self.__WEIGHT_ME*random_me*(self.__my_best_position - self.__my_position)
        random_us = np.random.rand(self.__my_position.size)
        item3     = self.__WEIGHT_US*random_us*(best_position - self.__my_position)
        self.__my_velocity = item1 + item2 + item3
        # 範囲外に出た粒子の速度は0とする
        if (self.__rs is not None): self.__my_velocity[r_out_of_range_index] = 0
        # if (self.__mins is not None): self.__my_velocity[min_out_of_range_index] = 0

class ParticleSwarmOptimization(object):
    """粒子群最適化法により最適化を行うクラス"""

    def __init__(self, particles):
        """
        コンストラクタ
        @param particles 粒子のリスト
        """
        self.__particles  = particles
        self.__best_score = sys.float_info.max
        for particle in self.__particles:
            if (particle.score < self.__best_score):
                self.__best_score    = particle.score
                self.__best_position = particle.position

    @property
    def best_score(self):
        return self.__best_score

    @property
    def best_position(self):
        return self.__best_position

    def position(self, index):
        return [particle.position[index] for particle in self.__particles]

    def update(self):
        """
        粒子の位置を更新するメソッド
        """
        for particle in self.__particles:
            particle.move(self.__best_position)

        for particle in self.__particles:
            if (particle.score < self.__best_score):
                self.__best_score    = particle.score
                self.__best_position = particle.position

if __name__ == '__main__':
    # 最適化する関数を入力し、極座標で戻す
    def f(x): return x[0]**2+x[1]**2
    # 変数が取り得る最大値
    #maxs = np.array([1, 1])
    # 変数が取り得る最小値
    #mins = np.array([-1, -1])
    # 変数が取り得る最大半径
    rs = 3
    # ばらまく粒子の個数
    PARTICLE_COUNT = 100
    # 粒子を作成する
    particles = []
    for i in range(PARTICLE_COUNT):
        position = (np.random.rand(2) - 0.5)
        velocity = (np.random.rand(2) - 0.5) / 10
        particle = Particle(f, position, velocity, rs)
        particles.append(particle)
    # ParticleSwarmOptimizationクラスの生成
    pso = ParticleSwarmOptimization(particles)
    # 計算回数
    ITERATION = 100

    # 計算開始
    for i in range(ITERATION):
        pso.update()
        if i == 0 or i == 9 or i == 49 or i == 99:
            fig = plt.figure()
            ax = fig.add_subplot(111, polar=True)
            ax.grid()
            plt.title('i = ' + str(i))
            ax.scatter(pso.position(0), pso.position(1))
            plt.show()
"""
referenced by ljvmiranda921 / pyswarms
"""
