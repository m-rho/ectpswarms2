import xlrd
import pandas as pd
import numpy as np

s = []
R = []
L = []

list_s = pd.read_excel('poyo.xlsx', sheet_name=0)
s.append(list_s)

list_r = pd.read_excel('poyo.xlsx', sheet_name=1)
R.append(list_r)

list_l = pd.read_excel('poyo.xlsx', sheet_name=2)
L.append(list_l)

total = np.sum(map((s - ([1,1]*[R,L]))**2))
print(total)
