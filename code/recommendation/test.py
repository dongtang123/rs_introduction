import pandas as pd
import pickle
import json

import numpy as np

# 创建一个 C-contiguous 数组
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Initial array contiguous:{arr.flags['C_CONTIGUOUS']}") # True

# 转置数组，使其不再是 C-contiguous
arr_transposed = arr.T
print(f"Transposed array contiguous:{arr_transposed.flags['C_CONTIGUOUS']}") # False

# 使用 ascontiguousarray 将其转换为 C-contiguous
arr_contiguous = np.ascontiguousarray(arr_transposed)
print(f"contiguous array contiguous:{arr_contiguous.flags['C_CONTIGUOUS']}") # True

#验证是否为新数组
print(f"Is it the same array:{arr_contiguous is arr_transposed}") # False
# with open('../rs_introduction_data/usercf_sim_matrix.pkl','rb') as f:
#     data = pickle.load(f)
# print(len(data))