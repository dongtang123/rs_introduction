import pandas as pd


# 示例字典
my_dict = {'a': 1546546640, 'b': 5, 'c': 30, 'd': 20}

# 按值从大到小排序
sorted_dict = sorted(my_dict.items(), key=lambda item: item[1], reverse=True)[:3]
print(sorted_dict)
