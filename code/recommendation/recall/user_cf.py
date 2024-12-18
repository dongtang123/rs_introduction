from get_feature.read_data import get_top_click_nums

import os
import json
import torch




if __name__ =="__main__":
    train_path = os.path.join("../../rs_introduction_data/train_click_log.csv")
    test_path = os.path.join("../../rs_introduction_data/train_click_log.csv")
    top = get_top_click_nums(train_path, test_path)
    print(top)