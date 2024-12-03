import os
import math
import json
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

from read_data import get_top_click_nums


def get_item_sim_matrix_cf(user_click_group_dict_path):
    with open(user_click_group_dict_path, 'r') as f:
        dict_user_item_time_list = json.load(f)
    print(len(dict_user_item_time_list))
    dict_combine_like = dict()
    dict_like_num = defaultdict(int)
    for user_id, item_time_list in dict_user_item_time_list.items():
        for item1, time1 in item_time_list:
            dict_like_num[item1] += 1
            dict_combine_like.setdefault(item1, {})
            for item2, time2 in item_time_list:
                if item2 == item1:
                    continue
                dict_combine_like[item1].setdefault(item2, 0)
                dict_combine_like[item1][item2] += 1 / math.log(len(item_time_list) + 1)
    dict_sim_matrix = dict_combine_like.copy()
    for item1, sim_to_item1 in dict_combine_like.items():
        for item2, weight in sim_to_item1.items():
            dict_sim_matrix[item1][item2] = weight / math.sqrt(dict_like_num[item1] * dict_like_num[item2])
    with open("../rs_introduction_data/itemcf_sim_matrix.json", 'w') as f:
        json.dump(dict_sim_matrix, f, ensure_ascii=False, indent=4)
    print("Similarity has computed!")
    return dict_sim_matrix


def itemcf_recommendation(train_path, test_path, user_clik_group_path, sim_matrix_path, top_k):
    df_test = pd.read_csv(test_path)
    df_test['item_time_list'] = list(zip(df_test["click_article_id"], df_test["click_timestamp"]))
    df_test.sort_values('click_timestamp')
    df_test = df_test.groupby('user_id')['item_time_list'].apply(list).reset_index()
    recall_res = dict()
    with open(user_clik_group_path, 'r') as f:
        user_clik_group = json.load(f)
    with open(sim_matrix_path, 'r') as f:
        sim_matrix = json.load(f)
    hot_top = get_top_click_nums(train_path, test_path)
    for user_id in tqdm(df_test['user_id']):
        item_time_list = user_clik_group[str(user_id)]
        list_sim = {}

        list_sim_item = []
        for item1, time in item_time_list:
            item1 = str(item1)
            if item1 in sim_matrix:
                for item2, sim in sorted(sim_matrix[item1].items(), key=lambda x: x[1], reverse=True)[:10]:
                    if item2 not in [i for i, j in item_time_list]:
                        if item2 not in list_sim_item:
                            list_sim.setdefault(item2,0)
                            list_sim_item.append(item2)
                            list_sim[item2] += sim
        # print(list_sim_item)
        count_hot = 0
        while len(list_sim) < 10:
            if str(hot_top[count_hot]) not in list_sim_item and str(hot_top[count_hot]) not in [i for i, j in item_time_list]:
                list_sim[str(hot_top[count_hot])] = -100 - count_hot
                list_sim_item.append(str(hot_top[count_hot]))
            count_hot += 1
        list_sim = sorted(list_sim.items(), key=lambda x: x[1], reverse=True)[:top_k]
        # print(list_sim)
        recall_res[user_id] = list_sim

    with open("../rs_introduction_data/result.json", 'w') as f:
        json.dump(recall_res, f, ensure_ascii=False, indent=4)
    recall_res = change_to_submit(recall_res)
    recall_res.to_csv("../rs_introduction_data/result.csv", index=False)

    return recall_res


def change_to_submit(recall_dict):
    df = pd.DataFrame()
    df['user_id'] = [user_id for user_id in recall_dict.keys()]
    df['article_1'] = [article[0][0] for article in recall_dict.values()]
    df['article_2'] = [article[1][0] for article in recall_dict.values()]
    df['article_3'] = [article[2][0] for article in recall_dict.values()]
    df['article_4'] = [article[3][0] for article in recall_dict.values()]
    df['article_5'] = [article[4][0] for article in recall_dict.values()]
    if_duplicates(df)
    return df


def if_duplicates(df):
    for u,i1, i2, i3, i4, i5 in tqdm(
            zip(df['user_id'],df['article_1'], df['article_2'], df['article_3'], df['article_4'], df['article_5'])):
        list_d = [i1, i2, i3, i4, i5]
        has_duplicates = len(list_d) != len(set(list_d))
        if has_duplicates:
            print(u)

    print("None")
    return


if __name__ == "__main__":
    original_feature_path = os.path.join("../rs_introduction_data/articles_emb.csv")
    user_click_group_dict_path = os.path.join("../rs_introduction_data/user_click_group.json")
    train_click_path = os.path.join("../rs_introduction_data/train_click_log.csv")
    test_click_path = os.path.join("../rs_introduction_data/testA_click_log.csv")
    # get_item_sim_matrix_cf(user_click_group_dict_path)
    sim_matrix_path = os.path.join("../rs_introduction_data/itemcf_sim_matrix.json")
    itemcf_recommendation(train_click_path, test_click_path, user_click_group_dict_path, sim_matrix_path, 5)
    res = pd.read_csv(os.path.join("../rs_introduction_data/result.csv"))
    if_duplicates(res)
