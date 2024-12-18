import os
import json
import torch
import pandas as pd
import numpy as np


def read_df(train_path, test_path, whether_all=True):
    if whether_all:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        all_data = train_data.append(test_data)
    else:
        train_data = pd.read_csv(train_path)
        all_data = train_data
    all_data = all_data.sort_values('click_timestamp')
    return all_data


def get_item_user_time_list(train_path, test_path, whether_all=True):
    data = read_df(train_path, test_path, whether_all)

    df_click = data.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))

    df_click['user_time_list'] = list(zip(df_click["user_id"], df_click["click_timestamp"]))
    df_click = df_click.groupby('click_article_id')['user_time_list'].apply(list).reset_index()
    dict_item_user_group = dict()
    for item, user_time_list in zip(df_click['click_article_id'],df_click['user_time_list']):
        dict_item_user_group[item] = user_time_list
    with open("../rs_introduction_data/item_user_time_group.json",'w') as f:
        json.dump(dict_item_user_group,f,ensure_ascii=False,indent=4)
    return dict_item_user_group

def get_item_embedding(path):
    data = pd.read_csv(path)
    data_index_dict = dict(zip(data.index, data['article_id']))
    df = [x for x in data.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(data[df].values, dtype=np.float32)
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)
    return item_emb_np,data_index_dict

if __name__ == "__main__":
    original_feature_path = os.path.join("../rs_introduction_data/articles_emb.csv")
    user_click_group_dict_path = os.path.join("../rs_introduction_data/user_click_group.json")
    train_click_path = os.path.join("../rs_introduction_data/train_click_log.csv")
    test_click_path = os.path.join("../rs_introduction_data/testA_click_log.csv")
    itemcf_sim_matrix_path = os.path.join("../rs_introduction_data/itemcf_sim_matrix.json")
    embedding_path = os.path.join("../rs_introduction_data/articles_emb.csv")
    # get_item_user_time_list(train_click_path,test_click_path,True)
    get_item_embedding(embedding_path)