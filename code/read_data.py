import json

import pandas as pd
import csv
import torch
import os


def read_feature(path):
    df = pd.read_csv(path)
    dict_feature = dict()
    for index, item in enumerate(df['article_id']):
        feature = list()
        dict_feature[item] = feature
        # if index>=100:
        #     break
    for i in range(250):
        for index, (emb, item) in enumerate(zip(df[f'emb_{i}'], df['article_id'])):
            dict_feature[item].append(emb)
            # if index >= 100:
            #     break
    return dict_feature


def read_click_item_time(path_train, path_test, offline=True):

    if not offline:
        all_click = pd.read_csv(path_train)
    else:
        trn_click = pd.read_csv(path_train)
        tst_click = pd.read_csv(path_test)

        all_click = trn_click.append(tst_click)
    all_click = all_click.sort_values('click_timestamp')
    df_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))

    df_click['item_time_list'] = list(zip(df_click["click_article_id"], df_click["click_timestamp"]))
    df_click = df_click.groupby('user_id')['item_time_list'].apply(list).reset_index()
    dict_user_item_time = dict()
    for user_id, item_time_list in zip(df_click['user_id'],df_click['item_time_list']):
        dict_user_item_time[user_id] = item_time_list
    # df_click.to_csv('../rs_introduction_data/user_click_group.csv', index=False)
    with open('../rs_introduction_data/user_click_group.json','w') as f:
        json.dump(dict_user_item_time,f,ensure_ascii=False,indent=4)


    return dict_user_item_time


def read_create_time(path):
    df = pd.read_csv(path)
    dict_create_time = dict()
    for article_id, timestamp in zip(df['article_id'], df['created_at_ts']):
        dict_create_time[article_id] = timestamp
    return dict_create_time


def read_type(path):
    df = pd.read_csv(path)
    dict_type = dict()
    for article_id, type_id in zip(df['article_id'], df['category_id']):
        dict_type[article_id] = type_id
    return dict_type


def get_top_click_nums(train_path,test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = df_train.append(df_test)
    topk_click = df['click_article_id'].value_counts().index[:50]
    return topk_click


def get_user_click_article(path):
    dict_user_click = dict()
    df = pd.read_csv(path)
    for user_id, article_id in zip(df['user_id'], df['click_article_id']):
        if user_id in dict_user_click:
            dict_user_click[user_id].append(article_id)
        else:
            dict_user_click[user_id] = [article_id]
    return dict_user_click


if __name__ == "__main__":
    path_feature = os.path.join("../rs_introduction_data/articles_emb.csv")
    # read_feature(path_feature)

    path_click_train_path = os.path.join("../rs_introduction_data/train_click_log.csv")
    path_click_test_path = os.path.join("../rs_introduction_data/testA_click_log.csv")
    read_click_item_time(path_click_train_path, path_click_test_path, True)
    # get_top_click_nums(path_click_train_path,path_click_test_path)

    path_article = os.path.join("../rs_introduction_data/articles.csv")
    # read_create_time(path_article)
    #
    # read_type(path_article)
