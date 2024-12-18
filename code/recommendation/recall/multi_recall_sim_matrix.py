import math

from get_feature.read_data import get_top_click_nums
import os
import pickle
import json
from get_feature.multi_recall_read_data import read_df, get_item_embedding
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import faiss


def get_user_active_degree(train_path, test_path, offline=True):
    df = read_df(train_path, test_path, offline)
    data = df.groupby('user_id')["click_article_id"].count().reset_index()
    mm = MinMaxScaler()
    data['click_article_id'] = mm.fit_transform(data[['click_article_id']])
    user_active_degree = dict(zip(data['user_id'], data['click_article_id']))
    return user_active_degree


def usercf_sim_matrix(item_user_time_group_path, train_path, test_path):
    with open(item_user_time_group_path, 'r') as f:
        item_user_time_group = json.load(f)
    # usercf: 用户i喜欢的物品与用户j喜欢的物品的交集/根号下(用户i喜欢的物品*用户j喜欢的物品)
    user_liked_items = dict()
    user_sim_matrix = dict()
    user_active_degree = get_user_active_degree(train_path, test_path, True)
    for item, user_time_list in tqdm(item_user_time_group.items()):
        for user1, timestamp1 in user_time_list:
            user_liked_items.setdefault(user1, 0)
            user_liked_items[user1] += 1
            user_sim_matrix.setdefault(user1, {})
            for user2, timestamp2 in user_time_list:
                if user2 == user1: continue
                user_sim_matrix[user1].setdefault(user2, 0)
                user_sim_matrix[user1][user2] += 50 * (
                        user_active_degree[user1] + user_active_degree[user2]) / math.log(len(user_time_list) + 1)
    user_sim_matrix_copy = user_sim_matrix.copy()
    for user1, user2_item_num in user_sim_matrix_copy.items():
        for user2, weight in user2_item_num.items():
            user_sim_matrix[user1][user2] = weight / math.sqrt(user_liked_items[user1] * user_liked_items[user2])
    with open('../../rs_introduction_data/usercf_sim_matrix.pkl', 'wb') as f:
        pickle.dump(user_sim_matrix, f)
    return user_sim_matrix


def itemcf_sim_matrix():
    return


def item_emb_sim_matrix(path_embedding, topk):
    df_emb,data_index_dict = get_item_embedding(path_embedding)
    # res = faiss.StandardGpuResources()
    item_index = faiss.IndexFlatIP(df_emb.shape[1])
    # index_gpu = faiss.index_cpu_to_gpu(res, 0, item_index)
    item_index.add(df_emb)
    sim, idx = item_index.search(df_emb, topk)
    item_sim_dict = dict()
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(df_emb)), sim, idx)):
        target_raw_id = data_index_dict[target_idx]
        item_sim_dict.setdefault(target_raw_id,{})
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = data_index_dict[rele_idx]
            item_sim_dict[target_raw_id].setdefault(rele_raw_id, 0)
            item_sim_dict[target_raw_id][rele_raw_id] += sim_value

        # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open('../../rs_introduction_data/emb_i2i_sim.pkl', 'wb'))


    return sim,idx


if __name__ == "__main__":
    root_path = os.path.join("../../rs_introduction_data")
    user_item_time_group_path = os.path.join(root_path, "user_click_group.json")
    item_user_time_group_path = os.path.join(root_path, "item_user_time_group.json")
    train_click_path = os.path.join("../../rs_introduction_data/train_click_log.csv")
    test_click_path = os.path.join("../../rs_introduction_data/testA_click_log.csv")
    item_embedding_path = os.path.join("../../rs_introduction_data/articles_emb.csv")
    # get_user_active_degree(train_click_path, test_click_path, True)
    # usercf_sim_matrix(item_user_time_group_path, train_click_path, test_click_path)
    item_emb_sim_matrix(item_embedding_path,5)
