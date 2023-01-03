from sentence_transformers import SentenceTransformer
import validators
import numpy as np
import time
import re

bert_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def bert_embedding(sentences):
    embeddings = bert_embedding_model.encode(sentences)
    return embeddings


def is_empty(data):
    if data is None and data.strip() == '':
        return True
    return False


# check if it is url
def is_url(data):
    if is_empty(data):
        return -1

    # Regex to check valid URL
    url_regex = ("((http|https)://)(www.)?" +
                 "[a-zA-Z0-9@:%._\\+~#?&//=]" +
                 "{2,256}\\.[a-z]" +
                 "{2,6}\\b([-a-zA-Z0-9@:%" +
                 "._\\+~#?&//=]*)")

    if re.search(url_regex, data) and validators.url(data):
        return 1
    return -1


# check if it is float
def is_float(data):
    if is_empty(data):
        return -1

    float_regex = ("[-+]?\d*\.\d+|\d+")

    if re.search(float_regex, data):
        return 1
    return -1


# check if it is number
def is_number(data):
    if is_empty(data):
        return -1

    number_regex = ("^\d*[.,]?\d*$")

    if re.search(number_regex, data):
        return 1
    return -1


# check if it is percent
def is_percent(data):
    if is_empty(data):
        return -1

    percent_regex = ("\d+(\.\d+)?%")

    if re.search(percent_regex, data):
        return 1
    return -1


# check if it is time
def is_time(data):
    if is_empty(data):
        return -1

    try:
        time.strptime(data, '%H:%M')
        return 1
    except ValueError:
        return -1


def extract_col_feature(sets_data):
    col_features = [[], [], [], [], []]
    embed_sets_data = []
    for column in sets_data:
        embed_sets_data += sets_data[column].tolist()
    embed_sets_data = bert_embedding(embed_sets_data)

    for col_name in sets_data:
        for col in sets_data[col_name].tolist():
            col_features[0].append(is_number(col))
            col_features[1].append(is_float(col))
            col_features[2].append(is_url(col))
            col_features[3].append(is_percent(col))
            col_features[4].append(is_time(col))

    for i, feature in enumerate(col_features):
        col_features[i] = np.reshape(
            feature, (len(sets_data) * len(sets_data.columns), 1))
    for feature in col_features:
        embed_sets_data = np.concatenate((embed_sets_data, feature), axis=1)

    return embed_sets_data


def cal_vote_result_for_KNN(model, column_num, row_number, embeddings_predict):
  #  column_num = len(df.columns)
  #  testing_row_number = len(df_predict)
    group_vote = []

    index = 0
    for j in range(0, column_num):
        vote = [0] * column_num
        for i in range(0, row_number):

            cell_prob = model.predict_proba(
                embeddings_predict[index + i][:].reshape(1, -1))
            # print(cell_prob)
            if np.max(cell_prob) > 0.6:
                vote[np.argmax(cell_prob)] += 1
        group_vote.append(vote)
        index = index + row_number
    # print(group_vote)
    return group_vote


def classifier(column_num, group_vote):
    column_result = [-1] * column_num
    while -1 in column_result:  # 當所有column都被對齊之後結束迴圈
        # print("start")
        for index, c in enumerate(group_vote):

            if all(kk == -1 for kk in group_vote[index]):  # 如果此column已對齊，跳過此回合
                # print("skip")
                continue

            # print("group " ,index , "vote_result",c ,"choose:" , np.argmax(c))

            # print(column_result[ np.argmax(c)])
            # -1表示目前沒有任何column與這個column對齊
            if column_result[np.argmax(c)] != -1:
                candidate = column_result[np.argmax(c)]

                candidate_votenum = group_vote[candidate][np.argmax(c)]

                c_votenum = group_vote[index][np.argmax(c)]

                if candidate_votenum < c_votenum:  # 競爭 看哪一個表較多
                    # print("candidate_votenum " , candidate_votenum ,"< c_votenum" , c_votenum)
                    column_result[np.argmax(c)] = index
            else:  # 不用競爭
                column_result[np.argmax(c)] = index
            # print(column_result)
        # 將已經選定的column設定成-1 使得下一次選column時 不需考慮這些已經對齊完成的
        for r_index, r in enumerate(column_result):
            if r != -1:
                group_vote[r] = [-1]*column_num
                for i in group_vote:
                    i[r_index] = -1

    return column_result
