from sentence_transformers import SentenceTransformer
from sklearn import svm, neighbors
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd

from .feature import bert_embedding, extract_col_feature, cal_vote_result_for_KNN, classifier
# from feature_extractor import extract


bert_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def cosine_similarity(a, b):
    similarity = (1 - cosine(a, b))
    return similarity


def DP_matching(master_data, slave_data, window_size=3):
    # set1: master_data
    # set2: slave_data
    set1_result = []
    set2_result = []

    #紀錄當前走多少
    set1_stop_len = 0
    set2_stop_len = 0 

    finish = max(len(master_data), len(slave_data))
    #終止條件
    set1_finish = finish
    set2_finish = finish

    set1_index = 0
    set2_index = 0
    while set1_stop_len<set1_finish and  set2_stop_len<set2_finish:
        temp = []  
        #print("set1_index:",set1_index)
        #print("set2_index:",set2_index)
        try: #0
            temp.append(cosine_similarity(master_data[set1_index],slave_data[set2_index]))
        except:
            temp.append(0)
        for size in range(1,window_size):
            try: #1
                temp.append(cosine_similarity(master_data[set1_index+size],slave_data[set2_index]))
            except:
                temp.append(0)

        for size in range(1,window_size):          
            try: #3
                temp.append(cosine_similarity(master_data[set1_index],slave_data[set2_index+size]))
            except:
                temp.append(0) 
              
                
        #print(temp)
        #     A'    B'    C'  D'   E'(set1)
        #  A  0     1     2   ...  w
        #  B  w+1   w'+1  1'  ...  w'
        #  C  w+2   w'+2  0'' ...  w'' 
        #  D  w+3   w'+3  3''
        #  E              
        #(set2)   
        #print(temp)
        max_value = max(temp)
        max_index = temp.index(max_value)
        #print('max_index:',max_index)
        if temp[0] > 0.95:
            if set1_index < len(master_data):
                    set1_result.append(set1_index)
            else:
                set1_result.append(-1)
            if set2_index < len(slave_data):
                set2_result.append(set2_index)
            else:
                set2_result.append(-1)    
            set1_index +=1
            set2_index +=1
            set1_stop_len +=1
            set2_stop_len +=1
            continue       
        if max_index == 0:
                if set1_index < len(master_data):
                    set1_result.append(set1_index)
                else:
                    set1_result.append(-1)
                if set2_index < len(slave_data):
                    set2_result.append(set2_index)
                else:
                    set2_result.append(-1)    
                set1_index +=1
                set2_index +=1
                set1_stop_len +=1
                set2_stop_len +=1
        elif max_index<window_size:    
            compare_cosine = []
            for size in range(1,window_size):
                try: #1
                    C1 = cosine_similarity(master_data[set1_index+max_index],slave_data[set2_index+size])
                    #C1 = 1 - spatial.distance.cosine(master_data[set1_index+1], slave_data[set2_index+1])
                except:
                    C1 = 0
                compare_cosine.append(C1)

            #print(' C1:', C1,'\nC2:', C2)
            if all( max_value >= checkvalue  for checkvalue in compare_cosine)and max_value>=0.5:
                for num in range(max_index):
                     set2_result.append(-1)


                if set1_index < finish:
                # set1_result.append(set1_index)
                    for num in range(max_index+1):
                         set1_result.append(set1_index+num)               
                else:
                    set1_result.append(-1)


                if set2_index <finish:
                    set2_result.append(set2_index)
                else:
                    set2_result.append(-1)   
                set1_index +=(1+max_index)
                set2_index +=1
                set1_stop_len +=(1+max_index)
                set2_stop_len +=1
                set1_finish += (1+max_index-1)                  
            else:
                set1_result.append(set1_index)
                set2_result.append(-1)
                set1_index +=1
        elif max_index>=window_size:
            compare_cosine = []
            for size in range(1,window_size):
                try: #1
                    C1 = cosine_similarity(master_data[set1_index+size],slave_data[set2_index+max_index])
                    #C1 = 1 - spatial.distance.cosine(master_data[set1_index+1], slave_data[set2_index+1])
                except:
                    C1 = 0
                compare_cosine.append(C1)

            #print(' C1:', C1,'\nC2:', C2)
            if all( max_value >= checkvalue  for checkvalue in compare_cosine)and max_value>=0.5:
                for num in range(max_index-window_size+1):
                     set1_result.append(-1)

                #print("長度:",len(master_data),"set2index:",set2_index)
                if set2_index < finish:
                    # set1_result.append(set1_index)
                    for num in range(max_index+1-window_size+1):
                         #print("set2 add :", set2_index+num)
                         set2_result.append(set2_index+num)               
                else:
                    set2_result.append(-1)


                if set1_index < finish:
                    set1_result.append(set1_index)
                else:
                    set1_result.append(-1) 
                    
                set1_index +=1
                set2_index +=(1+max_index-window_size+1)
                set1_stop_len +=1
                set2_stop_len +=(1+max_index-window_size+1)
                set2_finish += (1+max_index-1-window_size+1)                  
            else:
                set2_result.append(set2_index)
                set1_result.append(-1)
                set2_index +=1
        #print(set1_result)
        #print(set2_result)
        """
        print('stop1:',set1_stop_len)
        print('stop2:',set2_stop_len)
        print('finish1',set1_finish)
        print('finish2',set2_finish)
        print(set1_result)
        print(set2_result)
        """   
    set1_result = [x+1 for x in set1_result]
    set2_result = [x+1 for x in set2_result]
    # print(set1_result)
    # print(set2_result)
    while set1_result[-1] == 0 and set2_result[-1]==0:
        set1_result.pop()
        set2_result.pop()
    return  set1_result, set2_result


def sets_data_embedding(sets_data):
    embed_sets_data = []
    used_sets_index = []
    for index, set_data in enumerate(sets_data):
        if len(set_data) <= 3:
            print(f"Set_{index} data less than 3. Skip this set!")
            continue

        rows_data = []

        for row_data in set_data:
            rows_data.append(''.join(row_data))

        used_sets_index.append(index)
        embed_rows_data = bert_embedding(rows_data)
        avg_embed_rows_data = embed_rows_data.sum(axis=0)/len(embed_rows_data)
        embed_sets_data.append(avg_embed_rows_data.tolist())
    return embed_sets_data, used_sets_index


def sets_matching(master, slave):
    sets_combine_pairs = {} #用來處理多Page一起合併的結果

    print("- master -")
    embed_master_data, used_master_index = sets_data_embedding(master)
    print("- slave -")
    embed_slave_data, used_slave_index = sets_data_embedding(slave)

    master_sets_length, slave_sets_length = len(used_master_index), len(used_slave_index)
    
    # print('embed_master_data:', type(embed_master_data), len(embed_master_data))
    # print("embed_master_data:", embed_master_data)
    # print('embed_slave_data:', type(embed_slave_data),len(embed_slave_data)) 
    # print("embed_slave_data:", embed_slave_data)

    #print("master_sets_length:",master_sets_length, "slave_sets_length:",slave_sets_length)
    sets_length_diff = master_sets_length - slave_sets_length
    window_size = abs(sets_length_diff) + 5
    #print("window_size:", window_size)
    
    if master_sets_length == 0 and slave_sets_length == 0:
        raise ValueError('Both sets of data are zero')
    else:
        set1_result, set2_result = DP_matching(embed_master_data, embed_slave_data, window_size)

    collect_no_match_vec = [] #用來收集對齊到空的那些set的向量表示
    collect_no_match_num = [] #用來表示每個set有幾個對齊到空的數量 , for example, [2,3,4] 表示page1 有兩個set對到空 page2有3個...
    
    slave_page_no_match_count = 0
    for i in range(len(set2_result)):
        if set1_result[i] in sets_combine_pairs:
                #print(set1_result[i])
                sets_combine_pairs.get(set1_result[i]).append(set2_result[i])
        else:
            temp = [set2_result[i]]            
            sets_combine_pairs[set1_result[i]] = temp

        if set1_result[i] == 0: #表示page2的set2_result[i] 對齊到空集 要將資料額外取出來 最後再統整時重算相似度
                collect_no_match_vec.append(embed_slave_data[set2_result[i]-1]) #page2的set2_result[i] 的向量表示, set2_result[i]表示在page2第i個set
                slave_page_no_match_count +=1

    collect_no_match_num.append(slave_page_no_match_count)
    
    #print("master_result:", set1_result)
    #print("slave_result:", set2_result)
    return sets_combine_pairs, used_master_index, used_slave_index


def col_matching_forDB(set_result, train, predict, train_index, predict_index, model_select=2):
    #print(type(train))
    print(set_result)
    result = train.copy()
    #print(type(result))
    for master_index,slave_index in set_result.items():
            # print("master_index:",master_index,"slave_index:",slave_index)
            #print("serialNumber_index:",serialNumber_index-1)
            #print("slave_index:",slave_index[serialNumber_index-1])
            #print("start set matching")
            #print(slave_index[serialNumber_index-1]-1)
            if master_index == 0: 
                #no_matching_set += len(slave_index)
                #print("master_index == 0 skip")
                for le in range(len(slave_index)):
                    result.append(predict[le])
            else:
                if (slave_index[0]-1) == -1:
                   # print("slave_index[0]-1 skip")
                    #no_matching_set += len(slave_index)
                    continue
                else:

                    sets_train = train[train_index[master_index-1]]
                    #sets_test = predict[slave_index[serialNumber_index-1]-1]
                    if (slave_index[0]-1) <0:
                        print("slave_index[0]-1 <0 skip",slave_index[0])
                        continue;
                    sets_test = predict[predict_index[slave_index[0]-1]]

                    if len(sets_train[0]) == len(sets_test[0]):
                        #print()
                        print("same col")
                    else:
                        print("diff col",len(sets_train[0])," ",len(sets_test[0]))
                        #error += 1
                        continue
                    arr_t = np.array(sets_train)

                    df = pd.DataFrame(arr_t, columns=['col'+str(item) for item in range(0,len(arr_t.T))])

                    arr_p = np.array(sets_test)

                    df_predict = pd.DataFrame(arr_p, columns=['col'+str(item) for item in range(0,len(arr_p.T))])    

                    embeddings = extract_col_feature(df)

                    embeddings_predict = extract_col_feature(df_predict)

                    label_ = []
                    for j in range(0,len(df.columns)):
                        for i in range(0,len(df)):
                            label_.append(j)

                    label_prdict = []
                    for j in range(0,len(df_predict.columns)):
                        for i in range(0,len(df_predict)):
                            label_prdict.append(j)                    

                    label_ = np.array(label_)
                    #print(len(label_))
                    label_prdict = np.array(label_prdict)
                    #print(len(label_prdict))

                    if model_select == 1:
                        if 1 in label_:
                            my_model = svm.SVC(probability=True) 
                            my_model = my_model.fit(embeddings, label_)
                    elif model_select == 2: 

                        knn_clf = neighbors.KNeighborsClassifier(n_neighbors = len(df.columns))
                        my_model = knn_clf.fit(embeddings, label_)
                        y_predicted = my_model.predict(embeddings_predict)
                        #print(y_predicted)
                        #print(label_prdict)



                column_num = len(df.columns)
                testing_row_number = len(df_predict)
                if 1 in label_:
                        group_vote = cal_vote_result_for_KNN(my_model,column_num,testing_row_number,embeddings_predict)
                        ans = [x for x in range(0,column_num)]
                        column_result = classifier(column_num,group_vote)
                else:
                    ans = [0]         
                    column_result = [0]
                #print(ans)
                #print(column_result)

                #res = sum(x == y for x, y in zip(ans, column_result)) #計算當前set 正確的col數
                #print(res)
                #count_all_col_num_temp += len(ans)
                #count_correct_num_temp += res
                #Evaluation(ans,column_result,f1,precision,recall,accuracy_e,ans_list,column_result_list)
                #ans_.append(x[serialNumber_index])
                #ans_index.append(serialNumber_index)

                #紀錄main set正確col數量
                #if count_current_set_index == main:
                #    main_count_all_col_num.append(len(ans)) 
                #    main_count_correct_num.append(res)
                #    main_count_error_num.append(len(ans)-res)

                #tmp = res/len(ans) #計算當前set準確度
                #evl_avg_set += tmp

                #count_current_set_index +=1
                print(column_result)
                #total_set_num +=1
                
                #print(column_result)
                # print(predict[slave_index[0]-1])
                index = master_index-1
                #print(result[index])
                for i in range(len(predict[predict_index[slave_index[0]-1]])):
                    #print(i)
                    temp_arr = []
                    
                    for x in column_result:
                        #print(slave_index[0]-1)
                        #print(predict_index[slave_index[0]-1])
                        #print('x:',x)
                        #print('i:',i)
                        #print(len(predict[predict_index[slave_index[0]-1]][i]))
                        #print(predict[predict_index[slave_index[0]-1]][i])
                        #print(predict[predict_index[slave_index[0]-1]][i][x])
                        #print(len(predict[predict_index[slave_index[0]-1]][i][x]))
                        temp_arr.append(predict[predict_index[slave_index[0]-1]][i][x])
                    
                    
                    #print(result[index])
                    result[train_index[index]].append(temp_arr)
                #print(result[index])    
                print("done")
    #print(result)
    
    return result