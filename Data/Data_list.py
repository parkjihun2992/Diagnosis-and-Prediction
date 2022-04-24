import os
import pandas as pd
import numpy as np
import pickle

single_db = os.listdir('./single')
single_db_list = []
for i in single_db:
    if i[:6] == 'normal':
        single_db_list.append(i[:6])
    else:
        single_db_list.append(i[:7])
single_db_list = set(single_db_list)

single_db_dict = {i:[] for i in single_db_list}
for i in single_db:
    with open(f'./single/{i}', 'rb') as f:
        db = pickle.load(f)
    if i[:6] == 'normal':
        single_db_dict[i[:6]].append(len(db))
    else:
        single_db_dict[i[:7]].append(len(db[db['KCNTOMS']>=150]))
print(single_db_dict)

info_single = {'scenario':[], 'file_num':[], 'file_sum':[]}
for i in single_db_dict.keys():
    info_single['scenario'].append(i)
    info_single['file_num'].append(len(single_db_dict[i]))
    info_single['file_sum'].append(sum(single_db_dict[i]))

single_db_info = pd.DataFrame([info_single['scenario'], info_single['file_num'], info_single['file_sum']],index=['Scenario', 'Scenario_num', 'Scenario_sum']).T
single_db_info.to_csv('single_data_info.csv')

multi_db = os.listdir('./multi')
multi_db_list = []
for i in multi_db:
    multi_name_split = i.split('&')
    multi_name = f'{multi_name_split[0][:7]}_{multi_name_split[1][1:8]}'
    multi_db_list.append(multi_name)
multi_db_list = set(multi_db_list)

multi_db_dict = {i:[] for i in multi_db_list}
for i in multi_db:
    with open(f'./multi/{i}', 'rb') as f:
        db = pickle.load(f)
    split_name = i.split('&')
    db_name = f'{split_name[0][:7]}_{split_name[1][1:8]}'
    multi_db_dict[db_name].append(len(db[db['KCNTOMS']>=50]))
print(multi_db_dict)

info_multi = {'scenario':[], 'file_num':[], 'file_sum':[]}
for i in multi_db_dict.keys():
    info_multi['scenario'].append(i)
    info_multi['file_num'].append(len(multi_db_dict[i]))
    info_multi['file_sum'].append(sum(multi_db_dict[i]))

multi_db_info = pd.DataFrame([info_multi['scenario'], info_multi['file_num'], info_multi['file_sum']],index=['Scenario', 'Scenario_num', 'Scenario_sum']).T
multi_db_info.to_csv('multi_data_info.csv')