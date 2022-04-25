import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import lightgbm

class Data_processing:
    def __init__(self):
        self.single_list = os.listdir('D:/Diagnosis and Prediction/Data/single')
        self.multi_list = os.listdir('D:/Diagnosis and Prediction/Data/multi')
        self.total_list = self.single_list + self.multi_list
        self.para_list = pd.read_csv('D:/Diagnosis and Prediction/Data/Final_parameter_200825.csv')['0'].tolist()
        self.scaler_path = 'D:/Diagnosis and Prediction/Diagnosis/scaler.pkl'
        self.train_db = {'train_x':[], 'train_y':[], 'train_x_con':[]}
        self.lgb_db_path = 'D:/Diagnosis and Prediction/Diagnosis/lgb_db.pkl'
        self.train_db_path = 'D:/Diagnosis and Prediction/Diagnosis/train_db.pkl'

    def make_min_max_scaler(self):
        if not os.path.isfile(self.scaler_path):
            min_value, max_value, min_scale, max_scale = [], [], [], []
            for i in self.total_list:
                if len(i.split('&')) == 1: # single_db
                    db = pickle.load(open(f'D:/Diagnosis and Prediction/Data/single/{i}', 'rb'))
                elif len(i.split('&')) == 2: # multi_db
                    db = pickle.load(open(f'D:/Diagnosis and Prediction/Data/multi/{i}', 'rb'))
                db = db[self.para_list]
                each_min = db.min(axis=0)
                each_max = db.max(axis=0)
                min_value.append(each_min)
                max_value.append(each_max)
            min_scale.append(np.array(min_value).min(axis=0))
            max_scale.append(np.array(max_value).max(axis=0))
            scaler = MinMaxScaler()
            scaler.fit([min_scale[0], max_scale[0]])
            pickle.dump(scaler, open(self.scaler_path, 'wb'))
        else:
            scaler = pickle.load(open(self.scaler_path, 'rb'))
        return scaler

    def make_label_info(self):
        single_db_list, multi_db_list = [], []
        for i in self.single_list:
            if i[:6] == 'normal':
                single_db_list.append(i[:6])
            else:
                single_db_list.append(i[:7])
        single_db_list = list(set(single_db_list))
        for i in self.multi_list:
            multi_name_split = i.split('&')
            multi_name = f'{multi_name_split[0][:7]}_{multi_name_split[1][1:8]}'
            multi_db_list.append(multi_name)
        multi_db_list = list(set(multi_db_list))
        total_db_list = single_db_list + multi_db_list
        scenario_name, label_num, label_info = [], [], {}
        for num, scenario in enumerate(total_db_list):
            label_info[scenario] = num
            scenario_name.append(scenario)
            label_num.append(num)
        label_info_path = 'D:/Diagnosis and Prediction/Diagnosis/label_info.csv'
        if not os.path.isfile(label_info_path):
            label_info_save = pd.DataFrame([scenario_name, label_num], index=['scenario_name', 'label_num']).T
            label_info_save.to_csv(label_info_path)
        return label_info

    def make_train_db(self): # included labeling
        if not (os.path.isfile(self.lgb_db_path) and os.path.isfile(self.train_db_path)):
            scaler = self.make_min_max_scaler()
            label_info = self.make_label_info()
            for num, i in enumerate(self.total_list):
                print(f'{num+1}/{len(self.total_list)}, 진행률: {round((num/len(self.total_list))*100, 2)}%')
                if len(i.split('&')) == 1:  # single_db
                    db = pickle.load(open(f'D:/Diagnosis and Prediction/Data/single/{i}', 'rb'))
                    if i[:2] == 'ab':
                        db = db[db['KCNTOMS']>=150]
                    db = db[self.para_list]
                    scaled_db = scaler.transform(db)
                    self.train_db['train_x'].append(scaled_db)
                    if i[:6] == 'normal':
                        [self.train_db['train_y'].append(label_info[i[:6]]) for k in range(len(db))]
                    else:
                        [self.train_db['train_y'].append(label_info[i[:7]]) for k in range(len(db))]
                elif len(i.split('&')) == 2:  # multi_db
                    multi_name_split = i.split('&')
                    multi_name = f'{multi_name_split[0][:7]}_{multi_name_split[1][1:8]}'
                    db = pickle.load(open(f'D:/Diagnosis and Prediction/Data/multi/{i}', 'rb'))
                    db = db[db['KCNTOMS'] >= 50]
                    db = db[self.para_list]
                    scaled_db = scaler.transform(db)
                    self.train_db['train_x'].append(scaled_db)
                    [self.train_db['train_y'].append(label_info[multi_name]) for k in range(len(db))]
            self.train_db['train_x_con'].append(np.concatenate(self.train_db['train_x']))
            lgb_db = lightgbm.Dataset(self.train_db['train_x_con'], label=self.train_db['train_y'])
            pickle.dump(lgb_db, open(self.lgb_db_path, 'wb'))
            pickle.dump(self.train_db, open(self.train_db_path, 'wb'))
        else:
            lgb_db = pickle.load(open(self.lgb_db_path, 'rb'))
            train_db_set = pickle.load(open(self.train_db_path, 'rb'))
        return lgb_db, train_db_set





