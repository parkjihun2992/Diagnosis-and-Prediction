import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Data_processing:
    def __init__(self):
        self.single_list = os.listdir('D:/Diagnosis and Prediction/Data/single')
        self.multi_list = os.listdir('D:/Diagnosis and Prediction/Data/multi')
        self.para_list = pd.read_csv('D:/Diagnosis and Prediction/Data/Final_parameter_200825.csv')['0'].tolist()
        self.scaler = MinMaxScaler()
        self.scale_value = {'min_value':[], 'max_value':[]}

    def min_max_scaler(self):
        while True:




if __name__ == "__main__":
    D = Data_processing()
