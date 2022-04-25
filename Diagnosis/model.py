import pickle
import lightgbm
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os
import shap
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

class Model:
    def __init__(self):
        self.model_path = 'D:/Diagnosis and Prediction/Diagnosis/lgb_model.pkl'
        self.XAI_model_path = 'D:/Diagnosis and Prediction/Diagnosis/lgb_shap_model.pkl'

    def LightGBM_structure(self):
        # LightGBM 파라미터 조정
        params = {}
        params['learning_rate'] = 0.01
        params['boosting_type'] = 'goss'  # GradientBoostingDecisionTree
        params['objective'] = 'multiclass'  # Multi-class target feature
        params['metric'] = 'multi_logloss'  # metric for multi-class
        params['max_depth'] = 1024
        params['num_leaves'] = 128
        params['num_class'] = 80
        params['min_data_in_leaf'] = 20
        params['bagging_fraction'] = 0.8
        params['feature_fraction'] = 0.8
        params['max_bin'] = 128
        params['verbose'] = 1
        # params['device'] = 'gpu'
        # params['max_cat_group '] =
        # params['bagging_freq'] = 10
        # params['reg_sqrt'] = True
        return params

    def LightGBM_train(self, lgb_db):
        params = self.LightGBM_structure()
        model = lightgbm.train(params, lgb_db, 1)
        pickle.dump(model, open(self.model_path, 'wb'))
        return model

    def LightGBM_test(self, test_x, test_y):
        model = pickle.load(open(self.model_path, 'rb'))
        y_pred = model.predict(test_x)
        result = confusion_matrix(test_y, np.argmax(y_pred, axis=1))
        print(result)
        df_cfm = pd.DataFrame(result, index=np.arange(80), columns=np.arange(80))
        print('F-1 Score', f1_score(test_y, np.argmax(y_pred, axis=1), average='macro'))
        print('ACC', accuracy_score(test_y, np.argmax(y_pred, axis=1)))
        plt.figure(figsize=(50, 35))
        plt.title('Abnormal scenario diagnosis function')
        cfm_plot = sn.heatmap(df_cfm/np.sum(df_cfm), annot=True, fmt='.2%', cmap='Blues')
        plt.tight_layout()
        cfm_plot.figure.savefig("cfm.png")

    def make_XAI_model(self):
        model = pickle.load(open(self.model_path, 'rb'))
        lgb_shap_model = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
        pickle.dump(lgb_shap_model, open(self.XAI_model_path, 'wb'))
        return model

