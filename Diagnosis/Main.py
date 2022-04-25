from Data_Processing import Data_processing
from model import Model
import os

class Main:
    def __init__(self):
        model_path = 'D:/Diagnosis and Prediction/Diagnosis/lgb_model.pkl'
        XAI_model_path = 'D:/Diagnosis and Prediction/Diagnosis/lgb_shap_model.pkl'
        data_module = Data_processing()
        model_module = Model()
        lgb_db, train_db = data_module.make_train_db()
        if not os.path.isfile(model_path):
            model_module.LightGBM_train(lgb_db)
        model_module.LightGBM_test(test_x=train_db['train_x_con'][0], test_y=train_db['train_y'])
        if not os.path.isfile(XAI_model_path):
            model_module.make_XAI_model()


if __name__ == "__main__":
    main_operator = Main()