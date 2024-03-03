import numpy as np
import sys
sys.path.append("..")
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from data_loaders import *
from missing_process.block_rules import *
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, f1_score
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument(
      '--data_name',
      choices=[
            "banknote",
             "concrete_compression",
             "wine_quality_white",
            "wine_quality_red",
            "california",
            "climate_model_crashes",
            "connectionist_bench_sonar",
            "qsar_biodegradation", 
            "yeast", 
            "yacht_hydrodynamics","syn1"
            ],
      default="banknote",
      type=str)
parser.add_argument(
      '--miss_type',
      help='missing data type',
      choices=["quantile",
                   "diffuse",
                   "logistic"
                   ],
      default="logistic",
      type=str)
parser.add_argument(
      '--modelname',
      help='imputer',
      #choices=["mean","knn","hyper","gain","XGB","mice","mf","missforest","notmiwae","miwae","tabcsdi","ot"],
      default="mean",
      type=str)
args = parser.parse_args()


def process_target(dataname,y):
    if dataname in ["concrete_compression",
            "wine_quality_white","wine_quality_red",
            "california","yacht_hydrodynamics"
            ]:
        return y,"ML_rmse"
    
    else:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y.reshape(-1, 1))
        return y_encoded,"ML_f1"
    
seed = 1
nfold = 5


def load_impute_data(missingtype,model_name,rule_name,dataname,fold,seed = 1):

    train_impute = np.load(f'impute/{missingtype}/{dataname}/{model_name}/{rule_name}_seed-{seed}_{fold}_train.npy')
    test_impute = np.load(f'impute/{missingtype}/{dataname}/{model_name}/{rule_name}_seed-{seed}_{fold}_test.npy')
    return train_impute,test_impute

def load_train_test(index_file,norm_values,observed_masks,label_values):  

    train_index = index_file["train_index"]
    test_index = index_file["test_index"]

    train_values = norm_values[train_index,:]

    train_masks = observed_masks[train_index,:]

    test_values = norm_values[test_index,:]

    test_masks = observed_masks[test_index,:]


    train_label = label_values[train_index]

    test_label = label_values[test_index]

    return train_values,train_masks,train_label,test_values,test_masks,test_label




def main(args):

    dataname = args.data_name
    miss_type = args.miss_type
    missingtype = args.miss_type
    model_name = args.modelname
    
    if miss_type == "logistic":
        missing_rule = load_json_file("missing_rate.json")
    elif miss_type == "diffuse":
        missing_rule = load_json_file("diffuse_ratio.json")
    elif miss_type == "quantile":
        missing_rule = load_json_file("quantile_full.json")
    else:
        pass

    directory_path = f"datasets/{dataname}"
    data = dataset_loader(dataname)    
    norm_values = np.load(f'{directory_path}/{dataname}_norm.npy')
    label_values, task_type = process_target(dataname,data["target"])

    if task_type =="ML_rmse":
        ml_model_list = [Ridge(), MLPRegressor(random_state=1), SVR()]
    else:
        ml_model_list = [LogisticRegression(random_state=1), MLPClassifier(random_state=1), SVC()]
        #ml_model_list = [LogisticRegression(random_state=1)]

    for ml_model_i in range(len(ml_model_list)):

        ml_model = ml_model_list[ml_model_i]
            
        train_eval_mean = []
        train_eval_std = []
        test_eval_mean = []
        test_eval_std = []


        for rule_name in tqdm(missing_rule):
            
            observed_masks = np.load(f'{directory_path}/{missingtype}/{rule_name}.npy')
            f = open(f'{directory_path}/split_index_cv_seed-{seed}_nfold-{nfold}.json')
            index_file = json.load(f)

            

            train_eval_list = []
            test_eval_list = []


            for fold in index_file:
                index = index_file[fold]
                train_values,train_masks,train_label,test_values,test_masks,test_label = load_train_test(index,norm_values,observed_masks,label_values)
                impute_train,impute_test  = load_impute_data(missingtype,model_name,rule_name,dataname,fold)

                # use imputed train to train models, and full test to evaluation
                train_eval = model_eval(train_label,impute_train,test_values,test_label,task_type,ml_model)
                # use full train to train models, and imputed test to evaluation
                test_eval = model_eval(train_label,train_values,impute_test,test_label,task_type,ml_model)


                train_eval_list.append(train_eval)
                test_eval_list.append(test_eval)


            train_eval_mean.append(np.mean(train_eval_list))
            train_eval_std.append(np.std(train_eval_list))

            test_eval_mean.append(np.mean(test_eval_list))
            test_eval_std.append(np.std(test_eval_list))



            
        df = pd.DataFrame({
        f"train_{task_type}_mean": train_eval_mean,
        f"train_{task_type}_std":train_eval_std,


        f"test_{task_type}_mean": test_eval_mean,
        f"test_{task_type}_std": test_eval_std,


        },index = [rule_name for rule_name in missing_rule])
            
        path = f"results_sep/{missingtype}/{dataname}/{model_name}"
        if not os.path.exists(path):
                # If the path does not exist, create it
            os.makedirs(path)
            
        df.to_csv(f'{path}/{missingtype}_{task_type}_{ml_model_i}.csv')

        print(f'{path}/{missingtype}_{task_type}_{ml_model_i}.csv')



def model_eval(label_train, impute_train, impute_test, label_test, task_type,model):
    if task_type == "ML_rmse":
        # Define regressors
        reg = model
        # Check for NaNs

        if np.isnan(impute_train).any():
            impute_train = np.nan_to_num(impute_train, nan=0).astype(np.float64)
        if np.isnan(impute_test).any():
            impute_test = np.nan_to_num(impute_test, nan=0).astype(np.float64)
        if np.isnan(label_train).any():
            label_train = np.nan_to_num(label_train, nan=0).astype(np.float64)


        if np.isinf(impute_train).any():
            impute_train = np.where(np.isinf(impute_train), 1, impute_train).astype(np.float64)
        if np.isinf(impute_test).any():
            impute_test = np.where(np.isinf(impute_test), 1, impute_test).astype(np.float64)
        if np.isinf(label_train).any():
            label_train = np.where(np.isinf(label_train), 1, label_train).astype(np.float64)
        # try:
        #     reg.fit(impute_train, label_train)    
        # except:
        #     mlp = MLPRegressor()
        #     mlp.fit(impute_train, label_train)
        # try:
        #     y_pred_test = reg.predict(impute_test.astype(np.float64))
        #     y_pred_train = reg.predict(impute_train.astype(np.float64))
            
        #except Exception:
        #reg = SVR()
        reg.fit(impute_train, label_train)
        y_pred_train = reg.predict(impute_train)
        y_pred_test = reg.predict(impute_test)

        # Calculate average RMSE
        train_rmse = np.sqrt(mean_squared_error(label_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(label_test, y_pred_test))
        return test_rmse

    else:
        # Define classifiers

        clf = model
        # Evaluate each classifier and store F1 scores for both train and test

        clf.fit(impute_train, label_train)
        y_pred_train = clf.predict(impute_train)
        y_pred_test = clf.predict(impute_test)
   
        # Calculate average F1 score
        train_f1 = f1_score(label_train, y_pred_train, average='macro')
        test_f1 = f1_score(label_test, y_pred_test, average='macro')
        return test_f1

main(args)