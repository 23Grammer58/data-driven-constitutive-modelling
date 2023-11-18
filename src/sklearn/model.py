import os
from math import sin, pi
import joblib
from tqdm.notebook import tqdm
from MLPRegressorWrapper import MLPRegressorWrapper

import numpy as np
import pandas as pd
from scipy.linalg import norm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error 

# from p_k_2 import piola_kirchg`off_2

#load dataset
def load_data(path="biaxial_three_different_holes.xlsx", validation=False,
               split=True, extended_data=False, data_preprocessing=False):

    if extended_data:

        files = os.listdir("data")[:5]
        # print(files)
        df_list = []
        for file in files:
            df_list.append(pd.read_excel(os.path.join('data', file)))

        df = pd.concat(df_list)

    else:

        file =  os.path.join('data',path)
        df = pd.read_excel(file)

    # n = df.shape[0]
    # print("DataFrame shape =", n)

    X = df.drop(columns=['d(psi)/d(I1)', 'd(psi)/d(I2)'])
    y = df.drop(columns=['d(psi)/d(I1)', 'I1', 'I2'])

    # print("X shape = ", X.shape)
    # print("y shape = ", y.shape)
    # print(X["I1"].values)
    if data_preprocessing:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y)

    # y = df.drop(columns=['I1', 'I2'])

    if not split:
        return X, y
    
    X_train, X_test, y_train, y_test = \
            train_test_split(X, y,
                            test_size=0.2, random_state=2)
    
    if not validation:
        return X_train, X_test, y_train, y_test
    
    X_train, X_val, y_train, y_val = \
            train_test_split(X_train, y_train,
                            test_size=0.25, random_state=2)
    
    print("train")
    print("X shape = ", X_train.shape)
    print("y shape = ", y_train.shape)
    print("Test")
    print("X shape = ", X_test.shape)
    print("y shape = ", y_test.shape)
    print("Validation")
    print("X shape = ", X_val.shape)
    print("y shape = ", y_val.shape)
    
    return X_train, X_test, X_val, y_train, y_test, y_val
    
#compute mean squared error
def mse(model, y_pred, y_train, y_test, X_train):

    mse_train = mean_squared_error(y_train, model.predict(X_train))
    mse_test = mean_squared_error(y_test, y_pred)

    print("Train MSE:", mse_train )
    print("Test MSE:", mse_test)

#normalization data
def scale_data(data, return_params=False):

    scaler = MinMaxScaler()
    scaler = scaler.fit(data)
    if return_params:
        return scaler.fit_transform(data), (scaler.mean_, scaler.scale_)
    else:
        return scaler.fit_transform(data)
    # return scaler

def data_preprocessing(path="biaxial_three_different_holes.xlsx",
                        extended_data=False, validation=False):
    
    if not validation:
        X_train, X_test, y_train, y_test = \
            load_data(path=path, extended_data=extended_data,
                    validation=validation)
        
        X_train_reg = scale_data(X_train)
        X_test_reg = scale_data(X_test)
        y_train_reg = scale_data(y_train)
        y_test_reg = scale_data(y_test)

        return X_train_reg, X_test_reg, y_train_reg, y_test_reg
    
    else:
        X_train, X_test, X_val, y_train, y_test, y_val = \
            load_data(path=path, extended_data=extended_data,
                    validation=validation)
        
        X_train_reg = scale_data(X_train)
        X_test_reg = scale_data(X_test)
        X_val_reg = scale_data(X_val)

        y_train_reg = scale_data(y_train)
        y_test_reg = scale_data(y_test)
        y_val_reg = scale_data(y_val)

        return X_train_reg, X_test_reg, X_val_reg, \
                    y_train_reg, y_test_reg, y_val_reg


def perceptron_fit(model_name="binn",
                    save_model=False,
                    extended_data=False,
                    path="biaxial_three_different_holes.xlsx",
                    validation=False):

    clf = MLPRegressorWrapper(model_name=model_name,
                                activation='tanh',
                                random_state=0, 
                                hidden_layer_sizes=270,
                                max_iter=100)
    
    if not validation:
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = \
        load_data(path=path,
                    extended_data=extended_data,
                    validation=validation, 
                    data_preprocessing=data_preprocessing)
        
    else:         
        X_train_reg, X_test_reg, X_val_reg, \
                    y_train_reg, y_test_reg, y_val_reg = \
        load_data(path=path,
                    extended_data=extended_data,
                    validation=validation, 
                    data_preprocessing=data_preprocessing)


    clf.fit(X_train_reg, y_train_reg)
    y_pred = clf.predict(X_test_reg)

    mse(clf, y_pred, y_train_reg, y_test_reg, X_train_reg)
        # print("Score:", clf.score(X_test_reg, y_test_reg))

    # if validation:
    #     pk_anl = np.load("piola_kirchhoff_analytical.npy")
    #     y_pred_inversed = y_pred * (y.max() - y.min()) + y.min()

    if save_model: clf.save_model()

    return clf

#try to predict dpsi2 from different experiments with learned on biaxial_one_big_hole 
def validation(model="binn.joblib"):


    files = os.listdir("data")
    print(files)

    model_file = os.path.join('pretrained_models', model)

    loaded_model = joblib.load(model_file)

    errors = []
    for file in files:
        X, y = load_data(path=file, data_preprocessing=True, split=False)
        y_pred = loaded_model.predict(X)
        # print("y_true", y_reg)
        # print("y_pred", y_pred)
        print(file)
        # print("Prediction:", y_pred)
        errors.append(mean_squared_error(y, y_pred))
        print("error:", mean_squared_error(y, y_pred))
        print("______________________________")
    
    plt.plot(files, errors)
    plt.xlabel("test")
    plt.ylabel("error")
    plt.show()
    return errors

lambda1 = lambda I1, I2: I1 / 2 + ((I1 / 2) ** 2 - I2) ** 0.5
lambda2 = lambda I1, I2: I1 / 2 - ((I1 / 2) ** 2 - I2) ** 0.5

C_inv_ras = lambda lambda1, lambda2: np.array([[lambda1 ** (-2), 0], [0, lambda2 ** (-2)]])

f1 = 1
f2 = lambda invariant: - 2 / invariant ** 2 

def piola_kirchgoff_2(f1, f2, C_inv, miu=6600, H=1):
    T = miu * H * (f1 * np.eye(2) + f2 * C_inv) 
    return T


def metric(T_arr, T_pred_arr):
    
    mean = []
    max = []
    for i in range(len(T_arr)):
        
        if np.any(np.isnan(T_pred_arr[i])) == True or np.any(np.isnan(T_arr[i])) == True:
            continue
        znam = 4 * norm(T_arr[i], ord="fro")
        mean.append((T_pred_arr[i][0,0] - T_arr[i][0,0] + T_pred_arr[i][1,1] - T_arr[i][1,1]) / znam )
        max.append(np.max(norm(T_pred_arr[i] - T_arr[i])) / znam)

    return np.array(mean), np.array(max)
    
def test():
    error_mean, error_max = metric(pk2_anl, pk2_pred)
    
    print(error_mean - error_max)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    # Создание первого графика
    ax1.set_title('Mean Error')
    ax1.plot(range(len(error_mean)), error_mean, label='Mean Error')

    ax2.set_title('Max Error')
    # Создание второго графика
    ax2.plot(range(len(error_mean)), error_max, label='Max Error')

    # Настройка осей и заголовка
    ax1.set_xlabel('Element')
    ax1.set_ylabel('Error')

    ax2.set_xlabel('Element')
    ax2.set_ylabel('Error')
  

    # Отображение графиков
    plt.tight_layout()
    plt.show()
if __name__ =="__main__":

    X, y = load_data(data_preprocessing=True, split=False, extended_data=True)
    model = perceptron_fit(extended_data=True,
                            model_name="biaxial",
                            save_model=True,
                            validation=False)
    validation("biaxial.joblib")
    # file = os.path.join('pretrained_models', "full_extended_dpsi2.joblib")
    # model = joblib.load(file)

    # scaler_X = MinMaxScaler()
    # scaler_y = MinMaxScaler()
    # X_scaled = scaler_X.fit_transform(X)
    # y_scaled = scaler_y.fit_transform(y)

    # X_train, X_test, y_train, y_test = \
    #             train_test_split(X, y,
    #                             test_size=0.2, random_state=2)
    # X_train, X_val, y_train, y_val = \
    #         train_test_split(X_train, y_train,
    #                         test_size=0.25, random_state=2)

    # f2_pred = scaler_y.inverse_transform(model.predict(X_val).reshape(-1, 1))

    # C_inv_arr = np.load("C_inv.npy")

    # # print(f2_pred.shape)
    # # print(C_inv_arr.shape)

    # pk2_pred = []
    # for f, C in zip(f2_pred, C_inv_arr): 
    #     pk2_pred.append(piola_kirchgoff_2(f1, f, C))
    
    # pk2_pred = np.array(pk2_pred)
    # pk2_anl = np.load("piola_kirchhoff_analytical_val.npy")
    


        
    # print("MSE not scaled =", mean_squared_error(pk2_anl, pk2_pred))


    # piola_kirchgoff_2_pred = []
    # for invariant in X_val["I2"]:
    #     piola_kirchgoff_2_pred.append(piola_kirchgoff_2(f1, model.predict(invariant)))
    
    # # model = joblib.load(os.path.join("pretrained_models",
    # #                       "full_extended_dpsi2.joblib"))

    # X, y = load_data(extended_data=True, split=False)
    # # print(y)
    # # y = y.drop(columns=['d(psi)/d(I1)'])
    # X_reg = scale_data(X)
    # y_pred = model.predict(X_reg)
    # scaler = MinMaxScaler()
    # scaler = scaler.fit(y)
    # model.set_scale_params(min=scaler.data_min_,
    #                         scale=scaler.scale_, range=scaler.data_range_)
    # model.save_model()
    # y_scaled = scale_data(y)
    # y_pred_inversed = y_pred * (y.max() - y.min()) + y.min()

    # print("y: ", y[:5])
    # print("y_scaled: ", y_scaled[:5])
    # print("y_pred: ", y_pred[:5])
    # print("y_pred inverse scaled", y_pred_inversed[:5])
    
    # print("MSE scaled =", mean_squared_error(y, y_pred))
    # print("MSE not scaled =", mean_squared_error(y, y_pred_inversed))

    # errors = validation()

'''
TODO:
- изменить скейлинг на каждую выборку отдельно => валидировать 
- 

'''
