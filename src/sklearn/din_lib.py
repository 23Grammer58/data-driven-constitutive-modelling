from  model import load_data, scale_data
import sys 
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error 
import os 
import pandas as pd

#X - I1, I2 y - dpsi/dI1, dpsi/dI2
def response():

    file = os.path.join('../../pretrained_models', "full_extended_dpsi2.joblib")

    loaded_model = joblib.load(file)

    I1 = [float(sys.argv[1])]
    I2 = [float(sys.argv[2])]
    # I1 = [float(I1)]
    # I2 = [float(I2)]

    # mean, scale = loaded_model.get_scale_params()

    X = pd.DataFrame({'I1': I1, 'I2': I2})
    
    dpsi1 = 6600
    dpsi2 = loaded_model.predict(X)
    print(dpsi2)

    # y_pred_inv_scaled = y_pred * scale + mean

        

    with open("test/file.txt", "w") as file:
        file.write(str(dpsi1))
        file.write(" ")
        file.write(str(dpsi2))


    # X, y = load_data("uniaxial_three_different_holes.xlsx", False)
    
    # X_reg = scale_data(X)
    # y_reg = scale_data(y)
    # y_pred = scale_data(loaded_model.predict(X_reg), True)
    
    # scaler = StandardScaler()
    # scaler = scaler.fit(y)
    # y_pred_inversed = scaler.inverse_transform(y_pred)

    # print("y_reg:", y_reg)
    # print("y_pred: ", y_pred[:5])
    # print("y_pred inverse scaled", y_pred_inversed[:5])

    # print("MSE scaled =", mean_squared_error(scale_data(y), y_pred))
    # print("MSE not scaled =", mean_squared_error(y, y_pred_inversed))

    return 0

if __name__ =="__main__":

    response()