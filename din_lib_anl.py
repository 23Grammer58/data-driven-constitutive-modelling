from  model import load_data, scale_data
import sys 
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error 
import os 
import pandas as pd

#X - I1, I2 y - dpsi/dI1, dpsi/dI2
def response():

    I1 = float(sys.argv[1])
    I2 = float(sys.argv[2])
    
    dpsi1 = 6600 / 2 
    dpsi2 = - 3300 / I2 ** 2 

    with open("file.txt", "w") as file:
        file.write(str(dpsi1))
        file.write(" ")
        file.write(str(dpsi2))

if __name__ =="__main__":

    response()