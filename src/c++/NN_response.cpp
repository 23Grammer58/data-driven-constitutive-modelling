#include <torch/torch.h>
#include <iostream>
#include "torch/script.h"


double normalize_features(double features){
    return (features - 4.928164958953857) / 3.797274112701416;
}

double inverse_normalize_target(double target_normalized){
    return target_normalized * 1285.8721923828125 - 1121.634033203125;
}


extern "C"{ //for simple name mangling
    int NeuralNeoHookRespFunc(double I1, double I2, double* dpsi_dI1, double* dpsi_dI2){
        
        I1 = normalize_features(I1);
        I2 = normalize_features(I2);
        
         // Загрузка модели
        torch::jit::script::Module MLP = torch::jit::load("/home/biomech/PycharmProjects/data-driven-constitutive-modelling/src/torch/pretrained_models/model_cpp.pt");

        // Подготовка входных данных
        torch::Tensor input_tensor = torch::tensor({{I1, I2}}, torch::kDouble);

        // Инференс модели
        torch::Tensor output_tensor = MLP.forward({input_tensor}).toTensor();

        // Расчет значения dpsi_dI1
        const double mu = 0.35;
        const double H = 0.07;
        // *dpsi_dI1 = mu * H / 2;
        *dpsi_dI1 = 6600;
        
        // Присвоение значения dpsi_dI2
        *dpsi_dI2 = inverse_normalize_target(output_tensor[0][0].item<double>()); 

        return 0;
    }
}
