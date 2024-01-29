#include <torch/torch.h>
#include <iostream>
#include "torch/script.h"

#include <torch/torch.h>
#include <iostream>
#include "torch/script.h"

int main() {
    // Загрузка модели
  torch::jit::script::Module MLP = torch::jit::load("../models/model_cpp.pt");

  double I1 = 0.01; double I2 = 0.02;

  // Подготовка входных данных
  torch::Tensor input_tensor = torch::tensor({{I1, I2}}, torch::kDouble);
  std::cout << "Input Tensor Size: " << input_tensor.sizes() << std::endl;

  // Инференс модели
  torch::Tensor output_tensor = MLP.forward({input_tensor}).toTensor();
  
  // Присвоение значения dpsi_dI2
  auto dpsi_dI2 = output_tensor[0][0].item<double>();

  std::cout << "result = " << dpsi_dI2 << std::endl;

    return 0;
}
