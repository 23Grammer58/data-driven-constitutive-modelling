#include <iostream>
#include <vector>
#include <joblib.hpp>

int main() {
    // Load the model from file
    joblib::load("pretrained_models/full_extended_dpsi2.pkl", clf);

    // Define test data
    std::vector<std::vector<float>> X_test = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    // Make predictions on test data
    std::vector<float> y_pred = clf.predict(X_test);

    // Print predictions
    for (int i = 0; i < y_pred.size(); i++) {
        std::cout << "Prediction " << i << ": " << y_pred[i] << std::endl;
    }

    return 0;
}
