# data-driven-constitutive-modelling

Проект направлен на решениее задач на получение моделей .
The data required for this task is organized in the data folder with the following structure:

kotlin
Copy code
data
├── biaxial_one_big_hole.xlsx
├── biaxial_one_small_hole.xlsx
├── biaxial_three_different_holes.xlsx
├── biaxial_two_different_holes.xlsx
├── biaxial_two_small_holes.xlsx
├── uniaxial_one_big_hole.xlsx
├── uniaxial_one_small_hole.xlsx
├── uniaxial_three_different_holes.xlsx
├── uniaxial_two_different_holes.xlsx
└── uniaxial_two_small_holes.xlsx
Problem Statement
The code in this repository addresses the task of constructing neural correlations. The implementation includes two versions: one using scikit-learn and another using PyTorch. The project is gradually transitioning to PyTorch.

Implementation Details
Neural Network Architecture: The implemented neural network utilizes a single-layer perceptron.
Usage
To use the code, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/your-username/your-repository.git
Navigate to the project directory:

bash
Copy code
cd your-repository
Install the required dependencies:

Copy code
pip install -r requirements.txt
Run the code:

css
Copy code
python main.py
Data Validation
The project includes a data validation mechanism that compares the results with an analytical function. This ensures the accuracy of the constructed neural correlations.

Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

License
This project is licensed under the MIT License.

