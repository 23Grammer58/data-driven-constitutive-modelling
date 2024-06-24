import copy

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
import torch
from scipy.sparse import csr_matrix

class ExcelDataset(Dataset):
    """
    A custom PyTorch Dataset for loading mechanical experiments data from Excel files.

    Args:
        path (str): The path to the file containing the names of the Excel files to load.
        transform (callable, optional): Optional transform to be applied on a sample.
        psi (callable): The function used to calculate the target values from the features.

    Attributes:
        dataset (pandas.DataFrame): The concatenated dataframe containing the data from all Excel files.
        features (torch.Tensor): The features extracted from the dataset.
        dpsi (torch.Tensor): The derivatives of the target with respect to the features.
        target (torch.Tensor): The target values calculated from the features using the provided function `psi`.
        transform (callable): The transform to be applied to the features and target.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the features and target for the item at index `idx`.
        read_from_file(): Reads the Excel files specified in the `path` file and returns the concatenated dataframe.
    """
    def __init__(
                self,
                path="full_data_names.txt",
                transform=None,
                psi=None,
                dataset_type=torch.float32,
                numerical_data=False,
                device = "cuda",
                batch_size=1
    ):
        super().__init__()

        self.device = device
        if numerical_data:
            self.data = self.read_from_file()

            self.features = torch.tensor(self.data.drop(columns=['d(psi)/d(I1)', 'd(psi)/d(I2)']).values,
                                     dtype=dataset_type)
            self.dpsi = torch.tensor(self.data.drop(columns=['d(psi)/d(I1)', 'I1', 'I2']).values,
                                   dtype=dataset_type)
        # self.target = torch.tensor(NeoHookean_psi(self.features['I1'], self.features['I2']))
        # self.target = self.data[""]
            self.target = torch.tensor(self.data.apply(
                lambda row: psi(row['I1'], row['I2']), axis=1).values, dtype=torch.float32).unsqueeze(-1)
        else:
            self.path = path
            self.data = self.full_field_data(path)
            self.batch_size = batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = copy.deepcopy([*self.data.iloc[idx]])
        # features = [*self.data.iloc[idx]][0:2]
        target = features.pop(1)
        # target = values.pop(1)
        # features = values.remove(target)

        # if self.transform:
        #     features, target = self.transform(features, target)
        return features, target


    # def __str__(self, ):

    def read_from_file(self):
        # Read the Excel files and concatenate them into a single dataframe

        excel_files = []
        if len(excel_files) == 1:
            return pd.read_excel(excel_files[0])

        with open(self.path, "r") as file:
            for table_name in file:
                excel_files.append(table_name[:-1])

        data_frames = [pd.read_excel(file) for file in excel_files]

        return pd.concat(data_frames, ignore_index=True)

    def full_field_data(self, path):
        all_data = pd.read_excel(path, sheet_name="Sheet1", header=[1, 2, 3])
        brain_CR_TC_data = all_data.filter(like="CR-comten").copy()
        brain_CR_S_data  = all_data.filter(like="CR-shr").copy().dropna(axis=1)

        brain_CR_TC_data.columns = brain_CR_TC_data.columns.droplevel(level=[0, 2])
        brain_CR_S_data.columns  = brain_CR_S_data.columns.droplevel(level=[0, 2])

        mechanical_variables = {
            "I1": [I1_tc, I1_s],
            "I2": [I2_tc, I1_s],
            "F":  [F_tc, F_s],
            # "exp_type": [(lambda x: 1), (lambda x: 0)] # 1 - torsion&compression, 0 - shear
            # "torsion_compression": (lambda x: 1)
        }

        # calculate I1, I2, F from lambda (torsion&compression and shear)
        for variable in mechanical_variables.keys():
            func_calc = mechanical_variables.get(variable)
            brain_CR_TC_data[variable] = brain_CR_TC_data["lambda"].apply(func_calc[0])
            brain_CR_S_data[variable]  = brain_CR_S_data["gamma"].apply(func_calc[1])
            # I1 = pd.concat([brain_CR_TC_data[variable], brain_CR_S_data[variable]], ignore_index=True)
        brain_CR_S_data["lambda"] = brain_CR_S_data.pop("gamma")
        brain_CR_TC_data["exp_type"] = [0.5 if i < len(brain_CR_TC_data) / 2 else 1.5  for i in range(len(brain_CR_TC_data))]
        brain_CR_S_data["exp_type"] = [1.  for i in range(len(brain_CR_S_data))]
        data = pd.concat([brain_CR_TC_data, brain_CR_S_data], ignore_index=True)
        return data

    def to_tensor(self):
        for column in self.data.columns:
            self.data[column] = self.data[column].apply(
                lambda x: torch.tensor(x, device=self.device, dtype=dataset_type))


class BrainDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.astype(float)  # Преобразуем данные в числовой тип

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        Stretch = torch.tensor(self.data.iloc[idx]['stretch'], dtype=torch.float32)
        Gamma = torch.tensor(self.data.iloc[idx]['gamma'], dtype=torch.float32)
        Stress_UT = torch.tensor(self.data.iloc[idx]['stress_ut'], dtype=torch.float32)
        Stress_SS = torch.tensor(self.data.iloc[idx]['stress_ss'], dtype=torch.float32)
        return Stretch, Gamma, Stress_UT, Stress_SS


    def __getitem__(self, idx):
        all_items = [*self.data.iloc[idx]]
        features = all_items[0], all_items[2]
        target = all_items[1], all_items[3]
        # target = values.pop(1)
        # features = values.remove(target)

        # if self.transform:
        #     features, target = self.transform(features, target)
        return features, target


if __name__ == "__main__":
    all_data_path = r'C:\Users\User\PycharmProjects\data-driven-constitutive-modelling\data\brain_bade/CANNsBRAINdata.xlsx'
    xls = pd.ExcelFile(all_data_path)

    # Load the first sheet
    df_sheet1 = pd.read_excel(xls, sheet_name='Sheet1')

    # Clean the first sheet by removing unnecessary rows and renaming columns
    df_sheet1_cleaned = df_sheet1.iloc[3:].reset_index(drop=True)

    # Keep only the relevant columns and rename them
    relevant_columns = df_sheet1_cleaned.iloc[:, :4]
    relevant_columns.columns = ['stretch', 'stress_ut', 'gamma', 'stress_ss']
    brain_data = relevant_columns

    dataset = BrainDataset(brain_data)

    print(dataset[1])
"""
TODO:
1) реализовать метод __str__ в зависимости от предоставлемых данных

"""
