import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
import torch
# from potential_zoo import May_Yin_psi

gamma = 0.2
C_inv_shear = np.array([[1 + gamma * gamma, -gamma], [-gamma, 1]])

f1 = 3300
f2 = lambda invariant: - 2 / invariant ** 2

I1_tc = lambda lam: lam ** 2 + 2.0 / lam
I2_tc = lambda lam: 2.0 * lam + 1 / lam ** 2
I1_s = lambda gam: gam ** 2 + 3.0

F_tc = lambda lam: np.eye(3) * [lam, lam ** (-0.5), lam ** (-0.5)]
F_s = lambda gam: np.array([[1., gam, 0], [0, 1., 0], [0, 0, 1.]])



def di_df():
    di1_df = 2 * f
    di2_df = 2 * (I1*f - f*f.transpose*f)
    di3_df = None
    return np.array([di1_df, di2_df, di3_df])


def di_dc():
    I = np.eye(3)
    di1_dc = I
    di2_dc = I1 * I - C
    di3_dc = I3 * C.transpose().inverse()
    return np.array([di1_dc, di2_dc, di3_dc])


def piola_kirchgoff_2(f1, f2, C_inv, miu=6600, H=1):
    T = miu * H * (f1 * np.eye(2) + f2 * C_inv)
    return np.array(T)


def normalize_data(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


def number_to_matrix12(number):
    matrix = np.zeros((3, 3))
    matrix[0, 1] = number
    return matrix


def number_to_matrix11(number):
    matrix = np.zeros((3, 3))
    matrix[0, 0] = number
    return matrix

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
    def __init__(self, path="full_data_names.txt", transform=None, psi=None, dataset_type=torch.float32, non_one=False):
        # super(Dataset, self).__init__()
        super().__init__()

        self.path = path

        if non_one:
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
            self.data = pd.read_excel(path, sheet_name="Sheet1", header=[1, 2, 3])
            self.target = None
            self.features, self.target, self.F = self.full_field_data()

            self.dpsi = None
            self.lam = None
            self.invariants = None

        # Normalize the features and target
        if transform is not None:
            self.features = transform(self.features)
            self.target = transform(self.target)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.iloc[idx, :2].values, dtype=torch.float32)
        target = torch.tensor(self.data.iloc[idx, 2:].values, dtype=torch.float32)

        if self.transform:
            features, target = self.transform(features, target)

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

    def full_field_data(self):
        brain_CR_data_TC = self.data.filter(like="CR-comten").copy()
        brain_CR_data_S = self.data.filter(like="CR-shr").copy().dropna(axis=1)

        brain_CR_data_TC.columns = brain_CR_data_TC.columns.droplevel(level=[0, 2])
        brain_CR_data_S.columns = brain_CR_data_S.columns.droplevel(level=[0, 2])

        brain_CR_data_TC["P"] = brain_CR_data_TC["P"].apply(number_to_matrix11)
        brain_CR_data_S["P"] = brain_CR_data_S["P"].apply(number_to_matrix12)

        brain_CR_data_TC['I1'] = brain_CR_data_TC[brain_CR_data_TC.columns[0]].apply(I1_tc)
        brain_CR_data_TC['I2'] = brain_CR_data_TC[brain_CR_data_TC.columns[0]].apply(I2_tc)
        brain_CR_data_TC['F'] = brain_CR_data_TC[brain_CR_data_TC.columns[0]].apply(F_tc)

        brain_CR_data_S['I1'] = brain_CR_data_S[brain_CR_data_S.columns[0]].apply(I1_s)
        brain_CR_data_S['I2'] = brain_CR_data_S[brain_CR_data_S.columns[0]].apply(I1_s)
        brain_CR_data_S['F'] = brain_CR_data_S[brain_CR_data_S.columns[0]].apply(F_s)

        I1 = pd.concat([brain_CR_data_TC['I1'], brain_CR_data_S['I1']], ignore_index=True)
        I2 = pd.concat([brain_CR_data_TC['I2'], brain_CR_data_S['I2']], ignore_index=True)
        F = pd.concat([brain_CR_data_TC['F'], brain_CR_data_S['F']], ignore_index=True)

        invariants = pd.concat([I1, I2], axis=1)
        true_stress = pd.concat([brain_CR_data_TC["P"], brain_CR_data_S["P"]], ignore_index=True)

        return invariants, true_stress, F




if __name__ == "__main__":

    data_path = r"C:\Users\Biomechanics\PycharmProjects\dd\data-driven-constitutive-modelling\data\braid_bade\CANNsBRAINdata.xlsx"

    brain_dataset = ExcelDataset(data_path)


