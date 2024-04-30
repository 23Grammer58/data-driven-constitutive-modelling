import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
import torch
from scipy.sparse import csr_matrix

# from potential_zoo import May_Yin_psi
dataset_type = torch.float32
gamma = 0.2
C_inv_shear = np.array([[1 + gamma * gamma, -gamma], [-gamma, 1]])

f1 = 3300
f2 = lambda invariant: - 2 / invariant ** 2

I1_tc = lambda lam: lam ** 2 + 2.0 / lam
I2_tc = lambda lam: 2.0 * lam + 1 / lam ** 2
I1_s = lambda gam: gam ** 2 + 3.0
                                     # ([lam, 0, 0], [0, lam ** (-0.5), 0], [0, 0, lam ** (-0.5)])
F_tc = lambda lam: torch.tensor(([lam, 0, 0], [0, lam ** (-0.5), 0], [0, 0, lam ** (-0.5)]), dtype=dataset_type)
F_s = lambda gam: torch.tensor(([1., gam, 0], [0, 1., 0], [0, 0, 1.]), dtype=dataset_type)


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
    matrix = torch.zeros(3, 3, dtype=torch.float32)
    matrix[0, 1] = number
    # return csr_matrix(matrix)
    return matrix


def number_to_matrix11(number):
    matrix = torch.zeros(3, 3, dtype=torch.float32)
    matrix[0, 0] = number
    # return csr_matrix(matrix)
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
    def __init__(
                self,
                path="full_data_names.txt",
                transform=None,
                psi=None,
                dataset_type=torch.float32,
                multiple_experiments=False,
                device = "cuda"
    ):
        # super(Dataset, self).__init__()
        super().__init__()

        # self.path = path
        self.device = device
        if multiple_experiments:
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
            # self.features = [*self.data]
            # self.target = [*self.data]
            # self.features, self.target, self.F = self.full_field_data(path)
            # self.invariants = self.data["invariants"]
            # self.lam = self.data["lambda"]
            # self.target = self.data["P"]
            # self.features = self.data["F"]
            # self.features = torch.(self.features.values)
            # for value in self.F.values:
            #     value = torch.from_numpy(value)
            # self.F = torch.from_numpy(self.F.values)
            # self.target = torch.tensor(self.target.values, dtype=dataset_type)
            # for value in self.target.values:
            #     value = torch.from_numpy(value)

            # self.dpsi = None
            # self.lam = None
            # self.invariants = None

        # Normalize the features and target
        # if transform is not None:
        #     self.features = transform(self.features)
        #     self.target = transform(self.target)
        # self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # # lam = torch.tensor(self.data.iloc[idx, 0], dtype=torch.float32, device=self.device)
        # F = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32, device=self.device)
        # features = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32, device=self.device)
        # target = torch.tensor(self.data.iloc[idx, 3], dtype=torch.float32, device=self.device)
        # markers = torch.tensor(self.data.iloc[idx, 4], dtype=torch.float32, device=self.device)
        # features = [*self.data.iloc[idx]][2:3]
        # target = [*self.data.iloc[idx]][1]
        # if self.transform:
        #     features, target = self.transform(features, target)

        return [*self.data.iloc[idx]]


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
        
        # # calculate 1 stress tensor PK
        # brain_CR_TC_data["P"] = brain_CR_TC_data["P"].apply(number_to_matrix11)
        # brain_CR_S_data["P"]  = brain_CR_S_data["P"].apply(number_to_matrix12)


        mechanical_variables = {
            "I1": [I1_tc, I1_s],
            "I2": [I2_tc, I1_s],
            "F":  [F_tc, F_s],
            "exp_type": [(lambda x: 1), (lambda x: 0)] # 1 - torsion&compression, 0 - shear
        }

        # calculate I1, I2, F from lambda (torsion&compression and shear)
        for variable in mechanical_variables.keys():
            func_calc = mechanical_variables.get(variable)
            brain_CR_TC_data[variable] = brain_CR_TC_data["lambda"].apply(func_calc[0])
            brain_CR_S_data[variable]  = brain_CR_S_data["gamma"].apply(func_calc[1])
            # I1 = pd.concat([brain_CR_TC_data[variable], brain_CR_S_data[variable]], ignore_index=True)
        brain_CR_S_data["lambda"] = brain_CR_S_data.pop("gamma")
        data = pd.concat([brain_CR_TC_data, brain_CR_S_data], ignore_index=True)

        print(*data.iloc[1])
        # lambdas = brain_CR_TC_data.columns[0]
        # brain_CR_TC_data['I1']       = brain_CR_TC_data["lambda"].apply(I1_tc)
        # brain_CR_TC_data['I2']       = brain_CR_TC_data["lambda"].apply(I2_tc)
        # brain_CR_TC_data['F']        = brain_CR_TC_data["lambda"].apply(F_tc)
        # brain_CR_TC_data['exp_type'] = brain_CR_TC_data["lambda"] * 0 + 1
        #
        # print(brain_CR_TC_data['I1'] == brain_CR_TC_data['I1_new'])
        # # calculate I1, I2, F from gamma (shear)
        # gammas = brain_CR_S_data.columns[0]
        # brain_CR_S_data['I1']       = brain_CR_S_data["gamma"].apply(I1_s)
        # brain_CR_S_data['I2']       = brain_CR_S_data["gamma"].apply(I1_s)
        # brain_CR_S_data['F']        = brain_CR_S_data["gamma"].apply(F_s)
        # brain_CR_S_data['exp_type'] = brain_CR_S_data["gamma"] * 0
        #
        # lam = pd.concat([brain_CR_TC_data["lambda"], brain_CR_S_data["gamma"]], ignore_index=True)
        # I1  = pd.concat([brain_CR_TC_data['I1'],     brain_CR_S_data['I1']], ignore_index=True)
        # I2  = pd.concat([brain_CR_TC_data['I2'],     brain_CR_S_data['I2']], ignore_index=True)
        # F   = pd.concat([brain_CR_TC_data['F'],      brain_CR_S_data['F']], ignore_index=True)
        # # C = F.apply(lambda x: x.t().matmul(x))
        #
        # invariants = pd.Series(zip(torch.tensor(I1, dtype=dataset_type, device=self.device),
        #                            torch.tensor(I2, dtype=dataset_type, device=self.device)))
        # invariants.name = "invariants"
        #
        # markers = pd.concat([brain_CR_TC_data["exp_type"], brain_CR_S_data["exp_type"]], ignore_index=True, axis=0)
        #
        # true_stress = pd.concat([brain_CR_TC_data["P"], brain_CR_S_data["P"]], ignore_index=True, axis=0)
        # data = pd.concat((lam, F, invariants, true_stress, markers), axis=1)
        # data = data.rename(columns={0: 'lambda'})
        # print(data.iloc[1])
        return data


if __name__ == "__main__":
    data_path = r"C:\Users\drani\dd\data-driven-constitutive-modelling\data\brain_bade\CANNsBRAINdata.xlsx"

    brain_dataset = ExcelDataset(data_path, device="cpu")
    print(brain_dataset[10])
    # data = brain_dataset.data

    # lam, F, features, target = data
    # print(type(lam))

    # f = brain_dataset.features
    # t = brain_dataset.target
    #
    # print(t)

    # F = F_tc(0.9)
    # C = F.t() @ F
    # print(C)
    # print(np.linalg.eigvals(C))

    # print(a.t().inverse())
    # print(a.dim())
    # print(b)
    # print(b.dim())

"""
TODO:
1) реализовать метод __str__ в зависимости от предоставлемых данных

"""
