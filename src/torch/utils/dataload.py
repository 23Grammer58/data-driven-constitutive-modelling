import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
import torch

gamma = 0.2
C_inv_shear = np.array([[1 + gamma * gamma, -gamma], [-gamma, 1]])

f1 = 3300
f2 = lambda invariant: - 2 / invariant ** 2


def piola_kirchgoff_2(f1, f2, C_inv, miu=6600, H=1):
    T = miu * H * (f1 * np.eye(2) + f2 * C_inv)
    return np.array(T)


def Mooney_Rivlin_psi(I1, I2):
    return 10 * (I1 - 3) + 5 * (I2 - 3)


def NeoHookean_psi(I1, I2):
    return 10 * (I1 - 3)


def normalize_data(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


class ExcelDataset(Dataset):
    def __init__(self, path="full_data_names.txt", transform=None, psi=Mooney_Rivlin_psi):
        # super(Dataset, self).__init__()
        super().__init__()

        self.path = path

        if path is not None:
            self.data = self.read_from_file()

        self.features = torch.tensor(self.data.drop(columns=['d(psi)/d(I1)', 'd(psi)/d(I2)']).values,
                                     dtype=torch.float32)
        self.dpsi = torch.tensor(self.data.drop(columns=['d(psi)/d(I1)', 'I1', 'I2']).values,
                                   dtype=torch.float32)
        # self.target = torch.tensor(NeoHookean_psi(self.features['I1'], self.features['I2']))
        # self.target = self.data[""]
        self.target = torch.tensor(self.data.apply(
            lambda row: psi(row['I1'], row['I2']), axis=1).values, dtype=torch.float32).unsqueeze(-1)

        # self.target = torch.tensor(self.data.apply(
        #     lambda row: с(row['I1'], row['I2']), axis=1).values, dtype=torch.float32).unsqueeze(-1)

        if transform is not None:
            self.features = normalize_data(self.features)
            self.target = normalize_data(self.target)
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
        excel_files = []

        with open(self.path, "r") as file:
            for table_name in file:
                excel_files.append(table_name[:-1])

        data_frames = [pd.read_excel(file) for file in excel_files]

        return pd.concat(data_frames, ignore_index=True)


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":

    def normalize_data(data):
        mean = data.mean()
        std = data.std()
        return (data - mean) / std


    # data = list(range(100))
    # dataset = SimpleDataset(data)
    # # Задаем размеры для каждой части (train_size, test_size)
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    #
    # # Используем random_split для разделения dataset
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    #
    # # Выводим размеры полученных наборов данных
    # print(f"Размер обучающей выборки: {len(train_dataset)}")
    # print(f"Размер тестовой выборки: {len(test_dataset)}")
    # for idx in range(len(test_dataset)):
    #     data_point = test_dataset[idx]
    #     print(f"Элемент {idx + 1}: {data_point}")    # file_path = "../../data"
    # excel_files = os.listdir(file_path)
    # excel_files = [os.path.join(file_path, file) for file in excel_files][:2]

    # excel_files = []
    # f = open(r"full_data_names.txt")
    # for lines in f:
    #     excel_files.append(lines[:-1])
    dataset = ExcelDataset("../one_data_names.txt")

    f = dataset.features
    t = dataset.target
    print(dataset.features)
    print("фичи: \n", f)

    # f = normalize_data(f)
    # t = normalize_data(t)
    print("нормализованные фичи: \n", f)
    print("тип фичей -", type(f))
    print("тип target -", type(t))

    # t = torch.stack(list(map(lambda x: torch.unsqueeze(x, 0), t)))
    print(t)

    dataset = TensorDataset(f, t)


    # train_size = int(0.9 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # #
    # print("размер трейна =", len(train_dataset))
    # print("размер теста =", len(test_dataset))
    # # print("трейн: \n", train_dataset[10:20])
    # # print("тест: \n", test_dataset[10:20])
    #
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    #
    # # print("тест: \n", test_dataset[2].values())
    # # for inputs, targets in test_dataset: