import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
import torch


class ExcelDataset(Dataset):
    def __init__(self, path="full_data_names.txt", transform=None):

        self.path = path

        if path is not None:
            self.data = self.read_from_file()

        self.features = torch.tensor(self.data.drop(columns=['d(psi)/d(I1)', 'd(psi)/d(I2)']).values,
                                     dtype=torch.float32)
        self.target = torch.tensor(self.data.drop(columns=['d(psi)/d(I1)', 'I1', 'I2']).values,
                                   dtype=torch.float32)
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


if __name__ == "__main__":

    def normalize_data(data):
        mean = data.mean()
        std = data.std()
        return (data - mean) / std


    # file_path = "../../data"
    # excel_files = os.listdir(file_path)
    # excel_files = [os.path.join(file_path, file) for file in excel_files][:2]

    # excel_files = []
    # f = open(r"full_data_names.txt")
    # for lines in f:
    #     excel_files.append(lines[:-1])
    dataset = ExcelDataset("one_data_names.txt.txt")

    f = dataset.features
    t = dataset.target
    print(dataset.features)
    print("фичи: \n", f)

    f = normalize_data(f)
    t = normalize_data(t)
    print("нормализованные фичи: \n", f)
    print("тип фичей -", type(f))

    dataset = TensorDataset(f, t)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print("размер трейна =", len(train_dataset))
    print("размер теста =", len(test_dataset))
    print("трейн: \n", train_dataset[10:20])
    print("тест: \n", test_dataset[10:20])

    # for inputs, targets in test_dataset: