import pandas as pd
import os
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
import torch


def load_data(path="biaxial_three_different_holes.xlsx", validation=False,
              split=True, extended_data=False):
    if extended_data:

        files = os.listdir(os.path.join("..", "data"))[:5]
        # print(files)
        df_list = []
        for file in files:
            df_list.append(pd.read_excel(os.path.join('data', file)))

        df = pd.concat(df_list)

    else:

        file = os.path.join('data', path)
        df = pd.read_excel(file)

    # n = df.shape[0]
    # print("DataFrame shape =", n)

    X = df.drop(columns=['d(psi)/d(I1)', 'd(psi)/d(I2)'])
    y = df.drop(columns=['d(psi)/d(I1)', 'I1', 'I2'])

    # print("X shape = ", X.shape)
    # print("y shape = ", y.shape)
    # print(X["I1"].values)

    # y = df.drop(columns=['I1', 'I2'])

    if not split:
        return X, y

    X_train, X_test, y_train, y_test = \
        random_split(X, y,
                     test_size=0.2, random_state=2)

    if not validation:
        return X_train, X_test, y_train, y_test

    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train,
                         test_size=0.25, random_state=2)

    print("train")
    print("X shape = ", X_train.shape)
    print("y shape = ", y_train.shape)
    print("Test")
    print("X shape = ", X_test.shape)
    print("y shape = ", y_test.shape)
    print("Validation")
    print("X shape = ", X_val.shape)
    print("y shape = ", y_val.shape)

    return X_train, X_test, X_val, y_train, y_test, y_val

class ExcelDataset(Dataset):
    def __init__(self, excel_files, transform=None):

        data_frames = [pd.read_excel(file) for file in excel_files]
        data = pd.concat(data_frames, ignore_index=True)
        self.data = data

        self.features = torch.tensor(data.drop(columns=['d(psi)/d(I1)', 'd(psi)/d(I2)']).values,
                                     dtype=torch.float32)
        self.target = torch.tensor(data.drop(columns=['d(psi)/d(I1)', 'I1', 'I2']).values,
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


if __name__ == "__main__":

    def normalize_data(data):
        mean = data.mean()
        std = data.std()
        return (data - mean) / std

    file_path = "../../data"
    excel_files = os.listdir(file_path)
    excel_files = [os.path.join(file_path, file) for file in excel_files][:2]

    dataset = ExcelDataset(excel_files)

    f = dataset.features
    t = dataset.target
    print(dataset.features)
    f = normalize_data(f)
    print(f)
    features_tensor = torch.tensor(f.values, dtype=torch.float32)
    target_tensor = torch.tensor(t.values, dtype=torch.float32)
    print(type(features_tensor), type(target_tensor))
    # d = Dataset()
    dataset = TensorDataset(features_tensor, target_tensor)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)




