import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y


def get_data(year, month, day, df):
    indexs = []
    if len(day) == 1 and len(str(month)) == 1:
        for index, column in df.iterrows():
            if str(index)[:10] == f'{year}-0{month}-0{day}':
                indexs.append(index)
    elif len(day) == 1 and len(str(month)) == 2:
        for index, column in df.iterrows():
            if str(index)[:10] == f'{year}-{month}-0{day}':
                indexs.append(index)
    elif len(day) == 2 and len(str(month)) == 1:
        for index, column in df.iterrows():
            if str(index)[:10] == f'{year}-0{month}-{day}':
                indexs.append(index)
    else:
        for index, column in df.iterrows():
            if f'{year}-{month}-{day}' == str(index)[:10]:
                indexs.append(index)
    df = df.loc[indexs[0]:indexs[-1]]
    return df


def data_loader(X, y):
    scaler = MinMaxScaler()
    X_test_arr = scaler.fit_transform(X)
    y_test_arr = scaler.fit_transform(y)
    test_features = torch.Tensor(X_test_arr)
    test_targets = torch.Tensor(y_test_arr)
    test = TensorDataset(test_features, test_targets)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    return test_loader_one
