from SPE_model import TransformerBiasNet
import math
import torch

torch.manual_seed(0)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.metrics import mean_squared_error

from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, Dataset

import matplotlib as mpl
import matplotlib.pyplot as plt

# Plot
def figure(x_act, x_pre, y_act, y_pre):
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(x_act, y_act, label='Actual')
    ax.plot(x_pre, y_pre, label='Prediction')
    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    # ax.set_xlim(x[0], x[-1])
    ax.grid()
    fig.tight_layout(pad=0)
    return fig


def plot_two_curves(x, y1, y2):
    plt.plot(x, y1, label='curve 1')
    plt.plot(x, y2, label='curve 2')
    plt.legend()


# GPU device setting
device = torch.device("cuda:0")

# Constitutive data parameters
# row_undrain = 1401
column_undrain_input = column_input = 8
column_undrain_output = column_output = 2
column_undrain_total = 10


# Model parameters
dim_in = 3
in_loc = 7
dim_s = 4
s_loc = 4
# dim_model = 64
# dim_feedforward = 128
# num_layers = 1
# num_heads = 1
dim_model = 128
dim_feedforward = 128
num_layers = 1
num_heads = 1
dim_mlp_s = 16
warmup_epochs = 200
batch_size = 4096

# dim_mlp_s = 40
max_len = 2
dim_out = 3
# warmup_epochs = 150
epoch = 1000
dropout = 0.1
# batch_size = 1024





if __name__ == '__main__':
    # Data
    Data_train_import = pd.read_excel("training dataset (EX).xlsx", header=None)
    Data_test_import = pd.read_excel("testing dataset (EX).xlsx", header=None)
    Data_train_real = Data_train_import.iloc[1:, :].values[:, :].astype("float64")
    Data_test_real = Data_test_import.iloc[1:, :].values[:, :].astype("float64")

    # Data Normalization
    Scaler_train = MinMaxScaler(feature_range=(0, 1))
    Data_train = Scaler_train.fit_transform(Data_train_real)
    Data_test = Scaler_train.fit_transform(Data_test_real)

    X_train_scaled = Data_train[:, :in_loc]
    X_test_scaled = Data_test[:, :in_loc]
    y_train_scaled = Data_train[:, in_loc:]
    y_test_scaled = Data_test[:, in_loc:]

    # 真实输出值（未经过归一化）
    y_act_train = Data_train_real[:, in_loc:].astype("float64")
    p_train_act = y_act_train[:, :1]
    q_train_act = y_act_train[:, 1:2]
    s1_train_act = y_act_train[:, 2:]
    y_act_test = Data_test_real[:, in_loc:].astype("float64")
    p_test_act = y_act_test[:, :1]
    q_test_act = y_act_test[:, 1:2]
    s1_test_act = y_act_test[:, 2:]

    num_train = len(Data_train) // max_len
    num_test = len(Data_test) // max_len
    end_num_train = num_train * max_len
    end_num_test = num_test * max_len
    Data_train = Data_train[:end_num_train, :]
    Data_test = Data_test[:end_num_test, :]
    Data_train_real = Data_train_real[:end_num_train, :]
    Data_test_real = Data_test_real[:end_num_test, :]
    data_train = Data_train.reshape([num_train, max_len, column_undrain_total])
    data_test = Data_test.reshape([num_test, max_len, column_undrain_total])


    data_train_X = data_train[:, :, s_loc:in_loc]
    sf_train = data_train[:, :, :s_loc]
    data_train_y = data_train[:, :, in_loc:]
    data_test_X = data_test[:, :, s_loc:in_loc]
    sf_test = data_test[:, :, :s_loc]
    data_test_y = data_test[:, :, in_loc:]
    X_train_tensor = torch.tensor(data_train_X, dtype=torch.float32).to(device)
    sf_train_tensor = torch.tensor(sf_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(data_test_X, dtype=torch.float32).to(device)
    sf_test_tensor = torch.tensor(sf_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(data_train_y, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(data_test_y, dtype=torch.float32).to(device)



    data_train = torch.from_numpy(data_train.astype(np.float32))
    data_test = torch.from_numpy(data_test.astype(np.float32))
    train_dataset = TensorDataset(data_train)
    test_dataset = TensorDataset(data_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Initialization
    def initialize_weights(m):
        """
        Weight initialization
        """
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.kaiming_uniform_(m.weight.data)

    class CustomLRScheduler(optim.lr_scheduler.LRScheduler):
        def __init__(self, optimizer, warmup_epochs, T_max, lr_init, last_epoch=-1):
            self.warmup_epochs = warmup_epochs
            self.T_max = T_max
            self.lr_init = lr_init
            super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch < self.warmup_epochs:
                # 线性预热增加学习率
                lr = [self.lr_init * (self.last_epoch + 1) / self.warmup_epochs for _ in self.base_lrs]
            else:
                # 余弦退火调整学习率
                ep = self.last_epoch - self.warmup_epochs
                lr = [0.5 * base_lr * (1 + math.cos(math.pi * ep / (self.T_max - self.warmup_epochs)))
                      for base_lr in self.base_lrs]
            return lr


    model = TransformerBiasNet(dim_mlp_s=dim_mlp_s,
                               num_layers=num_layers,
                               dim_in=dim_in,
                               dim_model=dim_model,
                               num_heads=num_heads,
                               dim_feedforward=dim_feedforward,
                               dropout=dropout,
                               dim_s=dim_s,
                               seq_len=max_len,
                               dim_out=dim_out
                               )

    model.apply(initialize_weights)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-8)
    scheduler = CustomLRScheduler(optimizer, warmup_epochs=warmup_epochs, T_max=epoch, lr_init=1e-3)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    Loss_train = np.array(3).reshape(1)
    Loss_test = np.array(3).reshape(1)

    for i in range(epoch):
        print(f"Epochs = {i+1}")
        # Train
        for index, item in enumerate(train_loader):
            # type(item) = "list"
            item = torch.from_numpy(np.squeeze(np.array(item), 0)).to(device)
            # item.device = cuda:0
            optimizer.zero_grad()
            output_predict_train = model(src=item[:, :, s_loc:in_loc], sf=item[:, :, :s_loc])
            output = np.squeeze(item[:, :, in_loc:], axis=1)
            loss = criterion(output_predict_train, output)
            if index == 0:
                loss_train = loss.cpu().detach().numpy().reshape(1)
                Loss_train = np.concatenate((Loss_train, loss_train), 0)
            loss.backward()
            optimizer.step()
            scheduler.step()


        model.eval()
        with torch.no_grad():
            for index, item in enumerate(test_loader):
                item = torch.from_numpy(np.squeeze(np.array(item), 0)).to(device)
                output_predict_test = model(src=item[:, :, s_loc:in_loc], sf=item[:, :, :s_loc])
                output_test = np.squeeze(item[:, :, in_loc:], axis=1)
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(output_predict_test, output_test)
                if index == 0:
                    loss_test = loss.cpu().detach().numpy().reshape(1)
                    Loss_test = np.concatenate((Loss_test, loss_test), 0)

    Loss = np.concatenate((Loss_train.reshape(-1, 1), Loss_test.reshape(-1, 1)), axis=1)


    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train_tensor, sf_train_tensor).reshape(-1, dim_out).cpu().numpy()
        y_test_pred = model(X_test_tensor, sf_test_tensor).reshape(-1, dim_out).cpu().numpy()
        prediction_train = np.concatenate((X_train_scaled, y_train_pred), axis=1)
        prediction_test = np.concatenate((X_test_scaled, y_test_pred), axis=1)


    inversed_out_train = Scaler_train.inverse_transform(prediction_train)
    inversed_out_test = Scaler_train.inverse_transform(prediction_test)

    def calculate_metrics(target, predict):
        mae = (abs(target - predict)).mean()
        SSE = sum((target - predict) ** 2)
        SST = sum((target - (target).mean()) ** 2)
        r2 = 1 - (SSE / SST)
        r2 = r2[0]
        non_zero_indices = target != 0
        y_true_non_zero = target[non_zero_indices]
        y_pred_non_zero = predict[non_zero_indices]
        mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100
        return mae, mape, r2

    # Output predictions
    train_p = inversed_out_train[:, in_loc:in_loc+1]
    train_q = inversed_out_train[:, in_loc+1:in_loc+2]
    train_s1 = inversed_out_train[:, -1].reshape(-1, 1)
    test_p = inversed_out_test[:, in_loc:in_loc+1]
    test_q = inversed_out_test[:, in_loc+1:in_loc+2]
    test_s1 = inversed_out_test[:, -1].reshape(-1, 1)

    print(train_p.dtype)
    print(p_train_act.dtype)
    train_mae_p, train_mape_p, train_r2_p = calculate_metrics(p_train_act, train_p)
    train_mae_q, train_mape_q, train_r2_q = calculate_metrics(q_train_act, train_q)
    train_mae_s1, train_mape_s1, train_r2_s1 = calculate_metrics(s1_train_act, train_s1)
    test_mae_p, test_mape_p, test_r2_p = calculate_metrics(p_test_act, test_p)
    test_mae_q, test_mape_q, test_r2_q = calculate_metrics(q_test_act, test_q)
    test_mae_s1, test_mape_s1, test_r2_s1 = calculate_metrics(s1_test_act, test_s1)

    metrics = np.array([[train_mae_p, train_mape_p, train_r2_p],
                        [train_mae_q, train_mape_q, train_r2_q],
                        [train_mae_s1, train_mape_s1, train_r2_s1],
                        [test_mae_p, test_mape_p, test_r2_p],
                        [test_mae_q, test_mape_q, test_r2_q],
                        [test_mae_s1, test_mape_s1, test_r2_s1]], dtype=float)

