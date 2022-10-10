import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#import datetime

# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader


#excel_filename = "/data_for_climate_prediction.xlsx"
#excel_filename = "/learn_DLML/data_for_climate_prediction.xlsx"
excel_filename = "\learn_DLML\data_for_climate_prediction.xlsx"
excel_location = os.getcwd() + excel_filename
df = pd.read_excel(excel_location, sheet_name='Data')

# df = pdr.DataReader('005930.KS', 'yahoo', start, end)
print(df)
print("shape:", df.shape)
print("type:", type(df))
df.head(5)
df.tail(5)
# df.Close.plot(grid=True)
df.plot(grid=True)
X = df.iloc[:, :9]
# X = df.drop(columns='Volume')
y = df.iloc[:, 8:9]

print("X:", X)
print("y:", y)

"""
학습이 잘되기 위해 데이터 정규화 
StandardScaler	각 특징의 평균을 0, 분산을 1이 되도록 변경
MinMaxScaler	최대/최소값이 각각 1, 0이 되도록 변경
"""

mm = MinMaxScaler()
ss = StandardScaler()

X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y)

# Train Data
X_train = X_ss[:4032, :]
X_test = X_ss[4032:, :]

# Test Data 
"""
( 굳이 없어도 된다. 하지만 얼마나 예측데이터와 실제 데이터의 정확도를 확인하기 위해 
from sklearn.metrics import accuracy_score 를 통해 정확한 값으로 확인할 수 있다. )
"""
y_train = y_mm[:4032, :]
y_test = y_mm[4032:, :]

print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)

"""
torch Variable에는 3개의 형태가 있다. 
data, grad, grad_fn 한 번 구글에 찾아서 공부해보길 바랍니다. 
"""
X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))

X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device


print(torch.cuda.get_device_name(0))


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)  # internal state
        # Propagate input through LSTM

        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output

        return out


num_epochs = 30000  # 1000 epochs
learning_rate = 0.00001  # 0.001 lr

# input_size = 5  # number of features
# hidden_size = 2  # number of features in hidden state
input_size = 9  # number of features
hidden_size = 2  # number of features in hidden state
num_layers = 1  # number of stacked lstm layers

num_classes = 1  # number of output classes

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]).to(device)

loss_function = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)  # adam optimizer

for epoch in range(num_epochs):
    outputs = lstm1.forward(X_train_tensors_final.to(device))  # forward pass
    optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

    # obtain the loss function
    loss = loss_function(outputs, y_train_tensors.to(device))

    loss.backward()  # calculates the loss of the loss function

    optimizer.step()  # improve from loss, i.e backprop
    if epoch % 500 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

# df_X_ss = ss.transform(df.drop(columns='Volume'))
# df_y_mm = mm.transform(df.iloc[:, 5:6])

df_X_ss = ss.transform(df.iloc[:, :9])
df_y_mm = mm.transform(df.iloc[:, 8:9])

df_X_ss = Variable(torch.Tensor(df_X_ss))  # converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm))
# reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

train_predict = lstm1(df_X_ss.to(device))  # forward pass
data_predict = train_predict.data.detach().cpu().numpy()  # numpy conversion
dataY_plot = df_y_mm.data.numpy()

data_predict = mm.inverse_transform(data_predict)  # reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)

dataY_plot = dataY_plot[4032:]
data_predict = data_predict[4032:]

plt.figure(figsize=(10, 6))  # plotting
# plt.axvline(x=3600, c='r', linestyle='--')  # size of the training set

plt.plot(dataY_plot, label='Actual Data')  # actual plot
plt.plot(data_predict, label='Predicted Data')  # predicted plot
plt.title('Greenhouse Climate Prediction')
plt.legend()
plt.show()
