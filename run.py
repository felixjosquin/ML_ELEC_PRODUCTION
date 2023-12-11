import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

nb_days_cnn = 4
nb_epoch = 300
hidden_layer = 40
crit = nn.MSELoss()

NB_RECORD_FOR_DAY = 12
NB_CHANNELS = 8
NB_DAYS_BY_BATCH = 20

df = pd.read_csv(
    "./data/data.csv",
    parse_dates=["Start_time", "End_time"],
)

keywords = [
    "Fioul",
    "Charbon",
    "Gaz",
    "Nucléaire",
    "Eolien",
    "Solaire",
    "Hydraulique",
    "Bioénergies",
]
print(df[keywords].max())
df[keywords] /= df[keywords].max()

traindata = (
    df.loc[df["End_time"] < "2021-12-31"]
    .drop(
        columns=[
            "Start_time",
            "End_time",
        ]
    )
    .values
)
print(traindata.shape)

assert traindata.shape[0] % NB_DAYS_BY_BATCH * NB_RECORD_FOR_DAY == 0
traindata = np.reshape(
    traindata, (-1, NB_DAYS_BY_BATCH * NB_RECORD_FOR_DAY, NB_CHANNELS)
)
nb_days_batch = (traindata.shape[1] // NB_RECORD_FOR_DAY) - nb_days_cnn

trainx = torch.tensor(traindata[:, :-NB_RECORD_FOR_DAY, :], dtype=torch.float32)
trainy = torch.tensor(
    traindata[:, nb_days_cnn * NB_RECORD_FOR_DAY :, :], dtype=torch.float32
)
trainds = torch.utils.data.TensorDataset(trainx, trainy)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)

testdata = (
    df.loc[df["Start_time"] >= "2022-01-01"]
    .drop(
        columns=[
            "Start_time",
            "End_time",
        ]
    )
    .values
)
testx = torch.tensor(np.array([testdata[:-NB_RECORD_FOR_DAY, :]]), dtype=torch.float32)
testy = torch.tensor(
    np.array([testdata[nb_days_cnn * NB_RECORD_FOR_DAY :, :]]), dtype=torch.float32
)
testds = torch.utils.data.TensorDataset(testx, testy)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)


class Mod(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.kern = nb_days_cnn * NB_RECORD_FOR_DAY
        self.cnn = nn.Conv1d(
            in_channels=NB_CHANNELS,
            out_channels=nhid,
            kernel_size=self.kern,
            stride=NB_RECORD_FOR_DAY,
        )
        self.sigmoid = nn.Sigmoid()
        self.mlp = nn.Linear(nhid, NB_RECORD_FOR_DAY * NB_CHANNELS)

    def forward(self, x):
        # x = N * L * NB_CHANNELS
        x = torch.transpose(x, 1, 2)
        x = self.cnn(x)
        y = self.sigmoid(x)
        B, H, T = y.shape
        yy = y.transpose(1, 2)
        # yy = B, T, d
        y = self.mlp(yy.view(B * T, H))
        return y.view(-1, NB_CHANNELS)


def test(mod):
    mod.train(False)
    totloss, nbatch = 0.0, 0
    for inputs, goldy in testloader:
        haty = mod(inputs)
        goldy = goldy.view(-1, NB_CHANNELS)
        loss = crit(haty, goldy)
        totloss += loss.item()
        nbatch += nb_days_batch
    totloss /= float(nbatch)
    mod.train(True)
    return totloss


def train(mod):
    optim = torch.optim.Adam(mod.parameters(), lr=0.001)
    train_loss_data, test_loss_data = [], []
    for epoch in range(nb_epoch):
        totloss, nb_days = 0.0, 0
        for inputs, goldy in trainloader:
            optim.zero_grad()
            goldy = goldy.view(-1, NB_CHANNELS)
            haty = mod(inputs)
            loss = crit(haty, goldy)
            totloss += loss.item()
            nb_days += nb_days_batch
            loss.backward()
            optim.step()
        totloss /= float(nb_days)
        testloss = test(mod)
        print(
            f"epoch {epoch+1}/{nb_epoch} | train_loss: {totloss} | test_loss: {testloss}"
        )
        train_loss_data.append(totloss)
        test_loss_data.append(testloss)
    abs = list(range(nb_epoch))
    plt.plot(abs, train_loss_data)
    plt.plot(abs, test_loss_data)
    plt.savefig("./plot/myimage.png", dpi=1000)
    print("fin")


mod = Mod(hidden_layer)
print(
    "nparms",
    sum(p.numel() for p in mod.parameters() if p.requires_grad),
    file=sys.stderr,
)
train(mod)
