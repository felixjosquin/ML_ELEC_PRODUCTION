import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

###### HYPERPARAMETER ######
nb_epoch = 500
lr = 0.0001
hidden_layer = 120
nb_days_cnn = 6
nb_days_by_batch = 73  # 1 5 3 15 73 219 365 1095 (values possible)
#############################

NB_RECORD_FOR_DAY = 48
NB_CHANNELS = 8


def crit(output, target):
    # output = nb_days * ( NB_RECORD_FOR_DAY x NB_CHANNELS)
    loss_vector = (output - target) ** 2
    loss_vector = torch.sum(loss_vector, dim=1)
    return torch.mean(loss_vector)


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
df_max = df[keywords].max()
df[keywords] /= df_max[keywords]

traindata = np.array(
    [
        df.loc[df["End_time"] <= "2021-12-31"]
        .drop(
            columns=[
                "Start_time",
                "End_time",
            ]
        )
        .values
    ]
)
traindata = np.reshape(
    traindata,
    (
        traindata.shape[1] // (nb_days_by_batch * NB_RECORD_FOR_DAY),
        nb_days_by_batch * NB_RECORD_FOR_DAY,
        NB_CHANNELS,
    ),
)

trainx = torch.tensor(traindata[:, :-NB_RECORD_FOR_DAY, :], dtype=torch.float32)
trainy = torch.tensor(
    traindata[:, nb_days_cnn * NB_RECORD_FOR_DAY :, :], dtype=torch.float32
)
nb_day_batch_train = trainy.shape[1] // NB_RECORD_FOR_DAY
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
nb_day_batch_test = testy.shape[1] // NB_RECORD_FOR_DAY
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
        self.reLU = nn.ReLU()
        self.mlp = nn.Linear(nhid, NB_RECORD_FOR_DAY * NB_CHANNELS)

    def forward(self, x):
        # x = N * L * NB_CHANNELS
        x = torch.transpose(x, 1, 2)
        x = self.cnn(x)
        y = self.sigmoid(x)
        B, C_out, L_out = y.shape
        # y = B * C_out * L_out
        yy = y.transpose(1, 2)
        y = self.mlp(yy.view(B * L_out, C_out))
        return y.view(-1, NB_RECORD_FOR_DAY * NB_CHANNELS)


def test(mod):
    mod.train(False)
    totloss, nb_batch = 0.0, 0
    for inputs, goldy in testloader:
        haty = mod(inputs)
        goldy = goldy.view(-1, NB_RECORD_FOR_DAY * NB_CHANNELS)
        loss = crit(haty, goldy)
        totloss += loss.item()
        nb_batch += 1
    totloss /= float(nb_batch)
    mod.train(True)
    return totloss


def train(mod):
    optim = torch.optim.Adam(mod.parameters(), lr=lr)
    train_loss_data, test_loss_data = [], []
    for epoch in range(nb_epoch):
        totloss, nb_batch = 0.0, 0
        testloss = test(mod)
        for inputs, goldy in trainloader:
            optim.zero_grad()
            # goldy = 1 * batch_size * NB_CHANNELS
            goldy = goldy.view(-1, NB_RECORD_FOR_DAY * NB_CHANNELS)
            haty = mod(inputs)
            loss = crit(haty, goldy)
            totloss += loss.item()
            nb_batch += 1
            loss.backward()
            optim.step()
        totloss /= float(nb_batch)
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
torch.save(mod.state_dict(), "./data/model.pt")

mod.load_state_dict(torch.load("data/model.pt"))
input, goldy = next(iter(trainloader))
haty = (
    np.transpose(mod(input).view(-1, NB_CHANNELS).detach().numpy())
    * df_max.values[:, np.newaxis]
)
goldy = (
    np.transpose(goldy.view(-1, NB_CHANNELS).detach().numpy())
    * df_max.values[:, np.newaxis]
)
nb_days_show = 15
abscisse = np.linspace(0, nb_days_show, nb_days_show * NB_RECORD_FOR_DAY)
for key in keywords:
    indice = df_max.index.get_loc(key)
    plt.plot(
        abscisse,
        haty[indice, -nb_days_show * NB_RECORD_FOR_DAY :],
        label="prédiction",
        color="red",
    )
    plt.plot(
        abscisse,
        goldy[indice, -nb_days_show * NB_RECORD_FOR_DAY :],
        label="vrai valeur",
        color="green",
    )
    plt.savefig(f"./plot/prev_{key}.png", dpi=1000)
    plt.legend(loc="upper left")
    plt.clf()
