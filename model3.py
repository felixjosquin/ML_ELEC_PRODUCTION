import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

###### HYPERPARAMETER ######
nb_epoch = 500
lr = 0.00005
nb_days_by_batch = 50
nb_days_look_before = 4

hidden_layer_cnn = 96
hidden_layer_cnn_no_periodic = 96

hidden_layer_rnn = 48
n_record_rnn = 3  # 1 2 3 4 6 8 12 16 24 48
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
    "Bioénergies",
    "Eolien",
    "Solaire",
    "Hydraulique",
    "Nucléaire",
]
df_max = df[keywords].max()
df[keywords] /= df_max[keywords]

product_type_periodic = [
    "Solaire",
    "Hydraulique",
    "Nucléaire",
]
indice_periodic = [df_max.index.get_loc(i) for i in product_type_periodic]
indice_no_periodic = [i for i in range(NB_CHANNELS) if i not in indice_periodic]

traindata = np.array(
    [
        df.loc[df["End_time"] <= "2021-05-30 00:00:00"]
        .drop(
            columns=[
                "Start_time",
                "End_time",
            ]
        )
        .values
    ]
)
# shape 1 * (880j * 48 record_for_a_day) * 8 channels
traindata = (
    traindata[:, : -(traindata.shape[1] % (nb_days_by_batch * NB_RECORD_FOR_DAY)), :]
    if traindata.shape[1] % (nb_days_by_batch * NB_RECORD_FOR_DAY) != 0
    else traindata
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
    traindata[:, nb_days_look_before * NB_RECORD_FOR_DAY :, :], dtype=torch.float32
)
trainds = torch.utils.data.TensorDataset(trainx, trainy)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)

testdata = np.array(
    [
        (
            df.loc[df["Start_time"] >= "2021-05-31 00:00:00"]
            .drop(
                columns=[
                    "Start_time",
                    "End_time",
                ]
            )
            .values
        )
    ]
)
# shape 1 * (366j * 48 record_for_a_day) * 8 channels
testdata = (
    testdata[:, : -(testdata.shape[1] % (nb_days_by_batch * NB_RECORD_FOR_DAY)), :]
    if testdata.shape[1] % (nb_days_by_batch * NB_RECORD_FOR_DAY) != 0
    else testdata
)
testdata = np.reshape(
    testdata,
    (
        testdata.shape[1] // (nb_days_by_batch * NB_RECORD_FOR_DAY),
        nb_days_by_batch * NB_RECORD_FOR_DAY,
        NB_CHANNELS,
    ),
)
testx = torch.tensor(testdata[:, :-NB_RECORD_FOR_DAY, :], dtype=torch.float32)
testy = torch.tensor(
    testdata[:, nb_days_look_before * NB_RECORD_FOR_DAY :, :], dtype=torch.float32
)
testds = torch.utils.data.TensorDataset(testx, testy)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)


class Mod(nn.Module):
    def __init__(self):
        super().__init__()
        self.kern = nb_days_look_before * NB_RECORD_FOR_DAY
        self.cnn_periodic = nn.Conv1d(
            in_channels=NB_CHANNELS,
            out_channels=hidden_layer_cnn,
            kernel_size=self.kern,
            stride=NB_RECORD_FOR_DAY,
        )
        self.mlp_cnn_periodic = nn.Linear(
            hidden_layer_cnn, NB_RECORD_FOR_DAY * len(indice_periodic)
        )

        self.rnn = torch.nn.RNN(
            input_size=n_record_rnn * NB_CHANNELS,
            hidden_size=hidden_layer_rnn,
        )
        self.mlp_rnn = nn.Linear(
            hidden_layer_cnn_no_periodic,
            NB_RECORD_FOR_DAY * len(indice_no_periodic),
        )
        self.cnn_no_periodic = nn.Conv1d(
            in_channels=hidden_layer_rnn,
            out_channels=hidden_layer_cnn_no_periodic,
            kernel_size=self.kern // n_record_rnn,
            stride=NB_RECORD_FOR_DAY // n_record_rnn,
        )

        self.tanH = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Periodic energy type
        x_periodic = torch.transpose(x, 1, 2)
        # x = 1  * 3 * L
        x_periodic = self.cnn_periodic(x_periodic)
        y_periodic = self.tanH(x_periodic)
        Batch_size, C_out, L_out = y_periodic.shape
        # y_periodic = Batch_size * C_out * L_out
        y_periodic = y_periodic.transpose(1, 2)
        y_periodic = self.mlp_cnn_periodic(y_periodic.view(Batch_size * L_out, C_out))
        out_periodic = self.sigmoid(
            y_periodic.view(-1, NB_RECORD_FOR_DAY * len(indice_periodic))
        )

        # Not periodic energy type
        x_no_periodic = x.view(-1, n_record_rnn * NB_CHANNELS)
        y_no_periodic, _ = self.rnn(x_no_periodic)
        y_no_periodic = y_no_periodic.transpose(0, 1)
        out_cnn = self.cnn_no_periodic(y_no_periodic)
        out_cnn = out_cnn.transpose(0, 1)
        out_no_periodic = self.sigmoid(self.mlp_rnn(out_cnn))

        out = torch.cat((out_no_periodic, out_periodic), dim=1)
        return out


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
    abs = list(range(nb_epoch - 10))
    plt.plot(abs, train_loss_data[20:])
    plt.plot(abs, test_loss_data[20:])
    plt.savefig("./plot/model3/loss.png", dpi=1000)
    plt.clf()
    print("fin")


mod = Mod()
print(
    "nparms",
    sum(p.numel() for p in mod.parameters() if p.requires_grad),
    file=sys.stderr,
)
# train(mod)
# torch.save(mod.state_dict(), "./data/model3.pt")

mod.load_state_dict(torch.load("data/model3.pt"))
input, goldy = next(iter(testloader))
haty = np.transpose(mod(input).view(-1, NB_CHANNELS).detach().numpy())
goldy = np.transpose(goldy.view(-1, NB_CHANNELS).detach().numpy())
# haty = haty * df_max.values[:, np.newaxis]
# goldy = goldy * df_max.values[:, np.newaxis]
nb_days_show = 30
abscisse = np.linspace(0, nb_days_show, nb_days_show * NB_RECORD_FOR_DAY)
for key in keywords:
    indice = df_max.index.get_loc(key)
    plt.plot(
        abscisse,
        haty[
            indice,
            : nb_days_show * NB_RECORD_FOR_DAY,
        ],
        label="prédiction",
        color="red",
    )
    plt.plot(
        abscisse,
        goldy[
            indice,
            : nb_days_show * NB_RECORD_FOR_DAY,
        ],
        label="vrai valeur",
        color="green",
    )
    plt.savefig(f"./plot/model3/prev_{key}.png", dpi=1000)
    plt.legend(loc="upper left")
    plt.clf()
