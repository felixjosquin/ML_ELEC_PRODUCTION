import pandas as pd
import matplotlib.pyplot as plt
import os


csv_file_paths = [
    "./data/CO2mix_RTE_2019.csv",
    "./data/CO2mix_RTE_2020.csv",
    "./data/CO2mix_RTE_2022.csv",
]
data_clean_path = "./data/data.csv"


for csv_file in csv_file_paths:
    df = pd.read_csv(csv_file, sep=";")
    df_data = df.loc[
        df.index % 2 == 0,
        [
            "Date",
            "Heures",
            "Fioul",
            "Charbon",
            "Gaz",
            "Nucléaire",
            "Eolien",
            "Solaire",
            "Hydraulique",
            "Bioénergies",
        ],
    ]
    df_data.rename(columns={"Heures": "Start_time"}, inplace=True)
    df_data["Start_time"] = pd.to_datetime(
        df_data["Date"] + " " + df_data["Start_time"], format="%d/%m/%Y %H:%M"
    )
    df_data["End_time"] = df_data["Start_time"] + pd.Timedelta(minutes=30)
    df_data = df_data.drop(columns=["Date"])

    df_data.to_csv(
        data_clean_path,
        mode="a",
        index=False,
        header=not os.path.exists(data_clean_path),
    )

    # df_data.loc[
    #     (df_data["Start_time"] > "2021-08-02") & (df_data["Start_time"] < "2021-08-10")
    # ].plot(kind="line", x="Start_time", y=["Consommation"])
    # plt.savefig("plot/myimage.png", dpi=1200)
