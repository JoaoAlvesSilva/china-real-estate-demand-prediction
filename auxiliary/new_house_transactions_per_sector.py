import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
from auxiliary.static_data import *
from auxiliary.auxiliary_funcs import *

df_new_house_transactions = pd.read_csv(
    os.path.join(PROJECT_ROOT, "train", "new_house_transactions_imputed.csv")
)

list_yr_m = list(df_new_house_transactions[["yr", "m"]].drop_duplicates().to_numpy())
dict_yr_m_to_new_house_transaction_amount = {
    (yr, m): df_new_house_transactions.loc[
        (df_new_house_transactions["yr"] == yr) & (df_new_house_transactions["m"] == m)
    ]["new_house_transaction_amount"].sum()
    for yr, m in list_yr_m
}

df_new_house_transactions[
    "total_new_house_transaction_amount"
] = df_new_house_transactions.apply(
    lambda row: dict_yr_m_to_new_house_transaction_amount.get((row["yr"], row["m"]), 0),
    axis=1,
)

dict_sect_to_new_house_transaction_amount = {}
for sector in df_new_house_transactions["sector"].unique():
    dict_sect_to_new_house_transaction_amount[sector] = df_new_house_transactions[
        df_new_house_transactions["sector"] == sector
    ]["new_house_transaction_amount"].to_numpy()

dict_sect_to_new_house_transaction_amount["total"] = np.array(
    list(dict_yr_m_to_new_house_transaction_amount.values())
)

df_new_house_transactions_per_sector = pd.DataFrame(
    {
        sector
        + "_new_house_transaction_amount": dict_sect_to_new_house_transaction_amount[
            sector
        ]
        for sector in dict_sect_to_new_house_transaction_amount.keys()
    }
)
df_new_house_transactions_per_sector["yr_m"] = [
    (yr_m[0], yr_m[1]) for yr_m in list_yr_m
]
df_new_house_transactions_per_sector.to_csv(
    os.path.join(PROJECT_ROOT, "train", "new_house_transactions_per_sector.csv"),
    index=False,
)
