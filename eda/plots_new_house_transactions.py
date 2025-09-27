import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
from auxiliary.static_data import *
from auxiliary.auxiliary_funcs import *
import ast


df_new_house_transactions_per_sector = pd.read_csv(
    os.path.join(PROJECT_ROOT, "train", "new_house_transactions_per_sector.csv")
)
df_new_house_transactions_per_sector["yr_m"] = df_new_house_transactions_per_sector[
    "yr_m"
].apply(ast.literal_eval)

list_yr_m = list(df_new_house_transactions_per_sector["yr_m"])
time_difference_to_2019 = []
for yr_m in list_yr_m:
    yr = yr_m[0]
    m = yr_m[1]
    time_difference_to_2019.append(yr - 2019 + m / 12)

# WRITE SECTOR TO PLOT HERE
sector = 61

plt.figure(figsize=(10, 6))
sector_to_plot = "sector " + str(sector)
plt.plot(
    time_difference_to_2019,
    df_new_house_transactions_per_sector[
        sector_to_plot + "_new_house_transaction_amount"
    ],
    marker="o",
)
plt.xlabel("Years since 2019 (fractional)")
plt.ylabel(sector_to_plot + "_new_house_transaction_amount ")
plt.title("New House Transaction Amount Over Time for " + sector_to_plot)
plt.grid(True)
plt.tight_layout()
plt.show()
del (sector, sector_to_plot)
