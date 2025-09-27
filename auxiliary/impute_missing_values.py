import pandas as pd
import os
import itertools
from pathlib import Path

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
from auxiliary.static_data import *
from auxiliary.auxiliary_funcs import *

df_new_house_transactions = pd.read_csv(
    os.path.join(PROJECT_ROOT, "train", "new_house_transactions.csv")
)

df_new_house_transactions["yr"] = df_new_house_transactions["month"].apply(
    func_str_to_yr
)
df_new_house_transactions["m"] = df_new_house_transactions["month"].apply(
    func_str_to_month
)

# Create all expected combinations (2024 only has months 1-7)
expected_combinations = set()
for yr in range(2019, 2025):
    months = range(1, 8) if yr == 2024 else range(1, 13)
    expected_combinations.update(
        itertools.product(months, [yr], [f"sector {i}" for i in range(1, 97)])
    )

current_combinations = set(
    zip(
        df_new_house_transactions["m"],
        df_new_house_transactions["yr"],
        df_new_house_transactions["sector"],
    )
)
missing_combinations = expected_combinations - current_combinations

# Create missing rows with zeros for specified columns
zero_columns = [
    "num_new_house_transactions",
    "area_new_house_transactions",
    "price_new_house_transactions",
    "amount_new_house_transactions",
    "area_per_unit_new_house_transactions",
    "total_price_per_unit_new_house_transactions",
    "num_new_house_available_for_sale",
    "area_new_house_available_for_sale",
    "period_new_house_sell_through",
]

month_names = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
df_missing = pd.DataFrame(
    [
        {
            **{col: 0 for col in zero_columns},
            "m": m,
            "yr": yr,
            "sector": sector,
            "month": f"{yr}_{month_names[m-1]}",
        }
        for m, yr, sector in missing_combinations
    ]
)

# Merge with original dataframe and sort by yr, m, sector
df_new_house_transactions = pd.concat(
    [df_new_house_transactions, df_missing], ignore_index=True
)
df_new_house_transactions["sector_num"] = (
    df_new_house_transactions["sector"].str.extract("(\d+)").astype(int)
)
df_new_house_transactions = (
    df_new_house_transactions.sort_values(["yr", "m", "sector_num"])
    .drop("sector_num", axis=1)
    .reset_index(drop=True)
)
df_new_house_transactions.rename(
    columns={"amount_new_house_transactions": "new_house_transaction_amount"},
    inplace=True,
)
print(
    f"Added {len(df_missing)} missing rows. Final shape: {df_new_house_transactions.shape}"
)
df_new_house_transactions.to_csv(
    os.path.join(PROJECT_ROOT, "train", "new_house_transactions_imputed.csv"),
    index=False,
)
