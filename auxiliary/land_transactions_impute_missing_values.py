import pandas as pd
import os
from pathlib import Path
import itertools


PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
from auxiliary.static_data import *
from auxiliary.auxiliary_funcs import *

train_folder = PROJECT_ROOT / 'train'
land_transactions_path = train_folder / 'land_transactions.csv'
land_transactions_nearby_sectors_path = train_folder / 'land_transactions_nearby_sectors.csv'

df_land_transactions = pd.read_csv(land_transactions_path)
df_land_transactions_nearby_sectors = pd.read_csv(land_transactions_nearby_sectors_path)

df_land_transactions["yr"] = df_land_transactions["month"].apply(
    func_str_to_yr
)
df_land_transactions["m"] = df_land_transactions["month"].apply(
    func_str_to_month
)

df_land_transactions_nearby_sectors["yr"] = df_land_transactions_nearby_sectors["month"].apply(
    func_str_to_yr
)
df_land_transactions_nearby_sectors["m"] = df_land_transactions_nearby_sectors["month"].apply(
    func_str_to_month
)


# Create all expected combinations (2024 only has months 1-7)
expected_combinations = set()
for yr in range(2019, 2025):
    months = range(1, 8) if yr == 2024 else range(1, 13)
    expected_combinations.update(
        itertools.product(months, [yr], [f"sector {i}" for i in range(1, 97)])
    )

current_combinations_land_transactions = set(
    zip(
        df_land_transactions["m"],
        df_land_transactions["yr"],
        df_land_transactions["sector"],
    )
)

current_combinations_land_transactions_nearby_sectors = set(
    zip(
        df_land_transactions_nearby_sectors["m"],
        df_land_transactions_nearby_sectors["yr"],
        df_land_transactions_nearby_sectors["sector"],
    )
)
missing_combinations_land_transactions = expected_combinations - current_combinations_land_transactions

missing_combinations_land_transactions_nearby_sectors = expected_combinations - current_combinations_land_transactions_nearby_sectors

zero_columns_land_transactions = [ "num_land_transactions", "construction_area", 
"planned_building_area", "transaction_amount" ]


zero_columns_land_transactions_nearby_sectors = [ "num_land_transactions_nearby_sectors", 
"construction_area_nearby_sectors", 
"planned_building_area_nearby_sectors", "transaction_amount_nearby_sectors" ]

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
df_missing_land_transactions = pd.DataFrame(
        [
            {
                **{col: 0 for col in zero_columns_land_transactions},
                "m": m,
                "yr": yr,
                "sector": sector,
                "month": f"{yr}_{month_names[m-1]}",
            }
            for m, yr, sector in missing_combinations_land_transactions
        ]
)
df_missing_land_transactions_nearby_sectors = pd.DataFrame(
        [
            {
                **{col: 0 for col in zero_columns_land_transactions_nearby_sectors},
                "m": m,
                "yr": yr,
                "sector": sector,
                "month": f"{yr}_{month_names[m-1]}",
            }
            for m, yr, sector in missing_combinations_land_transactions_nearby_sectors
        ]
    )

df_land_transactions = pd.concat(
    [df_land_transactions, df_missing_land_transactions], ignore_index=True
)
df_land_transactions_nearby_sectors = pd.concat(
    [df_land_transactions_nearby_sectors, df_missing_land_transactions_nearby_sectors], ignore_index=True
)
df_land_transactions["sector_num"] = (
    df_land_transactions["sector"].str.extract("(\d+)").astype(int)
)

df_land_transactions_nearby_sectors["sector_num"] = (
    df_land_transactions_nearby_sectors["sector"].str.extract("(\d+)").astype(int)
)
df_land_transactions = df_land_transactions.sort_values(["yr", "m", "sector_num"]).reset_index(drop=True)
df_land_transactions_nearby_sectors = df_land_transactions_nearby_sectors.sort_values(["yr", "m", "sector_num"]).reset_index(drop=True)

df_land_transactions.to_csv(train_folder / 'land_transactions_imputed.csv', index=False)
df_land_transactions_nearby_sectors.to_csv(train_folder / 'land_transactions_nearby_sectors_imputed.csv', index=False)
