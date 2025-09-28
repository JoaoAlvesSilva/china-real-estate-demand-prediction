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

# Create plots folder if it doesn't exist
plots_dir = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Loop over all 96 sectors
for sector in range(1, 97):
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

    # Save plot to plots folder
    plot_filename = f"sector_{sector}_new_house_transactions.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    print(f"Saved plot for {sector_to_plot} to {plot_filename}")

print(f"All 96 plots have been saved to {plots_dir}")
