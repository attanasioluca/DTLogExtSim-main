import pandas as pd

FILE_PATH = "data/BIMP/sim_data_waiting_times.csv"

df= pd.read_csv(FILE_PATH)

# Remove duplicate scenario_ids
df = df.drop_duplicates(subset="scenario_id", keep="first")

# Sort
df = df.sort_values(by="scenario_id").reset_index(drop=True)

df.to_csv(FILE_PATH, index=False)

print(len(df))