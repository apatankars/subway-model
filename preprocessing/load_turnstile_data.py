import pandas as pd

# Read the CSV
year = 2024
df = pd.read_csv(f'MTA_Subway_Hourly_Ridership_{year}.csv')

# Safely convert columns
df["station_complex_id"] = pd.to_numeric(df["station_complex_id"], errors="coerce")
df["transfers"] = pd.to_numeric(df["transfers"], errors="coerce")
df["ridership"] = pd.to_numeric(df["ridership"], errors="coerce")

# Drop rows where critical conversions failed
df = df.dropna(subset=[
    "station_complex_id", "latitude", "longitude", "transfers", "ridership"
])

# Convert to correct types
df["station_complex_id"] = df["station_complex_id"].astype(int)
df["latitude"] = df["latitude"].astype(float)
df["longitude"] = df["longitude"].astype(float)
df["transfers"] = df["transfers"].astype(int)
df["ridership"] = df["ridership"].astype(int)

# Convert timestamp
df["transit_timestamp"] = pd.to_datetime(
    df["transit_timestamp"],
    format="%m/%d/%Y %I:%M:%S %p",
    errors="coerce"
)

# Convert to string (if not already)
df["borough"] = df["borough"].astype(str)
df["station_complex"] = df["station_complex"].astype(str)

# Save to CSV without quoting "station_complex" or any other string fields
# df.to_parquet(f'{year}_turnstile_data.parquet')

# df = pd.read_parquet(f'data/turnstile_data/{year}_turnstile_data.parquet')

# df_grouped = df.groupby(['station_complex_id', 'transit_timestamp']).agg({
#     'ridership': 'sum',
#     'transfers': 'sum',
#     'latitude': 'mean',
#     'longitude': 'mean'
# }).reset_index()

# Build full index
station_ids = df['station_complex_id'].unique()
full_times = pd.date_range(df['transit_timestamp'].min(),
                           df['transit_timestamp'].max(),
                           freq='h')

full_index = pd.MultiIndex.from_product(
    [station_ids, full_times],
    names=['station_complex_id', 'transit_timestamp']
)

# Reindex
df_filled = df.set_index(['station_complex_id', 'transit_timestamp']).reindex(full_index).reset_index()

# Fill missing values
df_filled['ridership'] = df_filled['ridership'].fillna(0)
df_filled['transfers'] = df_filled['transfers'].fillna(0)
df_filled[['latitude', 'longitude']] = df_filled.groupby('station_complex_id')[['latitude', 'longitude']].ffill().bfill()

# print(df_filled.loc[
#     (df_filled['station_complex_id'] == 611) & (df_filled['transit_timestamp'].dt.hour == 1) & (df_filled['transit_timestamp'].dt.month == 1),
#     ['transit_timestamp', 'ridership', 'transfers']
# ])
df_filled.to_parquet(f'data/turnstile_data/{year}_turnstile_data.parquet')

# Hourly