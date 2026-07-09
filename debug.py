import pandas as pd
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("csbender/xarm_pickandplace")
print("Total samples:", len(ds))

num_batches = len(ds) // 32
print("num_batches:", num_batches)

file_path = "/home/admin/.cache/huggingface/lerobot/csbender/xarm_pickandplace/data/chunk-000/episode_000001.parquet"
df = pd.read_parquet(file_path)
print(df.columns)
# Look at the 'episode_index' column
#print(df['episode_index'].unique())  # Shows all unique values in this column
print(df["task_index"].unique()[:20])
print("max task_index:", df["task_index"].max())