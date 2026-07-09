import pandas as pd
import json
from pathlib import Path
import shutil

# ==== CONFIGURATION ====
PARQUET_DIR = Path("/home/admin/.cache/huggingface/lerobot/csbender/xarm_pickandplace/data/chunk-000/")           # folder with episode_*.parquet
STATS_JSONL_PATH = Path("/home/admin/.cache/huggingface/lerobot/csbender/xarm_pickandplace/meta/episodes_stats.jsonl")  # path to episodes_stats.jsonl
EPISODES_JSONL_PATH = Path("/home/admin/.cache/huggingface/lerobot/csbender/xarm_pickandplace/meta/episodes.jsonl")
BACKUP_DIR = Path("/home/admin/.cache/huggingface/lerobot/csbender/xarm_pickandplace/backup/")
BACKUP_DIR.mkdir(exist_ok=True)


# ==== STEP 0: Backup files ====
print("🔹 Backing up files...")
for f in PARQUET_DIR.glob("episode_*.parquet"):
    shutil.copy(f, BACKUP_DIR / f.name)
shutil.copy(STATS_JSONL_PATH, BACKUP_DIR / STATS_JSONL_PATH.name)
shutil.copy(EPISODES_JSONL_PATH, BACKUP_DIR / EPISODES_JSONL_PATH.name)
print(f"✅ Backed up all files to {BACKUP_DIR}")

# ==== STEP 1: Renumber Parquet files ====
print("\n🔹 Renumbering parquet files...")
parquet_files = sorted(PARQUET_DIR.glob("episode_*.parquet"))
for new_idx, old_path in enumerate(parquet_files):
    df = pd.read_parquet(old_path)
    if "episode_index" in df.columns:
        df["episode_index"] = new_idx
        df.to_parquet(old_path, index=False)
    # Rename file to new sequential numbering
    new_name = f"episode_{new_idx:06d}.parquet"
    old_path.rename(PARQUET_DIR / new_name)
    print(f"  {old_path.name} → {new_name}")

# ==== STEP 2: Renumber episodes_stats.jsonl ====
print("\n🔹 Renumbering episodes_stats.jsonl...")
new_lines = []
with open(STATS_JSONL_PATH, "r") as f:
    for new_idx, line in enumerate(f):
        data = json.loads(line)
        data["episode_index"] = new_idx
        # Update stats.episode_index fields if present
        for k, v in data.get("stats", {}).items():
            epi_idx_stats = v.get("episode_index")
            if epi_idx_stats:
                # Replace min, max, mean with new index
                n = len(epi_idx_stats.get("min", []))
                epi_idx_stats["min"] = [new_idx] * n
                epi_idx_stats["max"] = [new_idx] * n
                epi_idx_stats["mean"] = [float(new_idx)] * n
        new_lines.append(json.dumps(data))
with open(STATS_JSONL_PATH, "w") as f:
    f.write("\n".join(new_lines) + "\n")
print("✅ Done renumbering episodes_stats.jsonl")

# ==== STEP 3: Renumber episodes.jsonl ====
print("\n🔹 Renumbering episodes.jsonl...")
new_lines = []
with open(EPISODES_JSONL_PATH, "r") as f:
    for new_idx, line in enumerate(f):
        data = json.loads(line)
        data["episode_index"] = new_idx
        new_lines.append(json.dumps(data))
with open(EPISODES_JSONL_PATH, "w") as f:
    f.write("\n".join(new_lines) + "\n")
print("✅ Done renumbering episodes.jsonl")

print("\n🎉 All episodes renumbered sequentially from 0.")
