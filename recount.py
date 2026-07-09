import json
from pathlib import Path

# ==== CONFIGURATION ====
EPISODES_JSONL_PATH = Path("/home/admin/.cache/huggingface/lerobot/csbender/xarm_pickandplace/meta/episodes.jsonl")

# ==== CALCULATE TOTALS ====
total_frames = 0
total_episodes = 0

with EPISODES_JSONL_PATH.open("r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        episode = json.loads(line)
        total_frames += episode.get("length", 0)
        total_episodes += 1

print(f"Total episodes: {total_episodes}, Total frames: {total_frames}")
