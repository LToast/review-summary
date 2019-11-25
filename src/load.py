import pandas as pd
import json
from pathlib import Path

DATASET_PATH = Path("/home/lineal/Bureau/WIA/data/Electronics_5.json")
LIMIT = 100000


def load(path, limit):
    with open(path, "r") as e:
        return pd.DataFrame.from_records(json.loads(e.readline()) 
                                         for _ in range(limit))


def load_default(limit=LIMIT):
    return load(DATASET_PATH, limit)


if __name__ == "__main__":
    df = load_default()
