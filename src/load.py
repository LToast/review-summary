import pandas as pd
import json
from pathlib import Path

DATASET_PATH = Path("/home/lineal/Bureau/WIA/data/Electronics_5.json")
LIMIT = 10000

def load(path, limit):
    with open(path, "r") as e:
        return pd.DataFrame.from_records(json.loads(e.readline()) for _ in range(limit))
    

def main():
    return load(DATASET_PATH, LIMIT)


if __name__ == "__main__":
    df = main()
    
