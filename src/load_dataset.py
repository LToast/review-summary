import pandas as pd
import json
from pathlib import Path

DATASET_PATH = Path("/home/lineal/Bureau/WIA/data/Electronics_5.json")
LIMIT = 1000
SEQ_LIM = 500


def load_max_seq(path, limit, max_review_size):
    with open(path, "r") as e:
        js = []
        while len(js) < limit:
            read = json.loads(e.readline())
            if len(read.get("reviewText")) < max_review_size:
                js.append(read)
        return pd.DataFrame.from_records(js)


def load(path, limit):
    with open(path, "r") as e:
        return pd.DataFrame.from_records(json.loads(e.readline()) for _ in range(limit))


def load_default(limit=LIMIT, max_review=SEQ_LIM):
    return load_max_seq(DATASET_PATH, limit, max_review)


if __name__ == "__main__":
    df = load_default()
