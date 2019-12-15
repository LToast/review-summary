import pandas as pd
import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "../data/Electronics_5.json"
LIMIT = 1000
REVIEW_LIM = 100
SUM_LIM = 10

def load_max_seq(path, limit, max_review_size, max_summary_size):
    with open(path, "r") as e:
        js = []
        while len(js) < limit:
            read = json.loads(e.readline())
            if (len(read.get("reviewText").split()) < max_review_size) and (len(read.get("summary").split()) < max_summary_size):
                js.append(read)
        return pd.DataFrame.from_records(js)


def load(path, limit):
    with open(path, "r") as e:
        return pd.DataFrame.from_records(json.loads(e.readline()) for _ in range(limit))


def load_default(limit=LIMIT, max_review=REVIEW_LIM, max_summmary=SUM_LIM):
    return load_max_seq(DATASET_PATH, limit, max_review, max_summmary)


if __name__ == "__main__":
    print(DATASET_PATH)
    df = load_default(100)
