import os
import pandas as pd
from typing import Optional, Tuple


def read_input_files() -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    train_path = "train.csv"
    test_path = "test.csv"
    sample_path = "sample_submission.csv"
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("train.csv or test.csv not found in the working directory.")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path) if os.path.exists(sample_path) else None
    return train, test, sample
