import os
import pandas as pd
from data import read_input_files
from model import train_and_predict


def main():
    train, test, sample = read_input_files()
    sub = train_and_predict(train, test, sample)
    if os.path.exists("sample_submission.csv"):
        ss = pd.read_csv("sample_submission.csv")
        expected_cols = ss.columns.tolist()
        cols = [c for c in expected_cols if c in sub.columns] + [c for c in sub.columns if c not in expected_cols]
        sub = sub[cols]
    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | shape={sub.shape}")


if __name__ == "__main__":
    main()
