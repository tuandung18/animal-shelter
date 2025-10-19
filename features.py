from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from config import CFG


def parse_age_to_days(s: Optional[str]) -> float:
    if not isinstance(s, str) or not s.strip():
        return np.nan
    s = s.strip().lower()
    parts = s.split()
    if len(parts) < 2:
        return np.nan
    try:
        val = float(parts[0])
    except Exception:
        return np.nan
    unit = parts[1]
    if "year" in unit:
        return val * 365.25
    if "month" in unit:
        return val * 30.4375
    if "week" in unit:
        return val * 7.0
    if "day" in unit:
        return val
    return np.nan


def split_breed(breed: Optional[str]) -> Tuple[str, str, int]:
    if not isinstance(breed, str) or not breed:
        return "", "", 0
    b = breed.strip()
    is_mix = 1 if ("mix" in b.lower() or "/" in b) else 0
    parts = [p.strip() for p in b.split("/") if p.strip()]
    primary = parts[0] if len(parts) >= 1 else b
    secondary = parts[1] if len(parts) >= 2 else ""
    return primary, secondary, is_mix


def split_color(color: Optional[str]) -> Tuple[str, str, int, int, int, int]:
    if not isinstance(color, str) or not color:
        return "", "", 0, 0, 0, 0
    c = color.strip()
    parts = [p.strip() for p in c.split("/") if p.strip()]
    primary = parts[0] if len(parts) >= 1 else c
    secondary = parts[1] if len(parts) >= 2 else ""
    color_count = len(parts)
    is_tabby = 1 if "tabby" in c.lower() else 0
    is_brindle = 1 if "brindle" in c.lower() else 0
    is_tricolor = 1 if ("tricolor" in c.lower() or "tricolour" in c.lower()) else 0
    return primary, secondary, color_count, is_tabby, is_brindle, is_tricolor


def encode_sex(s: Optional[str]) -> Tuple[str, float, float, int]:
    if not isinstance(s, str) or not s.strip():
        return "Unknown", np.nan, np.nan, 1
    sx = s.strip()
    low = sx.lower()
    sex_unknown = 0
    is_intact = np.nan
    is_male = np.nan
    if "unknown" in low:
        sex_unknown = 1
    else:
        if "intact" in low:
            is_intact = 1.0
        elif "neutered" in low or "spayed" in low:
            is_intact = 0.0
        if "male" in low:
            is_male = 1.0
        elif "female" in low:
            is_male = 0.0
    return sx, is_intact, is_male, sex_unknown


def add_time_features(df: pd.DataFrame, col: str = "DateTime") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    out["Year"] = out[col].dt.year
    out["Month"] = out[col].dt.month
    out["Day"] = out[col].dt.day
    out["Hour"] = out[col].dt.hour
    out["DayOfWeek"] = out[col].dt.dayofweek
    try:
        out["WeekOfYear"] = out[col].dt.isocalendar().week.astype("int32")
    except Exception:
        out["WeekOfYear"] = out[col].dt.weekofyear
    out["IsWeekend"] = (out["DayOfWeek"] >= 5).astype(int)
    # Cyclic encodings
    # Month (1-12) -> angle
    out["MonthSin"] = np.sin(2 * np.pi * (out["Month"].fillna(0) % 12) / 12)
    out["MonthCos"] = np.cos(2 * np.pi * (out["Month"].fillna(0) % 12) / 12)
    # DayOfWeek (0-6)
    out["DOWSin"] = np.sin(2 * np.pi * (out["DayOfWeek"].fillna(0) % 7) / 7)
    out["DOWCos"] = np.cos(2 * np.pi * (out["DayOfWeek"].fillna(0) % 7) / 7)
    # Hour (0-23)
    out["HourSin"] = np.sin(2 * np.pi * (out["Hour"].fillna(0) % 24) / 24)
    out["HourCos"] = np.cos(2 * np.pi * (out["Hour"].fillna(0) % 24) / 24)
    # Season (basic Northern Hemisphere)
    # DJF=Winter, MAM=Spring, JJA=Summer, SON=Autumn
    month = out["Month"].fillna(0).astype(int)
    season = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        [0, 1, 2, 3],
        default=-1,
    )
    out["Season"] = season
    return out


def feature_engineer(df: pd.DataFrame, is_train: bool) -> Tuple[pd.DataFrame, List[str], List[str], str]:
    df = df.copy()
    id_col = "ID" if "ID" in df.columns else ("AnimalID" if "AnimalID" in df.columns else None)
    if id_col is None:
        df["ID"] = np.arange(len(df))
        id_col = "ID"
    df["HasName"] = df["Name"].notnull() & (df["Name"].astype(str).str.strip() != "")
    df["Name"] = df["Name"].fillna("")
    df["NameLength"] = df["Name"].astype(str).str.len()
    name_counts = df["Name"].value_counts(dropna=False)
    df["NameFreq"] = df["Name"].map(name_counts).astype(int)
    df["AgeDays"] = df["AgeuponOutcome"].apply(parse_age_to_days)
    bins = [-np.inf, 120, 365, 7*365, np.inf]
    labels = ["lt4mo", "lt1y", "lt7y", "ge7y"]
    df["AgeGroup"] = pd.cut(df["AgeDays"], bins=bins, labels=labels)
    sex_feats = df["SexuponOutcome"].apply(encode_sex)
    df["SexClean"] = [t[0] for t in sex_feats]
    df["IsIntact"] = [t[1] for t in sex_feats]
    df["IsMale"] = [t[2] for t in sex_feats]
    df["SexUnknown"] = [t[3] for t in sex_feats]
    breeds = df["Breed"].apply(split_breed)
    df["PrimaryBreed"] = [t[0] for t in breeds]
    df["SecondaryBreed"] = [t[1] for t in breeds]
    df["IsMix"] = [t[2] for t in breeds]
    colors = df["Color"].apply(split_color)
    df["PrimaryColor"] = [t[0] for t in colors]
    df["SecondaryColor"] = [t[1] for t in colors]
    df["ColorCount"] = [t[2] for t in colors]
    df["IsTabby"] = [t[3] for t in colors]
    df["IsBrindle"] = [t[4] for t in colors]
    df["IsTricolor"] = [t[5] for t in colors]
    df = add_time_features(df, col="DateTime")
    if "OutcomeSubtype" in df.columns:
        df.drop(columns=["OutcomeSubtype"], inplace=True)
    expected_cols = [
        "AnimalType","SexuponOutcome","AgeuponOutcome","Breed","Color","DateTime"
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""
    cat_cols = [
        "AnimalType", "SexClean", "PrimaryBreed", "SecondaryBreed",
        "PrimaryColor", "SecondaryColor", "AgeGroup"
    ]
    if CFG.allow_name_as_categorical:
        cat_cols.append("Name")
    num_cols = [
        "HasName","NameLength","NameFreq",
        "AgeDays","IsIntact","IsMale","SexUnknown",
        "IsMix","ColorCount","IsTabby","IsBrindle","IsTricolor",
        "Year","Month","Day","Hour","DayOfWeek","WeekOfYear","IsWeekend",
        "MonthSin","MonthCos","DOWSin","DOWCos","HourSin","HourCos","Season"
    ]
    keep_cols = (cat_cols + num_cols + [id_col])
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols], cat_cols, num_cols, id_col
