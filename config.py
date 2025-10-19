from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Config:
    n_splits: int = 5
    random_state: int = 42
    iterations: int = 8000
    learning_rate: float = 0.02
    depth: int = 6
    l2_leaf_reg: float = 8.0
    early_stopping_rounds: int = 200
    use_temperature_scaling: bool = True
    temp_grid: Tuple[float, float, int] = (0.6, 2.0, 30)
    allow_name_as_categorical: bool = False
    verbose_eval: int = 200
    cv_strategy: str = "stratified"  # or "time"
    seeds: Tuple[int, ...] = (42, 1337, 2025)
    rare_threshold: int = 20
    auto_class_weights: Optional[str] = "Balanced"  # or None
    bagging_temperature: float = 0.5
    random_strength: float = 1.0
    one_hot_max_size: int = 16


CFG = Config()
