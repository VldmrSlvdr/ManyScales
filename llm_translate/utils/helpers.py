"""Utility helper functions."""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Set random seed to {seed}")


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_data(
    file_path: Path,
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load data from CSV or Excel file.
    
    Args:
        file_path: Path to data file
        required_columns: List of required column names
    
    Returns:
        DataFrame
    
    Raises:
        ValueError: If required columns are missing
    """
    logger.info(f"Loading data from: {file_path}")
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load based on extension
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif file_path.suffix == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Validate required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    return df


def save_results(
    data: Any,
    output_path: Path,
    file_format: Optional[str] = None
) -> None:
    """
    Save results to file.
    
    Args:
        data: Data to save (DataFrame, dict, array, etc.)
        output_path: Output file path
        file_format: File format override (inferred from path if None)
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    if file_format is None:
        file_format = output_path.suffix.lstrip(".")
    
    logger.info(f"Saving results to: {output_path}")
    
    if isinstance(data, pd.DataFrame):
        if file_format == "csv":
            data.to_csv(output_path, index=False)
        elif file_format in ["xlsx", "xls"]:
            data.to_excel(output_path, index=False)
        elif file_format == "json":
            data.to_json(output_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format for DataFrame: {file_format}")
    
    elif isinstance(data, dict):
        if file_format == "json":
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format for dict: {file_format}")
    
    elif isinstance(data, np.ndarray):
        if file_format in ["npy"]:
            np.save(output_path, data)
        elif file_format == "txt":
            np.savetxt(output_path, data)
        else:
            raise ValueError(f"Unsupported format for array: {file_format}")
    
    else:
        # Try generic text save
        with open(output_path, "w") as f:
            f.write(str(data))
    
    logger.info(f"Results saved successfully")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file = Path(log_file)
        ensure_dir(log_file.parent)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    logger.info(f"Logging setup complete (level: {level})")


def validate_item_dataframe(df: pd.DataFrame) -> None:
    """
    Validate item dataframe has required structure.
    
    Args:
        df: DataFrame to validate
    
    Raises:
        ValueError: If validation fails
    """
    required_columns = ["item_id", "lang", "trans_type", "text"]
    
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Validate languages
    valid_langs = df["lang"].unique()
    if "en" not in valid_langs:
        logger.warning("No English (en) items found in data")
    
    # Validate translation types
    valid_types = df["trans_type"].unique()
    expected_types = ["origin", "llm", "human"]
    
    logger.info(f"Found languages: {valid_langs}")
    logger.info(f"Found translation types: {valid_types}")
    
    # Convert item_id to string for consistent handling
    df["item_id"] = df["item_id"].astype(str)

