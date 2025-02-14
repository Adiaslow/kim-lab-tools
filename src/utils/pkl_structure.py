import pickle
import numpy as np
import pandas as pd
from typing import Any
from collections.abc import Sized

def describe_structure(obj: Any, level: int = 0, max_level: int = 10) -> None:
    indent = "  " * level
    obj_type = type(obj).__name__

    print(f"{indent}Type: {obj_type}")

    if level >= max_level:
        print(f"{indent}Max recursion depth reached")
        return

    if isinstance(obj, (dict, pd.DataFrame)):
        print(f"{indent}Shape/Length: {len(obj)}")
        if isinstance(obj, pd.DataFrame):
            print(f"{indent}Columns: {list(obj.columns)}")
            print(f"{indent}Data types:\n{obj.dtypes.to_string()}")
        else:
            # For dense dictionary keys, show a sample and summarize pattern
            keys = list(obj.keys())
            if len(keys) > 10 and all(isinstance(k, tuple) for k in keys):
                first_key = keys[0]
                last_key = keys[-1]
                print(f"{indent}Keys are tuples of form: {type(first_key[0]).__name__}")
                print(f"{indent}First key: {first_key}")
                print(f"{indent}Last key: {last_key}")
                print(f"{indent}Total keys: {len(keys)}")
            else:
                print(f"{indent}Keys: {keys}")

            if obj:
                first_key = next(iter(obj))
                print(f"{indent}Sample value structure (for key '{first_key}'):")
                describe_structure(obj[first_key], level + 1, max_level)

    elif isinstance(obj, (list, tuple, set, np.ndarray)):
        if isinstance(obj, np.ndarray):
            print(f"{indent}Shape: {obj.shape}")
            print(f"{indent}Data type: {obj.dtype}")
        else:
            print(f"{indent}Length: {len(obj)}")
            # For dense sequences, show pattern
            if len(obj) > 10 and all(isinstance(x, type(obj[0])) for x in obj):
                print(f"{indent}First element: {obj[0]}")
                print(f"{indent}Last element: {obj[-1]}")

        if len(obj) > 0:
            print(f"{indent}Sample element structure:")
            describe_structure(obj[0], level + 1, max_level)

    elif isinstance(obj, (str, int, float, bool)):
        print(f"{indent}Value: {obj}")

def analyze_pickle_file(file_path: str) -> None:
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Structure of pickle file: {file_path}\n")
        describe_structure(data)
    except Exception as e:
        print(f"Error loading pickle file: {str(e)}")

# Example usage
if __name__ == "__main__":
    file_path = "/Volumes/euiseokdataUCSC_1/Matt_Jacobs/Images_and_Data/H2B_quantification/p60/m776/M776_s030_RSPagl.pkl"  # Replace with your pickle file path
    analyze_pickle_file(file_path)
