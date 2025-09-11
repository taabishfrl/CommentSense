import pandas as pd

def load_csv(file_path):
    """Load a CSV file into a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def load_excel(file_path):
    """Load an Excel file into a DataFrame."""
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None