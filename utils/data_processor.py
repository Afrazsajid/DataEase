import pandas as pd
import numpy as np
from typing import Union, Optional

class DataProcessor:
    @staticmethod
    def load_data(file) -> pd.DataFrame:
        """Load data from uploaded file."""
        try:
            if file.name.endswith('.csv'):
                return pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                return pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")

    @staticmethod
    def get_file_info(file) -> dict:
        """Get file metadata."""
        return {
            "filename": file.name,
            "size": f"{file.size / 1024:.2f} KB",
            "format": file.name.split('.')[-1].upper()
        }

    @staticmethod
    def clean_data(df: pd.DataFrame,
                   remove_duplicates: bool = False,
                   fill_missing: Optional[str] = None,
                   fill_value: Union[str, float, None] = None) -> pd.DataFrame:
        """Clean the dataset based on specified options."""
        if remove_duplicates:
            df = df.drop_duplicates()

        if fill_missing:
            for column in df.columns:
                if df[column].isnull().any():
                    if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                        df[column] = df[column].fillna(df[column].mean())
                    elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                        df[column] = df[column].fillna(df[column].median())
                    elif fill_missing == 'mode':
                        df[column] = df[column].fillna(df[column].mode()[0])
                    elif fill_missing == 'custom' and fill_value is not None:
                        df[column] = df[column].fillna(fill_value)

        return df

    @staticmethod
    def export_data(df: pd.DataFrame, format: str) -> Union[str, bytes]:
        """Export data in specified format."""
        if format == 'csv':
            return df.to_csv(index=False)
        elif format == 'excel':
            return df.to_excel(index=False, engine='xlsxwriter')
        else:
            raise ValueError("Unsupported export format")
