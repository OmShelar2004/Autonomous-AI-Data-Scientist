import pandas as pd

def read_csv_tool(file_path:str):

    df = pd.read_csv(file_path)

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_summary": df.describe().to_dict(),
        "categorical_summary": df.describe(include="object").to_dict(),
        "unique_values": df.nunique().to_dict(),
    }


    return summary