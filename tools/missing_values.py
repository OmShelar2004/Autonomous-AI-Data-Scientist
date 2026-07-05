

def get_missing_values(df):

    return {
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (
            (df.isnull().sum() / len(df)) * 100
        ).round(2).to_dict()
    }