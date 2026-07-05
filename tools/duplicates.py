def get_duplicate_info(df):

    duplicate_rows = int(df.duplicated().sum())

    return {
        "duplicate_rows": duplicate_rows,
        "has_duplicates": duplicate_rows > 0
    }