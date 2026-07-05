def get_statistics(df):

    return {
        "statistics": df.describe().to_dict()
    }