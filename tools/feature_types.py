def get_feature_types(df):

    return {
        "numerical_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist()
    }