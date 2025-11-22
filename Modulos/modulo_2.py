import pandas as pd

def preprocesar_datos(df):
    df = df.dropna(subset=['Subscription Status', 'Frequency of Purchases'])
    df["Subscription Status"] = df["Subscription Status"].map({"No": 0, "Yes": 1})
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    freq_dummies = pd.get_dummies(df["Frequency of Purchases"], prefix="Freq", drop_first=True)
    df = pd.concat([df, freq_dummies], axis=1)
    feature_cols = [
        'Age', 'Gender', 'Purchase Amount (USD)',
        'Review Rating', 'Previous Purchases'
    ] + list(freq_dummies.columns)
    X = df[feature_cols]
    y = df["Subscription Status"]
    return X, y, feature_cols
