import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]

def preprocess_data(df):
    # Feature engineering
    df["brand"] = df["Car_Name"].str.split(" ").str[0]
    df["Age"] = 2025 - df["Year"]

    # Outlier removal
    df = remove_outliers_iqr(df, "Present_Price")
    df = remove_outliers_iqr(df, "Driven_kms")

    # Target encoding for brand
    target_encoding = df.groupby('brand')["Selling_Price"].mean().to_dict()
    df["brand_encoded"] = df["brand"].map(target_encoding)

    # One-hot encoding
    df_encoded = pd.get_dummies(
        df,
        columns=["Fuel_Type", "Selling_type", "Owner", "Transmission"],
        drop_first=True
    )

    # Drop unnecessary columns
    df_encoded.drop(columns=["Car_Name", "Year", "brand"], inplace=True)

    return df_encoded

def scale_split(df_encoded, target="Selling_Price"):
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns, X_train, X_test
