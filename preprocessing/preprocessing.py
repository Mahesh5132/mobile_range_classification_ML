from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    X = df.drop("price_range", axis=1)
    y = df["price_range"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def split_data(X_scaled, y):
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
