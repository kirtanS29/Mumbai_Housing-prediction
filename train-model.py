import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import joblib

# =========================
# Load Dataset
# =========================
df = pd.read_csv("Mumbai.csv")

columns = [
    "Area", "Location", "No. of Bedrooms", "Resale",
    "Gymnasium", "SwimmingPool", "ClubHouse", "CarParking",
    "PowerBackup", "LiftAvailable", "VaastuCompliant", "Price", "24X7Security"
]
df = df[columns]

# =========================
# Outlier Removal
# =========================
def remove_outliers_iqr(data, column, factor=1.75):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

df = remove_outliers_iqr(df, "Area")
df = remove_outliers_iqr(df, "Price")

# =========================
# Location-based Average Price
# =========================
location_price_mean = (df["Price"] / df["Area"]).groupby(df["Location"]).mean().to_dict()
df["Location_AvgPrice"] = df["Location"].map(location_price_mean)

# =========================
# Create Amenity Score (sum of amenities)
# =========================
amenity_features = ["CarParking", "Gymnasium", "SwimmingPool", "ClubHouse",
                    "PowerBackup", "LiftAvailable", "VaastuCompliant", "24X7Security"]

df["Amenity_Count"] = df[amenity_features].sum(axis=1)

# =========================
# Features and Target
# =========================
X = df[["Area", "No. of Bedrooms", "Resale", "Location_AvgPrice", "Amenity_Count"]]
y = df["Price"]

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# Preprocessing
# =========================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["Area", "Location_AvgPrice", "Amenity_Count"]),
    ("ord", OrdinalEncoder(), ["No. of Bedrooms"]),
    ("bin", "passthrough", ["Resale"])
])

# =========================
# Models
# =========================
poly_reg = Pipeline([
    ("pre", preprocessor),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin", LinearRegression())
])

rf_reg = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42))
])

xgb_reg = Pipeline([
    ("pre", preprocessor),
    ("xgb", XGBRegressor(
        n_estimators=500,
        learning_rate=0.045,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

linear_reg = Pipeline([
    ("pre", preprocessor),
    ("lin", LinearRegression())
])

# =========================
# Voting Regressor
# =========================
voting_reg = VotingRegressor([
    ("poly", poly_reg),
    ("rf", rf_reg),
    ("xgb", xgb_reg),
    ("linear", linear_reg)
])

# =========================
# Train & Evaluate
# =========================
voting_reg.fit(X_train, y_train)
y_pred = voting_reg.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

# =========================
# Save Model
# =========================
joblib.dump(voting_reg, "model.joblib")
print("Model saved as 'model_main_amenity_score.joblib'")
# Save the column names used in training
joblib.dump(X.columns.tolist(), "model_columns.pkl")

