import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
import joblib

# Load the data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Create interaction term in both training and test sets
train_df['AvgViewingDuration_WatchlistSize'] = train_df['AverageViewingDuration'] * train_df['WatchlistSize']
test_df['AvgViewingDuration_WatchlistSize'] = test_df['AverageViewingDuration'] * test_df['WatchlistSize']
train_df['Total_viewing'] = train_df['AverageViewingDuration']*train_df['ContentDownloadsPerMonth']
test_df['Total_viewing'] = test_df['AverageViewingDuration']*test_df['ContentDownloadsPerMonth']

# Define categorical and numerical columns
categorical_cols = ['SubscriptionType', 'PaymentMethod', 'PaperlessBilling', 'ContentType', 
                    'MultiDeviceAccess', 'DeviceRegistered', 'GenrePreference', 'Gender', 
                    'ParentalControl', 'SubtitlesEnabled']
numerical_cols = ["AccountAge", "MonthlyCharges", "TotalCharges", "ViewingHoursPerWeek", 
                  "AverageViewingDuration", "ContentDownloadsPerMonth", "UserRating", 
                  "SupportTicketsPerMonth", "WatchlistSize", 'AvgViewingDuration_WatchlistSize', 'Total_viewing']

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Prepare training data
X_train = train_df[categorical_cols + numerical_cols]
Y_train = train_df["Churn"]

# Split the data for early stopping
X_train_split, X_val, Y_train_split, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Fit and transform the training data
X_train_split_transformed = preprocessor.fit_transform(X_train_split)
X_val_transformed = preprocessor.transform(X_val)

# Verify shapes and feature names
print(f"Shape of transformed training data: {X_train_split_transformed.shape}")
print(f"Shape of transformed validation data: {X_val_transformed.shape}")

# Get feature names after transformation
feature_names = list(preprocessor.named_transformers_['num'].feature_names_in_) + \
                list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
print(f"Number of feature names: {len(feature_names)}")
print(f"Feature names: {feature_names}")

# Define parameter distributions for RandomizedSearchCV
param_distributions = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}

# Create the XGBClassifier with early stopping
xgb_model = xgb.XGBClassifier(early_stopping_rounds=10, eval_metric="auc", use_label_encoder=False)

# Perform RandomizedSearchCV
randomized_search = RandomizedSearchCV(xgb_model, param_distributions, n_iter=10, cv=3, n_jobs=-1, verbose=2)

# Fit the model with parallel processing and verbose output
with joblib.parallel_backend('loky'):
    randomized_search.fit(X_train_split_transformed, Y_train_split, eval_set=[(X_val_transformed, Y_val)], verbose=True)

# Get feature importances
importances = randomized_search.best_estimator_.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
})

# Drop features with 0 importance
important_features_mask = feature_importance_df['importance'] > 0
important_features = feature_importance_df[important_features_mask]['feature'].tolist()

# Apply the same filter to the training, validation, and test data
X_train_split_transformed = X_train_split_transformed[:, important_features_mask]
X_val_transformed = X_val_transformed[:, important_features_mask]

print(f"Number of important features: {len(important_features)}")
print(f"Important features: {important_features}")

# Refit the preprocessor with only the important features
important_numerical_cols = [col for col in numerical_cols if col in important_features]
important_categorical_cols = [col for col in categorical_cols if any(col in feature for feature in important_features)]

important_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), important_numerical_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), important_categorical_cols)
    ]
)

# Transform the entire training data with only important features
X_train_transformed = important_preprocessor.fit_transform(X_train)
X_val_transformed = important_preprocessor.transform(X_val)

# Split the data again for early stopping
X_train_split_transformed, X_val_transformed, Y_train_split, Y_val = train_test_split(
    X_train_transformed, Y_train, test_size=0.2, random_state=42)

# Refit the model with the important features
randomized_search.fit(X_train_split_transformed, Y_train_split, eval_set=[(X_val_transformed, Y_val)], verbose=True)

# Transform the test data using the new preprocessor
X_test = test_df[categorical_cols + numerical_cols]
X_test_transformed = important_preprocessor.transform(X_test)

# Verify the shape of the transformed test data
print(f"Shape of transformed test data: {X_test_transformed.shape}")

# Predict probabilities for the test data using the transformed test data
xgb_pred_prob = randomized_search.predict_proba(X_test_transformed)[:, 1]

dict_df = {"CustomerID": test_df["CustomerID"],
           "predicted_probability": xgb_pred_prob}

prediction_df = pd.DataFrame(dict_df)
