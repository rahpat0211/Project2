# Import Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import permutation_importance

# ----------------------------------------------------------------------------------------------------------------------

"""
Data Preprocessing
"""

class PreProcessData:
    def __init__(self):
        df = pd.read_csv("vgsales.csv")  # Read the raw dataset
        self.df = df[df['Platform'].isin(['PSP', 'PS2'])].copy()  # Only contain information on 2 Main Sony Platforms and create a copy

        print(f"Null Information:\n{self.df.isnull().sum()}")

    def visualize_null_values(self, outlier=False):
        if outlier:
            plt.figure(1)  # Explicitly create figure 1
            plt.clf()  # Clear any existing content in figure 1
            sns.boxplot(self.df['Global_Sales'], color='orange')
            plt.title('Outlier Detection in Global Sales')
            plt.xlabel('Global Sales')
            plt.tight_layout()
            plt.show()
            print('-' * 100)
        else:
            plt.figure(2)  # Explicitly create figure 2
            plt.clf()  # Clear any existing content in figure 2
            sns.heatmap(self.df.isnull(), cbar=False, cmap='viridis')
            plt.title("Missing Values")
            plt.tight_layout()
            plt.show()
            print('\n')

    def process_null_values(self):
        """
        Drop the records for Publisher where the values are NaN (Do not need to incorporate in model)
        Take the previous year into account for NaN values for Year Column
        """

        self.df = self.df.dropna(subset=['Publisher']).copy()  # Create a copy after dropping NaNs
        self.df['Year'] = self.df['Year'].ffill()

        # Convert the data type for Year back to INT
        self.df['Year'] = self.df['Year'].astype(int)

    def remove_outlier(self):
        return self.df[self.df['Global_Sales'] < 9]

# ----------------------------------------------------------------------------------------------------------------------

preprocess = PreProcessData()
preprocess.visualize_null_values()  # Visualize the NaN values in the Heatmap
preprocess.process_null_values()  # Process the NaN Values
preprocess.visualize_null_values(outlier=True)  # Visualize the outlier through BoxPlot
df_cleaned = preprocess.remove_outlier()

# ----------------------------------------------------------------------------------------------------------------------

# Random Forest Regression

class RandomForestRegression:

    def __init__(self, df):
        df_cleaned = df

        for platform in df_cleaned['Platform'].unique():
            print(f"\n{'=' * 50}\nProcessing data for Platform: {platform}\n{'=' * 50}")
            df_platform = df_cleaned[df_cleaned['Platform'] == platform].copy()

            # Set X and y data for the current platform
            X = df_platform.drop(columns=["Global_Sales", "Name", "Rank", "Platform"])  # Exclude 'Platform' as it's constant
            y = df_platform['Global_Sales']

            # Set Categorical Columns (excluding 'Platform' now)
            categorical_columns = ["Genre", "Publisher"]

            # Encode the categories
            label_encoders = self.encode_labels(X, categorical_columns)

            # Train and evaluate the model for the current platform
            self.train_evaluate_platform(X, y, platform)

    @staticmethod
    def encode_labels(X, categorical_columns):
        label_encoders = {}
        for col in categorical_columns:
            if col in X.columns:  # Ensure the column exists after potential filtering
                le = LabelEncoder()
                X.loc[:, col] = le.fit_transform(X[col])  # Use .loc for safe modification
                label_encoders[col] = le  # Store the encoder
        return label_encoders

    def train_evaluate_platform(self, X, y, platform):
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # --- Optimized Random Forest ---
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
        )
        rf.fit(X_train, y_train)

        print(f"\nModel trained and fitted for Platform: {platform}")
        y_pred = rf.predict(X_test)
        self.evaluate_model(y_test, y_pred, platform)

        # Feature Importance
        self.predict_features(rf, X.columns, X_test, y_test, platform)

    @staticmethod
    def evaluate_model(y_test, y_pred, platform):
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\n--- Evaluation for Platform: {platform} ---")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"R² Score: {r2:.4f}")

    def predict_features(self, rf, features, X_test, y_test, platform):
        # Feature Importance (using built-in)
        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]  # Sort in descending order

        print(f"\nFeature Importances (Random Forest) for Platform: {platform}")
        for i in sorted_idx:
            print(f"{features[i]}: {importances[i]:.4f}")

        # Visualize feature importances
        plt.figure(figsize=(10, 6))
        plt.title(f"Random Forest Feature Importance for {platform}")
        plt.bar(range(X_test.shape[1]), importances[sorted_idx], align="center", color='skyblue')
        plt.xticks(range(X_test.shape[1]), [features[i] for i in sorted_idx], rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()
        print('-' * 100)

        # Permutation Importance
        result = permutation_importance(
            rf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        perm_sorted_idx = result.importances_mean.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        plt.title(f"Permutation Feature Importance for {platform}")
        plt.bar(range(X_test.shape[1]), result.importances_mean[perm_sorted_idx], align="center", color='lightcoral')
        plt.xticks(range(X_test.shape[1]), features[perm_sorted_idx], rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------

RFC = RandomForestRegression(df=df_cleaned)