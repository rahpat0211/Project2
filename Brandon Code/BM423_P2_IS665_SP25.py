# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset and filter for PSP and PS2
df = pd.read_csv("/Users/user/Desktop/cleaned_vgsales (1).csv")
df = df[df['Platform'].isin(['PSP', 'PS2'])]  # Filter to include only PSP and PS2 games

# 1. DATA CLEANING
# Drop rows with missing values in 'Year' or 'Publisher' columns
df.dropna(subset=['Year', 'Publisher'], inplace=True)

# Convert 'Year' column to integer type
df['Year'] = df['Year'].astype(int)

# Define a function to classify games based on Global_Sales into sales tiers
def classify_sales(sales):
    if sales >= 10:
        return 'High'
    elif sales >= 1:
        return 'Medium'
    else:
        return 'Low'

# Create a new target column 'Sales_Tier' based on Global_Sales
df['Sales_Tier'] = df['Global_Sales'].apply(classify_sales)

# Encode categorical variables using LabelEncoder
le_platform = LabelEncoder()
df['Platform_Enc'] = le_platform.fit_transform(df['Platform'])

le_genre = LabelEncoder()
df['Genre_Enc'] = le_genre.fit_transform(df['Genre'])

le_publisher = LabelEncoder()
df['Publisher_Enc'] = le_publisher.fit_transform(df['Publisher'])

# 2. EXPLORATORY ANALYSIS (Medium & Low Sales Tier only)
# Filter for relevant Platform and Sales_Tier combinations
grouped_df = df[((df['Sales_Tier'] == 'Medium') & (df['Platform'].isin(['PS2', 'PSP']))) |
                ((df['Sales_Tier'] == 'Low') & (df['Platform'].isin(['PS2', 'PSP'])))]

# Create a count DataFrame for plotting
plot_df = grouped_df.groupby(['Platform', 'Sales_Tier']).size().reset_index(name='Count')

# Plot bar graph for specified platform and sales tier combinations
plt.figure(figsize=(10, 6))
sns.barplot(data=plot_df, x='Platform', y='Count', hue='Sales_Tier', palette='Set2')
plt.title("Sales Tier Distribution for PSP and PS2")
plt.xlabel("Platform")
plt.ylabel("Number of Games")
plt.legend(title="Sales Tier")
plt.show()

# 3. k-NN CLASSIFICATION
# Select encoded features to use in the model
features = ['Platform_Enc', 'Genre_Enc', 'Publisher_Enc']
X = df[features]
y = df['Sales_Tier']

# Standardize feature values for better k-NN performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Use GridSearchCV to find the best value for k (number of neighbors)
param_grid = {'n_neighbors': list(range(3, 21))}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)  # Fit the grid search to training data

# Extract the best model from grid search
best_knn = grid.best_estimator_

# Predict the sales tier on the test set
y_pred = best_knn.predict(X_test)

# 4. RESULTS AND VISUALIZATION
# Print the best value of k and the classification report
print("Best k:", grid.best_params_['n_neighbors'])
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Generate and sort unique labels to ensure confusion matrix includes all categories
labels = sorted(df['Sales_Tier'].unique())

# Plot the confusion matrix to visualize prediction performance
cm = confusion_matrix(y_test, y_pred, labels=labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Display the average encoded values for each feature grouped by sales tier
grouped = df.groupby('Sales_Tier')[features].mean()
print("\nAverage Encoded Feature Values by Sales Tier:\n", grouped)
