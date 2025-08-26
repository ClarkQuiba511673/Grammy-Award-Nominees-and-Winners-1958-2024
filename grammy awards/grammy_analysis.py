import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Set random seed for reproducibility
np.random.seed(42)

# 1. Load and Clean Data
def load_data():
    """Load and preprocess the Grammy Awards dataset"""
    df = pd.read_csv('data/Grammy Award Nominees and Winners 1958-2024.csv')
    
    # Basic cleaning
    df = df.dropna(subset=['Award Name', 'Nominee', 'Winner'])  # Remove rows with missing critical values
    df['Winner'] = df['Winner'].astype(bool)  # Convert to boolean
    
    # Feature engineering
    df['Year_norm'] = (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min())
    
    return df

# 2. Exploratory Data Analysis
def perform_eda(df):
    """Perform exploratory data analysis and visualization"""
    print("\n=== Basic Dataset Info ===")
    print(f"Total records: {len(df)}")
    print(f"Years covered: {df['Year'].min()} to {df['Year'].max()}")
    print(f"Unique award categories: {df['Award Name'].nunique()}")
    
    # Set style for plots
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plot 1: Nominations by Year
    plt.subplot(2, 2, 1)
    sns.countplot(x='Year', data=df)
    plt.title('Nominations by Year')
    plt.xticks(rotation=90)
    
    # Plot 2: Winners vs Nominees
    plt.subplot(2, 2, 2)
    winner_counts = df['Winner'].value_counts()
    plt.pie(winner_counts, labels=['Nominees', 'Winners'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
    plt.title('Winners vs Nominees')
    
    # Plot 3: Top Award Categories
    plt.subplot(2, 2, 3)
    top_categories = df['Award Name'].value_counts().head(10)
    sns.barplot(x=top_categories.values, y=top_categories.index, palette='viridis')
    plt.title('Top 10 Award Categories')
    plt.xlabel('Count')
    
    # Plot 4: Win Rate by Year
    plt.subplot(2, 2, 4)
    win_rates = df.groupby('Year')['Winner'].mean() * 100
    sns.lineplot(x=win_rates.index, y=win_rates.values)
    plt.title('Win Rate by Year (%)')
    plt.ylabel('Win Rate (%)')
    
    plt.tight_layout()
    plt.savefig('grammy_eda.png')
    plt.show()

# 3. Prepare Data for Modeling
def prepare_model_data(df):
    """Prepare data for machine learning models"""
    # Encode award names
    le = LabelEncoder()
    df['Award_encoded'] = le.fit_transform(df['Award Name'])
    
    # Features and target
    X = df[['Year_norm', 'Award_encoded']]
    y = df['Winner']
    
    return X, y, le

# 4. Train and Evaluate Random Forest Model
def train_random_forest(X, y):
    """Train and evaluate Random Forest classifier"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    print("\n=== Random Forest Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return rf

# 5. Cluster Award Categories
def cluster_awards(df):
    """Cluster award categories using K-Means"""
    award_names = df['Award Name'].unique()
    
    # Vectorize award names
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(award_names)
    
    # Find optimal k
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('elbow_plot.png')
    plt.show()
    
    # Apply K-means with selected k
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Print cluster examples
    print("\n=== Award Category Clusters ===")
    for i in range(5):
        cluster_awards = award_names[clusters == i]
        print(f"\nCluster {i+1} ({len(cluster_awards)} awards):")
        print(", ".join(cluster_awards[:5]) + ("..." if len(cluster_awards) > 5 else ""))
    
    return kmeans

# 6. Save Models and Artifacts
def save_models(rf_model, le_encoder, kmeans_model=None):
    """Save trained models and encoders"""
    os.makedirs('models', exist_ok=True)
    
    # Save Random Forest and LabelEncoder
    with open('models/grammy_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le_encoder, f)
    
    # Save KMeans if provided
    if kmeans_model:
        with open('models/award_clusterer.pkl', 'wb') as f:
            pickle.dump(kmeans_model, f)
    
    print("\nModels saved to 'models' directory")

# Main Execution
if __name__ == "__main__":
    print("=== Grammy Awards Data Analysis ===")
    
    # 1. Load and clean data
    df = load_data()
    
    # 2. Perform EDA
    perform_eda(df)
    
    # 3. Prepare modeling data
    X, y, le = prepare_model_data(df)
    
    # 4. Train Random Forest
    rf_model = train_random_forest(X, y)
    
    # 5. Cluster award categories
    kmeans_model = cluster_awards(df)
    
    # 6. Save all models
    save_models(rf_model, le, kmeans_model)
    
    print("\nAnalysis complete!")