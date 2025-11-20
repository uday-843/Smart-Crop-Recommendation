import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocessDataset(dataset):
    """
    Preprocess the dataset by handling missing values, splitting into features and labels,
    encoding labels, and standardizing features.

    Parameters:
    dataset (pd.DataFrame): The input dataset containing features and labels.

    Returns:
    tuple: x_train, x_test, y_train, y_test, le, scaler, dataset
    """
    if dataset is None or 'label' not in dataset.columns:
        logging.error("Invalid dataset: Missing required 'label' column.")
        raise ValueError("Invalid dataset: Missing required 'label' column.")

    # Handle missing values for numerical features
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns
    dataset[numeric_columns] = dataset[numeric_columns].fillna(dataset[numeric_columns].mean())
    dataset.fillna(0, inplace=True)  # Ensure no NaN remains

    # Split dataset into features and labels
    X = dataset.drop(['label'], axis=1).values
    Y = dataset['label'].values

    if len(Y) == 0:
        logging.error("Dataset contains no valid labels. Cannot proceed.")
        raise ValueError("Dataset contains no valid labels. Cannot proceed.")

    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Encode labels only after splitting
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Standardize features using training set only
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)  # Fit only on training data
    x_test = scaler.transform(x_test)  # Transform test data using trained scaler

    logging.info("Dataset preprocessed successfully.")
    return x_train, x_test, y_train, y_test, le, scaler, dataset

def train_models(x_train, y_train, x_test, y_test, dataset, le, scaler):
    """
    Train individual models and a hybrid model (Voting Classifier), then save them along with accuracies.

    Parameters:
    x_train (np.array): Training features.
    y_train (np.array): Training labels.
    x_test (np.array): Testing features.
    y_test (np.array): Testing labels.
    dataset (pd.DataFrame): Original dataset.
    le (LabelEncoder): Label encoder.
    scaler (StandardScaler): Feature scaler.

    Returns:
    tuple: hybrid_model, accuracies
    """
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=0),
        "SVM": SVC(probability=True, random_state=0),
        "NaiveBayes": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, activation='relu', solver='adam', random_state=0)
    }

    accuracies = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracies[name] = accuracy_score(y_test, y_pred) * 100
        logging.info(f"{name} model trained with accuracy: {accuracies[name]:.2f}%")

    # Train Hybrid Model (Voting Classifier)
    hybrid_model = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    hybrid_model.fit(x_train, y_train)
    y_pred_hybrid = hybrid_model.predict(x_test)
    accuracies["Hybrid"] = accuracy_score(y_test, y_pred_hybrid) * 100
    logging.info(f"Hybrid model trained with accuracy: {accuracies['Hybrid']:.2f}%")

    # Save models & accuracies
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    joblib.dump(hybrid_model, "saved_models/hybrid_model.pkl")
    joblib.dump(le, "saved_models/label_encoder.pkl")
    joblib.dump(scaler, "saved_models/scaler.pkl")
    joblib.dump(accuracies, "saved_models/accuracies.pkl")
    logging.info("Models and accuracies saved successfully.")

    return hybrid_model, accuracies

def save_plots(dataset, y_test, y_pred, model=None, accuracies=None):
    """
    Generate and save various plots for data visualization and model evaluation.

    Parameters:
    dataset (pd.DataFrame): The original dataset.
    y_test (np.array): True labels for testing set.
    y_pred (np.array): Predicted labels for testing set.
    model (VotingClassifier): The trained hybrid model (optional).
    accuracies (dict): Dictionary of model accuracies (optional).
    """
    plots_path = "static/plots"
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    # Feature Distribution Plot
    plt.figure(figsize=(12, 8))
    sns.histplot(dataset.drop(columns=['label']), kde=True)
    plt.title("Feature Distribution")
    plt.savefig(os.path.join(plots_path, "feature_distribution.png"), bbox_inches='tight')
    plt.close()
    logging.info("Feature distribution plot saved.")

    # Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(dataset.drop(columns=['label']).corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(plots_path, "correlation_heatmap.png"), bbox_inches='tight')
    plt.close()
    logging.info("Correlation heatmap saved.")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plots_path, "confusion_matrix.png"), bbox_inches='tight')
    plt.close()
    logging.info("Confusion matrix saved.")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred, pos_label=1)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label="Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plots_path, "precision_recall_curve.png"), bbox_inches='tight')
    plt.close()
    logging.info("Precision-recall curve saved.")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plots_path, "roc_curve.png"), bbox_inches='tight')
    plt.close()
    logging.info("ROC curve saved.")

    # Model Comparison Plot
    if accuracies:
        plt.figure(figsize=(10, 6))
        model_names = list(accuracies.keys())
        accuracy_values = list(accuracies.values())
        sns.barplot(x=accuracy_values, y=model_names, hue=model_names, palette="viridis", legend=False)
        plt.xlabel("Accuracy (%)")
        plt.ylabel("Models")
        plt.title("Model Comparison: Individual vs Hybrid")
        plt.xlim(0, 100)
        plt.savefig(os.path.join(plots_path, "model_comparison.png"), bbox_inches='tight')
        plt.close()
        logging.info("Model comparison plot saved.")

def predict(models=None, le=None, scaler=None, input_values=None):
    """
    Predict the crop recommendation based on input values using the trained model.

    Parameters:
    models (VotingClassifier): The trained hybrid model.
    le (LabelEncoder): The label encoder.
    scaler (StandardScaler): The feature scaler.
    input_values (list): List of input features.

    Returns:
    str: The predicted crop name.
    """
    if models is None or le is None or scaler is None:
        try:
            models = joblib.load("saved_models/hybrid_model.pkl")
            le = joblib.load("saved_models/label_encoder.pkl")
            scaler = joblib.load("saved_models/scaler.pkl")
        except Exception as e:
            logging.error(f"Failed to load models or encoders: {e}")
            raise

    input_values = np.array(input_values, dtype=float).reshape(1, -1)
    input_values = scaler.transform(input_values)
    pred = models.predict(input_values)[0]
    return le.inverse_transform([pred])[0]