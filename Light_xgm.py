import numpy as np
import xgboost as xgb
import joblib  # For saving and loading the model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset (replace with actual path)
train_features = np.load (r"C:\Users\User\Desktop\Ransomware_Detection\ember_dataset_2018_2\ember2018\ember_model_2018.txt", allow_pickle= True)
train_labels = np.load("\\Users\\User\\Desktop\\Ransomware_Detection\\ember_dataset_2018_2\\ember2018")

# Remove unknown labels (-1) if present
valid_indices = train_labels != -1
X = train_features[valid_indices]
y = train_labels[valid_indices]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic", 
    eval_metric="logloss", 
    use_label_encoder=False
)

# Train the model
xgb_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(xgb_model, "xgboost_ransomware_model.pkl")
print("Model saved successfully as 'xgboost_ransomware_model.pkl'")

# Load the model (for future use)
loaded_model = joblib.load("xgboost_ransomware_model.pkl")
print("Model loaded successfully")

# Make predictions using the loaded model
y_pred = loaded_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
