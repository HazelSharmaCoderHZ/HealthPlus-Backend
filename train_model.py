import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data=pd.read_csv("diabetes.csv")
x=data.drop("Outcome", axis=1)
y=data["Outcome"]

x_train, x_test, y_train, y_test= train_test_split(x , y, test_size=0.2, random_state=42)

pipeline=Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

pipeline.fit(x_train, y_train)
y_pred=pipeline.predict(x_test)
accuracy=accuracy_score(y_test, y_pred)

print(f"Improved Model Accuracy: {accuracy*100:.2f}%")

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/diabetes_model.pkl")

print("Scaled Model Saved Successfully!!!")