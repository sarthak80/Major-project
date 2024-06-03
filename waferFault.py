import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image
import pymongo
import joblib

# Load your trained model (replace 'your_model.joblib' with your actual model file)
forest_model=RandomForestClassifier()

# Initial data processing (commented out as it might be run separately for training purposes)

data = pd.read_csv("Train.csv")
x = pd.concat([data['feature_969'], data['feature_1048'], data['feature_1144'], data['feature_1154'],
               data['feature_1155'], data['feature_1199'], data['feature_1219'], data['feature_1244'],
               data['feature_1345'], data['feature_1400'], data['feature_1423'], data['feature_1425']], axis=1)
y = data['Class']
for column in x.columns:
    x[column] = (x[column] - x[column].min()) / (x[column].max() - x[column].min()) 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
forest_model = RandomForestClassifier()
forest_model.fit(X_train, y_train)
random_pred = forest_model.predict(X_test)
random = accuracy_score(y_test, random_pred)
print("Random Forest accuracy : ", round(random*100, 2))


class PlacementPredictorApp:
    def __init__(self):
        self.labels = [
            "feature_969", "feature_1048", "feature_1144", "feature_1154", "feature_1155",
            "feature_1199", "feature_1219", "feature_1244", "feature_1345", "feature_1400",
            "feature_1423", "feature_1425"
        ]
        self.input_values = {}
        self.client = pymongo.MongoClient("mongodb+srv://sarthaktechtitudetribe:mongodb@cluster0.n8atzie.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        self.db = self.client['sarthak']
        self.col = self.db['table']
        
    def get_user_input(self):
        for label in self.labels:
            a=st.number_input(label)
            if a< 0.5 and a>0:
                a=0
            if a> 0.5 and a<1:
                a=1
            self.input_values[label] = a

    def predict_placement(self):
        # Convert input values to a DataFrame
        data = pd.DataFrame([self.input_values.values()], columns=self.labels)

        # Insert data into MongoDB
        self.col.insert_one(self.input_values)

        # Make a prediction
        prediction = forest_model.predict(data)
        
        # Interpret and display the prediction
        if prediction[0] == 1:
            result = "Fault exists"
        else:
            result = "No fault"
        
        st.write(f"Wafer Fault Prediction: {result}")

    def run(self):
        st.title("Wafer Fault Predictor")

        # Load the image
        image = Image.open("Why-Semiconductors-Are-a-Really-Big-Deal.png")
        st.image(image, caption="Why Semiconductors Are a Really Big Deal", use_column_width=True)

        # Get user input
        st.subheader("Enter Feature Values:")
        self.get_user_input()

        # Button to trigger prediction
        if st.button("Predict Fault"):
            self.predict_placement()
 
if __name__ == "__main__":
    app = PlacementPredictorApp()
    app.run()
