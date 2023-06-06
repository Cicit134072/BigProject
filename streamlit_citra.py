import streamlit as st
import pickle
from sklearn.tree import DecisionTreeClassifier

# Load the DTREE model from file
model = pickle.load(open('dtree.pkl', 'rb'))
scaler = pickle.load(open('sc.pkl', 'rb'))

# Function to make predictions
def predict(preprocessed_data, children, region):
    # Make predictions using the loaded model
    predictions = model.predict([[preprocessed_data[0][0], preprocessed_data[0][1], preprocessed_data[0][2], children, region]])
    # Map the prediction to the corresponding result
    if predictions[0] == 0:
        return "NO"
    else:
        return "YES"


# Main Function
def main():
    # Set the title and description of your web app
    st.title("DTREE Prediction App")
    st.write("Enter the input data to get predictions.")

    # Get user input
    age = st.number_input("Age", value=0)
    bmi = st.number_input("Bmi", value=0.00)
    children = st.number_input("Children", value=0, min_value=0)
    region = st.number_input("Region", value=0)
    charges = st.number_input("Charges", value=0.0)

    # Preprocess the input data
    preprocessed_data = scaler.transform([[age, bmi, charges]])

    # Make Predictions
    if st.button("Predict"):
        predictions = predict(preprocessed_data, children, region)
        st.write('Predictions:', predictions)


if __name__ == '__main__':
    main()
