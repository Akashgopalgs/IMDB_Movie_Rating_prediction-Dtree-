import streamlit as st
import pandas as pd
import pickle

# Load your pre-trained model
with open("decision_tree_model.pkl", "rb") as file:
    dt_regressor = pickle.load(file)

# Define the Streamlit app
def main():
    st.title("Movie Rating Prediction App")
    st.write("This app predicts the movie rating based on input features.")

    # Set min and max values based on your preprocessed dataset
    feature_ranges = {
        'Year': (2000, 2020),  # Example values; replace with actual dataset min/max
        'Votes': (500, 500000),
        'Duration': (60, 180),
        'G_mean_rat': (3.0, 8.0),
        'Dir_enc': (4.0, 9.0),
        'A1_enc': (3.5, 9.0),
        'A2_enc': (3.5, 9.0),
        'A3_enc': (3.5, 9.0)
    }

    # Create inputs for user to enter features with appropriate min/max
    year = st.slider("Year", min_value=feature_ranges['Year'][0], max_value=feature_ranges['Year'][1],value=2010, step=1)
    votes = st.slider("Votes", min_value=feature_ranges['Votes'][0], max_value=feature_ranges['Votes'][1],value=105000, step=1000)
    duration = st.slider("Duration (in minutes)", min_value=feature_ranges['Duration'][0], max_value=feature_ranges['Duration'][1], value=120,step=1)
    g_mean_rat = st.slider("Genre Mean Rating", min_value=feature_ranges['G_mean_rat'][0], max_value=feature_ranges['G_mean_rat'][1],value=5.0, step=0.1)
    dir_enc = st.slider("Director Rating", min_value=feature_ranges['Dir_enc'][0], max_value=feature_ranges['Dir_enc'][1], value=6.0,step=0.1)
    a1_enc = st.slider("Main Actor or Artress Rating", min_value=feature_ranges['A1_enc'][0], max_value=feature_ranges['A1_enc'][1],value=6.0, step=0.1)
    a2_enc = st.slider("Second Main ctor or Artress Rating", min_value=feature_ranges['A2_enc'][0], max_value=feature_ranges['A2_enc'][1],value=6.0, step=0.1)
    a3_enc = st.slider("Third Main Actor or Artress Rating", min_value=feature_ranges['A3_enc'][0], max_value=feature_ranges['A3_enc'][1],value=6.0, step=0.1)

    # Collect the inputs into a DataFrame
    input_data = pd.DataFrame({
        'Year': [year],
        'Votes': [votes],
        'Duration': [duration],
        'G_mean_rat': [g_mean_rat],
        'Dir_enc': [dir_enc],
        'A1_enc': [a1_enc],
        'A2_enc': [a2_enc],
        'A3_enc': [a3_enc]
    })

    # Predict using the trained model
    if st.button("Predict Rating"):
        prediction = dt_regressor.predict(input_data)
        st.write(f"Predicted Rating: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()
