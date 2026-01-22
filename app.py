# Import Flask modules for web app
from flask import Flask, render_template, request
# Import joblib for loading saved models
import joblib
# Import numpy for array operations
import numpy as np

# Create Flask application instance
app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("model/wine_cultivar_model.pkl")
# Load the feature scaler
scaler = joblib.load("model/scaler.pkl")

# Define route for home page, handling GET and POST requests
@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize prediction variable
    prediction = None

    # Check if the request method is POST (form submitted)
    if request.method == "POST":
        # Extract input values from form and convert to floats
        values = [
            float(request.form["alcohol"]),
            float(request.form["malic_acid"]),
            float(request.form["magnesium"]),
            float(request.form["total_phenols"]),
            float(request.form["color_intensity"]),
            float(request.form["proline"])
        ]

        # Scale the input features using the loaded scaler
        scaled = scaler.transform([values])
        # Make prediction using the model
        pred = model.predict(scaled)[0]
        # Format the prediction result
        prediction = f"Cultivar {pred + 1}"

    # Render the HTML template with the prediction
    return render_template("index.html", prediction=prediction)

# Run the app if this script is executed directly
if __name__ == "__main__":
    # Start the Flask development server with debug mode
    app.run(debug=True)
