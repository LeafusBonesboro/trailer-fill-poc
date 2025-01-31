from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load the trained model
def load_model(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model("./model/trailer_fill_model_53ft.pkl")


# Define Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        carton_count = int(request.form.get("carton_count"))
        load_type = request.form.get("load_type")

        # Encode load type as numerical value
        load_type_encoded = {"IC": 0, "Overhead": 1, "Chewy": 2, "Combination": 3}[load_type]

        # Calculate estimated cargo volume
        avg_box_volume = {"IC": 6.0, "Overhead": 3.0, "Chewy": 4.5, "Combination": 4.0}[load_type]
        estimated_cargo_volume = carton_count * avg_box_volume

        # Create input data
        input_data = pd.DataFrame([[carton_count, load_type_encoded, estimated_cargo_volume]],
                                  columns=["Carton Count", "Load Type Encoded", "Estimated Cargo Volume"])

        # Make prediction
        predicted_fill = model.predict(input_data)[0]
        prediction = f"Predicted Trailer Fill: {predicted_fill:.2f}%"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
