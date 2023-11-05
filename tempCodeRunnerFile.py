from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the trained model and OneHotEncoder
with open("random.pkl", "rb") as file:
    model = pickle.load(file)

# Load the OneHotEncoder instance used during training
with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def home():
    if request.method == "POST":
        # Get input data from the form
        name = request.form["a"]
        inputs = [
            float(request.form[f]) for f in ["b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        ]

        # Transform inputs using the OneHotEncoder
        transformed_inputs = encoder.transform([inputs]).toarray()

        # Make prediction using the model
        result = model.predict(transformed_inputs)
        print(result)
        result = "{:.2f}".format(result[0])

        # Format prediction output based on different scenarios
        if (all(i == 0 for i in inputs[0:3])) or (inputs[5] > 5) or (inputs[8] < 40):
            if all(i == 0 for i in inputs[0:3]):
                prediction_output = "Fail: All semesters' GPA are 0."
            elif inputs[5] > 5:
                prediction_output = "Drop: Backlogs greater than 5."
            else:
                prediction_output = "Expected Pointer less than 4: IQ less than 40."
        else:
            prediction_output = "Predicted GPA for {}: {}".format(name, result)

        return render_template("index.html", abcc=prediction_output)

if __name__ == "__main__":
    app.run(debug=True)
