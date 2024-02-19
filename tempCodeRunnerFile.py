# from flask import Flask, render_template, request, jsonify
# import pickle
# import numpy as np
# from sklearn.impute import SimpleImputer

# app = Flask(__name__)

# # Load the trained model and OneHotEncoder
# with open("random.pkl", "rb") as file:
#     model = pickle.load(file)

# # Load the OneHotEncoder instance used during training
# with open("encoder.pkl", "rb") as file:
#     encoder = pickle.load(file)

# subject_tips = {
#     'NLP': [
#         "Explore pre-trained word embeddings like Word2Vec and GloVe.",
#         "Understand the challenges of named entity recognition and sentiment analysis.",
#         "Stay updated on the latest advancements in NLP research."
#     ],
#     'ML': [
#         "Master the fundamentals of supervised and unsupervised learning algorithms.",
#         "Practice feature engineering to improve model performance.",
#         "Participate in Kaggle competitions to apply ML concepts to real-world problems."
#     ],
#     'BigData': [
#         "Learn distributed computing frameworks like Apache Hadoop and Apache Spark.",
#         "Understand how to handle and process large-scale datasets efficiently.",
#         "Explore data streaming technologies for real-time analytics."
#     ]
# }

# @app.route("/")
# def main():
#     return render_template("form.html")

# @app.route("/predict", methods=["POST"])
# def home():
#     if request.method == "POST":
       
#         name = request.form["a"]
#         inputs = [
#             float(request.form[f]) for f in ["b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
#         ]

#         # Transform inputs using the OneHotEncoder
#         transformed_inputs = encoder.transform([inputs]).toarray()

#         result = model.predict(transformed_inputs)
#         result = "{:.2f}".format(result[0])
        

#         # Format prediction output based on different scenarios
#         if (all(i == 0 for i in inputs[0:3])) or (inputs[5] > 5) or (inputs[8] < 40):
#             if all(i == 0 for i in inputs[0:3]):
#                 prediction_output = "Fail: All semesters' GPA are 0."
#             elif inputs[5] > 5:
#                 prediction_output = "Drop: Backlogs greater than 5."
#             else:
#                 prediction_output = "Expected Pointer less than 4: IQ less than 40."
#         else:
#             prediction_output = "Predicted GPA for {}: {}".format(name, result)

#         return render_template("form.html", abcc=prediction_output)

# @app.route("/get_tips")
# def get_tips():
#     subject = request.args.get("subject", "")
#     tips = subject_tips.get(subject, [])
#     return jsonify({"tips": tips})

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
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

subject_tips = {
    'NLP': [
        "Explore pre-trained word embeddings like Word2Vec and GloVe.",
        "Understand the challenges of named entity recognition and sentiment analysis.",
        "Stay updated on the latest advancements in NLP research."
    ],
    'ML': [
        "Master the fundamentals of supervised and unsupervised learning algorithms.",
        "Practice feature engineering to improve model performance.",
        "Participate in Kaggle competitions to apply ML concepts to real-world problems."
    ],
    'BigData': [
        "Learn distributed computing frameworks like Apache Hadoop and Apache Spark.",
        "Understand how to handle and process large-scale datasets efficiently.",
        "Explore data streaming technologies for real-time analytics."
    ]
}

@app.route("/")
def main():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def home():
    if request.method == "POST":
        name = request.form["a"]
        inputs = [
            float(request.form[f]) for f in ["b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        ]

        # Transform inputs using the OneHotEncoder
        transformed_inputs = encoder.transform([inputs]).toarray()

        result = model.predict(transformed_inputs)
      
        result = "{:.2f}".format(result[0])

        # Format prediction output based on different scenarios
        if (all(i == 0 for i in inputs[0:3])) or (inputs[5] > 5) or (inputs[8] < 40):
            if all(i == 0 for i in inputs[0:3]):
                prediction_output = "Fail: All semesters' GPA are 0."
                tips = []  # Define an empty list for tips
            elif inputs[5] > 5:
                prediction_output = "Drop: Backlogs greater than 5."
                tips = []  # Define an empty list for tips
            else:
                prediction_output = "Expected Pointer less than 4: IQ less than 40."
                tips = []  # Define an empty list for tips
        else:
            prediction_output = "Predicted GPA for {}: {}".format(name, result)
            
            # Fetch tips based on the selected subject
            subject = request.form.get("subject")  # Use request.form.get to avoid KeyError
            tips = subject_tips.get(subject, [])

        return render_template("form.html", abcc=prediction_output, tips=tips)
@app.route("/get_tips")
def get_tips():
    subject = request.args.get("subject", "")
    tips = subject_tips.get(subject, [])
    return jsonify({"tips": tips})

if __name__ == "__main__":
    app.run(debug=True)
