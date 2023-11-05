import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
import pickle

warnings.filterwarnings("ignore")

data = pd.read_csv("D:/portfolio/xyz/3.3.csv")
data.drop(["FIRST NAME", "LAST NAME", "SEX"], axis="columns", inplace=True)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy="mean")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split data and preprocess for the model
feature = data_imputed[
    [
        "Sem3",
        "Sem4",
        "Sem5",
        "Travel-time",
        "Studytime hrs",
        "Backlogs",
        "Freetime",
        "Outdoor Activity",
        "IQ",
        "Daily Activities",
    ]
]
target = data_imputed["GPA"]

# Create and fit the OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
nfeatures = encoder.fit_transform(feature).toarray()

x_train, x_test, y_train, y_test = train_test_split(nfeatures, target, random_state=120)

# Train the model
model = RandomForestRegressor()
model.fit(x_train, y_train)

# Save the model and OneHotEncoder to a file
with open("random.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the OneHotEncoder to another file
with open("encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

