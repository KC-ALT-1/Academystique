# Academystique - Stage 1: Student Marks Prediction Model



Welcome to the Academystique Stage 1 project repository! This project is designed to build a web portal that empowers students to improve their academic results by analyzing their past marks and understanding how daily chores affect their scores. This platform will offer valuable insights and guidance to both students and teachers, helping them identify weak areas and work on improving them.

## Modules Required

Before you get started, make sure you have the following Python modules installed:

- Flask
- scikit-learn
- RandomForestRegressor
- train_test_split
- OneHotEncoder
- SimpleImputer
- warnings
- pickle

You can install these modules using `pip`:

```bash
pip install flask scikit-learn
```

## Getting Started

Follow these steps to run the project on your local machine:

1. **Update CSV Path**:
   - In the `ml_model.py` file, update the path to the CSV data file to point to the location where you have stored the dataset on your machine.

2. **Generate Necessary .pkl Files**:
   - Run the `ml_model.py` file. This will generate the necessary .pkl files used by the prediction model.

3. **Run the Web Application**:
   - Execute the web application by running the `app.py` file.
   - Open a web browser and navigate to the local server (usually `http://localhost:5000`) to access the prediction webpage.

## Features

- Analyze past academic marks.
- Visualize the impact of daily chores on academic performance.
- Receive personalized tips and recommendations for improvement.
- Suitable for both students and teachers.

## Contributing

We welcome contributions to make this project even better. Feel free to submit issues, feature requests, or pull requests. Let's collaborate to enhance the Academystique platform!

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---





