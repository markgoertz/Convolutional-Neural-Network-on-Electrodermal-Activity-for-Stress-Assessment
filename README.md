![Keras](https://a11ybadges.com/badge?logo=keras)
![Python](https://a11ybadges.com/badge?logo=python)
![GitHub](https://a11ybadges.com/badge?logo=github)

[![CI](https://github.com/markgoertz/Convolutional-Neural-Network-on-Electrodermal-Activity-for-Stress-Assessment/actions/workflows/CI.yml/badge.svg)](https://github.com/markgoertz/Convolutional-Neural-Network-on-Electrodermal-Activity-for-Stress-Assessment/actions/workflows/CI.yml)


# 1D CNN for Electrodermal Activity Signal Classification
This project implements a 1D Convolutional Neural Network (CNN) for classifying signals in raw Electrodermal Activity (EDA) data. EDA is a physiological measure of sweat gland activity, which can be indicative of emotional arousal or cognitive workload.

## Project Structure
```
.
├── README.md (this file)
├── data/
│   └── merged_data.csv  (EDA dataset)
├── code/
│   └── eda_signal_classification.ipynb  (Jupyter notebook for model development)
├── .github/
|        └── workflows
│           └── CI.yml  (CI-pipeline)
└── requirements.txt  (dependencies for project)
```

## Getting started
1. **Clone the repository:**
```
git clone https://github.com/markgoertz/Convolutional-Neural-Network-on-Electrodermal-Activity-for-Stress-Assessment .git
```
2. **Install dependencies**
```
pip install -r requirements.txt
```
3. **Prepare the data**
* Place the pre-processed EDA data using ```unzip-script.ipynb```, called ```merged_data.csv``` in the data folder. Ensure it's a CSV file with features and labels.

4. **Train the model:**
* Open ```eda_signal_classification.ipynb``` in a Jupyter notebook environment.
* This notebook guides you through data loading, model building, training, and evaluation.

5. **Best Models**
* During the training process, the best model (based on the epochs and folds) get chosen. And saved as ```Best_model.keras```.
* Modify the notebook in a new branch to load the model and make prediction on new data.

## Dependencies
This project uses various libraries like Tensorflow for building the CNN. The specific dependencies are listed in `requirements.txt`
<div align="center">
	<code><img width="100" src="https://user-images.githubusercontent.com/25181517/183914128-3fc88b4a-4ac1-40e6-9443-9a30182379b7.png" alt="Jupyter Notebook" title="Jupyter Notebook"/></code>
	<code><img width="100" src="https://user-images.githubusercontent.com/25181517/183423507-c056a6f9-1ba8-4312-a350-19bcbc5a8697.png" alt="Python" title="Python"/></code>
	<code><img width="100" src="https://user-images.githubusercontent.com/25181517/223639822-2a01e63a-a7f9-4a39-8930-61431541bc06.png" alt="TensorFlow" title="TensorFlow"/></code>
</div>

## Screen shots and results

### Data preparation
<img src="https://github.com/markgoertz/Convolutional-Neural-Network-on-Electrodermal-Activity-for-Stress-Assessment/blob/main/code/dataset_distribution.png"/>

### Folds training
<img src="https://github.com/markgoertz/Convolutional-Neural-Network-on-Electrodermal-Activity-for-Stress-Assessment/blob/main/code/folds.png"/>

### Confusion Matrix
<img src="https://github.com/markgoertz/Convolutional-Neural-Network-on-Electrodermal-Activity-for-Stress-Assessment/blob/main/code/model_results.png"/>

