# EPFL Machine Learning - Project 1: Heart Attack Prediction

## Team Members
* Mohamed ZIRAR
* Ema GREGANOVA 
* Christophe BITAR

## Overview
This repository contains our code and report for Project 1 of the EPFL Machine Learning course (CS-433) Fall 2026. The objective of this project is to predict the risk of cardiovascular disease (MICHD) using health-related survey data from the BRFSS dataset.

## Repository Structure
* `implementations.py`: the six required regression implementations. It stays at the repository root because the official grader imports it from there.
* `run.py`: reproduces our best submission (AIcrowd F1 = 0.446) from the raw CSVs and writes `submission.csv`.
* `code/`: everything else.
  * `feature_engineering.py`: the notebook 01 cleaning with each excluded feature group as a switch, extra features (squares, interactions), removal of redundant columns, and regularized logistic regression solved with Newton's method.
  * `cross_validation.py`: stratified k-fold cross-validation, with optional under/oversampling of the training folds.
  * `metrics.py`: accuracy, precision, recall, F1 and the F1-optimal decision threshold.
  * `grid_search.py`: hyperparameter grid search on top of cross-validation (`python code/grid_search.py --model ridge_regression`).
  * `helpers.py`: data loading and submission helper functions (course template).
  * `scripts/`: local tests (`pytest code/scripts -q`).
* `notebooks/`:
  * `01_data_exploration_and_cleaning.ipynb`: missing values, cleaning, class imbalance.
  * `02_logistic_regression_experiment.ipynb`: first logistic-regression experiment.
  * `03_mean_squared_error_gd.ipynb`, `04_regularized_logistic_regression.ipynb`: MSE GD and regularized logistic regression, tuned on a stratified 80/20 split.
  * `05_feature_engineering_and_final_models.ipynb`: from the notebook 01 features to the final model, with stratified 5-fold cross-validation (threshold tuning, ridge, feature engineering, class imbalance, least squares, logistic regression with Newton's method).
* `dataset/`: training and test data (not versioned).
* `project1_description.pdf`: project brief and requirements.

## Requirements & Constraints
* **Python Version:** Python 3.x
* **Allowed Libraries:** Python standard libraries, `NumPy`, and visualization libraries (`matplotlib`, `seaborn`) used strictly for plotting. **No external ML libraries (e.g., Scikit-Learn, Pandas) are allowed.**

## Dataset
Download the project data from the [AIcrowd Project 1 dataset page](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1/dataset_files), under **Resources**.

Place the downloaded files in the `dataset/` directory before running the project.

## How to Run
Create the project environment from `grading_tests/environment.yml`, then run:

```bash
conda env create -f grading_tests/environment.yml -n project1-grading
conda activate project1-grading
python run.py
```

To work through the data preparation notebooks, open the files in `notebooks/` from
Jupyter or VS Code. Run the local implementation tests with:

```bash
conda run --no-capture-output -n project1-grading \
	python -m pytest code/scripts/test_implementations.py -q
```

On Apple Silicon, if the pinned `scikit-learn` version is unavailable, create the
test environment with only the packages required by the public tests:

```bash
conda create -n project1-grading python=3.9 numpy pytest gitpython black
```

## Automated Tests
The public tests check the function signatures and the expected results of the implementations.

Run all tests with:

```bash
./test.sh
```

The GitHub link format test is excluded for local testing because the project is passed as the local path `.`.

To run only the tests related to one function, use pytest's `-k` option:

```bash
./test.sh -k least_squares
```
