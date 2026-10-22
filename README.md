# EPFL Machine Learning - Project 1: Heart Attack Prediction

## Team Members
* Mohamed ZIRAR
* Ema GREGANOVA 
* Christophe BITAR

## Overview
This repository contains our code and report for Project 1 of the EPFL Machine Learning course (CS-433) Fall 2026. The objective of this project is to predict the risk of cardiovascular disease (MICHD) using health-related survey data from the BRFSS dataset.

## Repository Structure
* `implementations.py`: linear and logistic regression implementations. It stays at the repository root because the official grader imports it from there.
* `helpers.py`: data loading and submission helper functions (course template).
* `metrics.py`: accuracy, precision, recall, F1 and the F1-optimal decision threshold.
* `cross_validation.py`: stratified k-fold cross-validation for all six models, with optional under/oversampling of the training folds.
* `grid_search.py`: hyperparameter grid search on top of cross-validation (`python grid_search.py --model ridge_regression`).
* `code/scripts/test_implementations.py`: additional local tests for the implemented functions.
* `code/scripts/test_model_selection.py`: tests for the metrics, cross-validation and grid search.
* `notebooks/`: data cleaning and missing-value analysis notebooks.
* `02_ChrisNotebook1.ipynb`: first logistic-regression experiment (grid search + submission).
* `notebooks/03_mean_squared_error_gd.ipynb`, `notebooks/04_regularized_logistic_regression.ipynb`: MSE GD and regularized logistic regression, tuned on a stratified 80/20 split, with submissions.
* `notebooks/05_model_selection_cv.ipynb`: comparison of all six models with stratified 5-fold cross-validation (same hyperparameter grids as notebooks 03 and 04).
* `dataset/`: training, test, and cleaned data files.
* `run.py`: entry point for running the project experiments.
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
