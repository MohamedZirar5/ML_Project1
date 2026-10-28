# EPFL Machine Learning (CS-433) - Project 1: Heart Attack Prediction

[![Course](https://img.shields.io/badge/EPFL-CS433-red.svg)](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-completed-success.svg)]()

## Team Members
- **Mohamed ZIRAR**
- **Ema GREGANOVA** 
- **Christophe BITAR**

---

## Overview
This repository contains the source code and final report for **Project 1** of the EPFL Machine Learning course (Fall 2026). The objective is to predict the risk of cardiovascular disease / myocardial infarction (MICHD) using health-related survey data from the 2015 BRFSS dataset ($328,135$ observations, $321$ raw features).

- **Best AIcrowd F1-Score:** `0.446`
- **Group Name:** ML_isAllYouNeed 
- **Key Method:** Regularized Logistic Regression optimized via Newton's method with advanced feature engineering and decision threshold tuning.

---

## Repository Structure

```text
├── code/                     # Core codebase modules and scripts
├── dataset/                  # Dataset directory (place raw CSV files here)
├── grading_tests/            # Official grading tests & environment configuration
├── notebooks/                # Exploratory and experimental Jupyter notebooks
│   ├── 01_data_exploration_and_cleaning.ipynb      # Missing values, survey cleaning rules, and class imbalance analysis
│   ├── 02_logistic_regression_experiment.ipynb     # Initial baseline logistic regression experiments
│   ├── 03_mean_squared_error_gd.ipynb              # Mean Squared Error with Gradient Descent implementation
│   ├── 04_regularized_logistic_regression.ipynb    # Regularized logistic regression and tuning on a stratified split
│   └── 05_feature_engineering_and_final_models.ipynb # Final pipeline: features, Ridge, Least Squares, and Newton's method
├── updates/                  # Project updates and supplementary materials
├── .gitignore                # Git ignore configuration
├── implementations.py        # The 6 required core ML implementations (at root for grader)
├── project1_description.pdf  # Official project guidelines
├── Greganova_Zirar_Bitar_Project1_Report.pdf        # Team report for Project 1
├── README.md                 # Project documentation
├── run.py                    # Main script reproducing the best submission
└── test.sh                   # Automated testing script
