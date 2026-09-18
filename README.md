# EPFL Machine Learning - Project 1: Heart Attack Prediction

## Team Members
* Mohamed ZIRAR
* Ema GREGANOVA 
* Cristophe BITAR

## Overview
This repository contains our code and report for Project 1 of the EPFL Machine Learning course (CS-433) Fall 2026. The objective of this project is to predict the risk of cardiovascular disease (MICHD) using health-related survey data from the BRFSS dataset.

## Repository Structure
* `implementations.py`: linear and logistic regression implementations.
* `run.py`: entry point for running the project experiments.
* `project1_description copie.pdf`: project brief and requirements.

## Requirements & Constraints
* **Python Version:** Python 3.x
* **Allowed Libraries:** Python standard libraries, `NumPy`, and visualization libraries (`matplotlib`, `seaborn`) used strictly for plotting. **No external ML libraries (e.g., Scikit-Learn, Pandas) are allowed.**

## How to Run
Create the project environment from `grading_tests copie/environment.yml`, then run:

```bash
python run.py
```

To run the public checks locally:

```bash
pytest "grading_tests copie" --github_link .
```
