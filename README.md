# COVID-19 Analysis and Modeling

This repository contains an analysis of COVID-19 impact on hospital capacity by state, with a focus on identifying significant features contributing to COVID-19 deaths and finding an accurate, interpretable machine learning model.

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Results](#results)
- [Key Features](#key-features)

## Introduction

This project analyzes a dataset containing COVID-19 reported patient impact and hospital capacity by state. The analysis includes data preprocessing, visualization, and training multiple machine learning models to predict COVID-19 deaths.

## Objectives

The objectives of this analysis are to:
- Identify significant features contributing to COVID-19 deaths.
- Find an accurate and interpretable machine learning model.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/danielshort3/covid-analysis.git
    cd covid-analysis
    ```

2. Install the required packages:
    ```bash
    pip install pandas matplotlib seaborn numpy scikit-learn
    ```

3. Ensure the dataset `COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries__RAW_.csv` is in the `data/` directory.

## Usage

1. Open the Jupyter notebook `covid_analysis.ipynb` to run the analysis from start to finish.

## Preprocessing

- Import and format the dataset.
- Handle missing values by forward filling and removing unnecessary columns.
- Filter data to remove early pandemic dates with little information.

## Model Training

Three machine learning models are trained and evaluated:
- **Random Forest**
- **Decision Tree**
- **Linear Regression**

Each model is analyzed using a train-test split, and their performance is measured using Root Mean Squared Error (RMSE).

## Results

- The Decision Tree model showed the lowest RMSE, followed by Random Forest and Linear Regression.
- Key features identified include `staffed_icu_adult_patients_confirmed_covid`, `adult_icu_bed_covid_utilization_numerator`, and `total_adult_patients_hospitalized_confirmed_covid`.

## Key Features

Significant features contributing to the models' predictions were identified and analyzed to understand their impact on COVID-19 deaths.

For more details, refer to the Jupyter notebook `covid_analysis.ipynb` in this repository.
