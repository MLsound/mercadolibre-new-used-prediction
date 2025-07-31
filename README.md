# Opportunities@MeLi - Code Exercise - Data Scientist
## Mercado Libre New/Used Item Prediction

This repository contains a machine learning solution to predict whether an item listed on Mercado Libre's Marketplace is **new** or **used**. The goal is to achieve a minimum accuracy of **0.86** on held-out test data.

---

## 1. Project Overview

Mercado Libre, a prominent e-commerce platform in Latin America, requires an algorithm to accurately classify item conditions. This project addresses that need by developing a machine learning model to distinguish between "new" and "used" items based on provided data.

---

## 2. Dataset

The dataset used for this project is `MLA_100k_checked_v3.jsonlines`. A helper function `build_dataset` in `new_or_used.py` is provided to facilitate reading and processing this data.

---

## 3. Deliverables

### 3.1. Code

The core of the solution lies in the scripts and/or runnable notebooks that define, train, and evaluate the machine learning model. These files are designed for reproducibility and legibility, adhering to collaborative development best practices.

* **`Model_Design.ipynb`**: This notebook serves as the primary location for model definition, training, and evaluation on the test set. It encapsulates the iterative process of model development.
* **`data_processor.py`**: This script is responsible for data loading, preprocessing, and potentially feature engineering, preparing the data for model consumption.
* **`new_or_used.py`**: Contains the `build_dataset` function, a helper function for reading and initially processing the raw data.

---

### 3.2. Documentation

A comprehensive document explaining the methodology, choices, and results.

* **`documentation_report.pdf`**:
    * **Feature Selection Criteria**: Detailed explanation of the features chosen for the model and the rationale behind their selection.
    * **Secondary Metric**:
        * Identification of an appropriate secondary evaluation metric.
        * Justification for choosing this metric, highlighting its relevance to the problem.
        * Performance achieved on this secondary metric.
* **`EDA.ipynb`**: An exploratory data analysis (EDA) notebook to further illustrate data insights, including data understanding, distribution analysis, and initial feature insights.

---

## 4. Getting Started

### 4.1. Prerequisites

To run this project, ensure you have the following installed:

* Python 3.x
* `pip` (Python package installer)

---

### 4.2. Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/MLsound/mercadolibre-new-used-prediction.git](https://github.com/MLsound/mercadolibre-new-used-prediction.git)
    cd mercadolibre-new-used-prediction
    ```

2.  **Install dependencies:**

    It is highly recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    
---

## 5. Project Structure
```
.
├── .venv/                         # Python virtual environment
├── data/                          # Raw and processed datasets
│   └── MLA_100k_checked_v3.jsonlines
├── models/                        # Trained machine learning models
├── scripts/                       # Python scripts for data processing, modeling, and evaluation
│   └── data_processor.py          # Script for loading data and preparing it for feed the model
├── .gitignore                     # Specifies intentionally untracked files to ignore
├── EDA.ipynb                      # Exploratory Data Analysis notebook
├── LICENSE                        # Project license file
├── Model_Design.ipynb             # Notebook for model design and experimentation
├── new_or_used.py                 # Contains the build_dataset function
├── Opportunities@MeLi - CodeExercise DS_ML.docx # Project documentation in Word format
├── pairplot.png                   # Image of pairplot
├── README.md                      # This file
├── documentation_report.pdf       # Documentation
└── requirements.txt               # Lists all Python dependencies
```
---

## 6. License
This project is licensed under the MIT License - see the LICENSE file for details.