# House Price Prediction

## Overview

The relationship between house prices and the economy is an important motivating factor for predicting house prices. For accurate house price prediction, factors such as location, house type, size, build year, local amenities, and other relevant factors are considered. This project aims to create a house price prediction model using regression techniques to help both buyers and sellers make informed decisions.

## Objective

This project aims to develop a house price prediction model using regression to obtain optimal prediction results. The model will help prospective buyers plan their finances better and assist property investors in understanding housing price trends in specific locations.

## Models Used

1. **Linear Regression:**
    - Models the relationship between two variables by fitting a linear equation to observed data.

2. **Decision Tree Regression:**
    - Builds regression models in the form of a tree structure, breaking down datasets into smaller subsets incrementally.

3. **Gradient Boost Regression:**
    - An ensemble technique that produces a prediction model by combining multiple weak prediction models, typically decision trees.

## Packages

- **Pandas:** For data manipulation and analysis.
- **Scikit-learn:** Provides machine learning algorithms and tools.
- **Numpy:** For numerical operations on arrays.
- **Seaborn:** For statistical data visualization.
- **Matplotlib:** For 2D plotting.

## Technologies Used

- **Anaconda:** For managing packages and environments.
- **Jupyter Notebook:** For creating and sharing documents with live code, equations, visualizations, and narrative text.

## Data Preparation

- **Dataset:** Contains information about house prices in Melbourne, Australia, including features such as suburb, address, number of rooms, price, method of sale, type of property, and more.
- **Data Analysis and Cleaning:** Involves handling missing values, detecting and correcting or removing corrupt records, and visualizing relationships between variables.
- **Feature Selection:** Selecting relevant features that contribute the most to the prediction variable to improve model performance.

## Model Building and Evaluation

- Models are built using Linear Regression, Decision Tree Regressor, and Gradient Boosting Regressor.
- Evaluation metrics include R2 score and Mean Absolute Error (MAE).
- Hyperparameters are fine-tuned to improve model accuracy.

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/shravanik31/house-price-prediction.git
    cd house-price-prediction
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset:**

    Download the dataset from [Kaggle](https://www.kaggle.com/code/alexisbcook/xgboost/data) and place it in your directory.

5. **Run the Jupyter Notebook:**

    ```bash
    jupyter notebook HousePricePrediction.ipynb
    ```

6. **Follow the steps in the notebook to load the data, preprocess it, build models, and evaluate them.**

## Conclusion and Future Work

The project successfully built and evaluated various models for predicting house prices in Melbourne. Future work includes incorporating additional factors such as economic indicators and inflation rates to improve prediction accuracy.

## References

- [Understanding Gradient Boosting Machines](https://towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab)
- [Linear Regression](https://towardsdatascience.com/tagged/linear-regression)
- [Gradient Boosting Regressor - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
