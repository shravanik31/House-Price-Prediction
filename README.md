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

1. **Dataset:**
    - The dataset contains information about house prices in Melbourne, Australia, including features such as suburb, address, number of rooms, price, method of sale, type of property, and more.

2. **Data Analysis:**
    - Load and explore the dataset.
    - Handle missing values through imputation or by dropping columns with missing values.

3. **Data Cleaning:**
    - Detect and correct or remove corrupt or inaccurate records from the dataset.

4. **Data Visualization:**
    - Use Seaborn and Matplotlib for visualizing relationships between variables.

5. **Feature Selection:**
    - Select relevant features that contribute the most to the prediction variable to improve model performance.

## Model Building

- Split the data into training and testing sets.
- Use Linear Regression, Decision Tree Regressor, and Gradient Boosting Regressor to build prediction models.

## Model Evaluation

- Evaluate each model using metrics such as R2 score and Mean Absolute Error (MAE).
- Fine-tune hyperparameters to improve model accuracy.

## Implementation

1. **Data Exploration:**
    - Load and explore the dataset.
    - Print a summary of the dataset.

2. **Dealing with Missing Values:**
    - Handle missing values through imputation or by dropping columns.

3. **Data Cleaning:**
    - Clean the data by detecting and correcting or removing corrupt records.

4. **Data Visualization:**
    - Visualize data using Seaborn and Matplotlib.

5. **Feature Selection:**
    - Select relevant features for model building.

6. **Model Building:**
    - Build regression models using Linear Regression, Decision Tree Regressor, and Gradient Boosting Regressor.

7. **Model Evaluation:**
    - Evaluate models using R2 score and Mean Absolute Error (MAE).

8. **Improving Accuracy:**
    - Fine-tune hyperparameters and use techniques like GridSearchCV to improve model accuracy.

## Conclusion and Future Work

The project successfully built and evaluated various models for predicting house prices in Melbourne. Future work includes incorporating additional factors such as economic indicators and inflation rates to improve prediction accuracy.

## References

- [Understanding Gradient Boosting Machines](https://towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab)
- [Kaggle](https://www.kaggle.com/)
- [Linear Regression](https://towardsdatascience.com/tagged/linear-regression)
- [Gradient Boosting Regressor - Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [Numpy Quickstart](https://docs.scipy.org/doc/numpy/user/quickstart.html)
- [Scikit-learn Tutorial](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
