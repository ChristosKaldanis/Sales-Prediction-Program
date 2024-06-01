# Sales-Prediction-Program
# Info

# Project Description
The Sales-Prediction-Program project is aimed at predicting sales (e.g., of an item) based on a .csv file, which must contain two columns with values for views and rating stars, using the Pandas, Numpy, Sklearn, and Matplotlib libraries.

# Methodology
First, after loading the dataset from the .csv file, we need to check if the values in the views column contain the ',' symbol and, if so, replace it with a blank space to avoid incorrect results or value losses. Next, we convert all our data to numerical values for verification purposes and round them to one decimal place. We examine if there are any missing data and, if so, remove them. We add a column to the dataset with random sales figures to compare them later with those predicted by the linear regression model (Predicted Sales). If there is already a column with actual sales in the dataset, we can skip the step of creating the random sales column. Additionally, we check if our dataset has sufficient values, and if it does, we train our model and evaluate it using the mean_squared_error and r2_score functions. Finally, we visualize the predictions and the distribution of the data.

# Implementation
The implementation of this project was carried out by Christos Kaldanis, an undergraduate student at the Department of Informatics, Ionian University.
