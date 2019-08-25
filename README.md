# EY_Datascience_Challenge_2019
Code used to compete in the 2019 EY Next Wave datascience challenge @ https://datascience.ey.com/challenges.

Data required significant feature engineering, including computation of journey times, journey trajectories, distance
from target locations and home coordinates.

Regression and classification algorithms were both applied to the problem. Regression required training an algorithm to predict a
set of GPS coordinates from a featureset. These could then be used to determine whether the prediction was within or outside the 
target zone. The classification algorithms explicitly classified the features within the zone or not.

XGBoost, Random Forest, Elastic Net were some of the regressors applied using SK-learn's multi-output regression module.

The primary classifier looked into was a deep neural network.

Classifiers performed better than regression for this task.
