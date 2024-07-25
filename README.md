  There are my 3 portfolio projects: "Data Science challenge", "DataAnalysingProject", "carsmarket database". 
  ---
  ## [Data Science challenge - Coursera](https://github.com/Firdavs222/Portfolio-projects/blob/main/Data%20Science%20challenge.py) project is a challenge which was held in Coursera.  
 **Project description**: Challenge's task is to train the "train.csv" data with any model; e.g. Neural networks, Decision Trees, XGBoost, ...;  and predict the target "Churn" value for "test.csv".  I've chosen XGBoost and neural networks to train the model. But XGBoost's performance was better and faster than neural networks and I've chosen XGBoost to train and predict test data. My work above this project includes loading, exploring, feature engineering, preprocessing, splitting data into train and validation data, training, feature selection, retraining with selected features, predicting target values with selected features of test data.  
  - First of all, I loaded the data into my workspace; and engineered 2 new features for both of train and test datasets: 'AvgViewingDuration_WatchlistSize' = ['AverageViewingDuration'] * ['WatchlistSize'];   'Total_viewing' = ['AverageViewingDuration'] * ['ContentDownloadsPerMonth'];
  - After feature engineering data was splitted into train and cross validation sets;
  - Then I used sklearn's "compose.ColumnTransformer" modul to preprocess categorical data with 'OneHotEncoder' and numerical data with 'StandardScaler';
  - After preprocessing, train data was trained using XGBoost model; and after data was trained got important features, which affects to target value;
  - Then data was re-preprocessed and retrained with important features;
  - After training test data's target values were predicted using the model which was trained before;
  - At the end, new dataframe was created and predicted probabilities and Customer's ID were uploaded into it.  
    Model's performance was 74,88, which was measured by 'AUC-ROC' score.  
## [DataAnalysingProject](https://github.com/Firdavs222/Portfolio-projects/blob/main/DataAnalysingProject.py)
