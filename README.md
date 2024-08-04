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
## [DataAnalysingProject](https://github.com/Firdavs222/Portfolio-projects/blob/main/DataAnalysingProject.py) - this project is aimed to analyze social media users'usage, like Twitter, YouTube, Facebook, etc. But, all data was generated randomly and not a real one. This project includes 5  steps:  
1. Generating random data;
2. Exploring data;
3. Cleaning data;
4. Visualising data;
5. Statistical interpreting;  
   At the first step, I randomly generate numerical and datetime data with 'NumPy' and 'Random' modules. Categories of generated dataframe are: "Food", "Travel", "Fashion", "Music", "Culture", "Family", "Health". At the second step, data was explored: dataframe includes 3 columns("Date", "Category", "Likes") and 500 rows; In this step, dataframe's head 20 values, brief information and description of it was shown. At the third step, data was cleaned: was dropped null values and duplicate values, converted date format to 'datetime' and 'Likes' to integer. At the fourth step, data was shown at the hystogram and boxplot. At the final step, got mean of 'Likes' and mean of 'Likes' of groupped by 'Category'. At the above, was shown the result:
   <div style="display: flex; justify-content: space-between;">
   <img src="./Screenshot%202024-05-14%20152012.png" alt="Project Overview" width="auto" height="300"/>
   <img src="./Screenshot%202024-05-14%20152135.png" alt="Project Overview" width="auto" height="300"/>
   </div>
## [DatabaseOfMuscleCarsMarket](https://github.com/Firdavs222/Portfolio-projects/blob/main/carsmarket%20database.sql) 
This project is intended to create database which is called MuscleCarsMarket and fill it with data of cars. Then some queries are written to get results.  
First of all MuscleCarsMarket database was created and then in it the table "cars" created with 4 features: "Make", "Model", "Year", Unique_ID" and table "Employees" with features "id", "first_name", "last_name", "email", "gender", "ip_address". And then, some data inserted into table which was randomly generated with https://www.mockaroo.com/ . And, finally some queries were entered to take some intended information.



   
