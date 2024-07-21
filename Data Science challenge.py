# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:40:11 2024

@author: user
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X_train = train_df[["MonthlyCharges", "TotalCharges","ViewingHoursPerWeek",
                    "AverageViewingDuration","ContentDownloadsPerMonth","UserRating",
                    "SupportTicketsPerMonth","WatchlistSize" 
                    ]]
Y_train = train_df["Churn"]

model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)

model.fit(X_train, Y_train, early_stopping_rounds=10, eval_set=[(X_train, Y_train)], eval_metric='logloss', verbose=True)

X_test = test_df[["MonthlyCharges", "TotalCharges","ViewingHoursPerWeek",
                    "AverageViewingDuration","ContentDownloadsPerMonth","UserRating",
                    "SupportTicketsPerMonth","WatchlistSize"
                    ]]

xgb_pred_prob = model.predict_proba(X_test)[:, 1]
dict_df = {"CustomerID": test_df["CustomerID"],
        "predicted_probability": f2_x.tolist()
        }

prediction_df = pd.DataFrame(dict_df)
