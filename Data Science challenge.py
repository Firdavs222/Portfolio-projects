# -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:40:11 2024

@author: user
"""


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import roc_auc_score, roc_curve


train_df = pd.read_csv("D:/Downloads/Files/Files/train.csv")
test_df = pd.read_csv("D:/Downloads/Files/Files/test.csv")

X_train = train_df[["MonthlyCharges", "TotalCharges","ViewingHoursPerWeek",
                    "AverageViewingDuration","ContentDownloadsPerMonth","UserRating",
                    "SupportTicketsPerMonth","WatchlistSize" 
                    ]]
Y_train = train_df["Churn"]

model = Sequential([
    Dense(units = 15, activation = 'sigmoid'),
    Dense(units = 10, activation = 'sigmoid'),
    Dense(units = 1, activation = 'linear')
    ])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = (1e-3)/2), 
              loss = BinaryCrossentropy(from_logits = True))

model.fit(X_train, Y_train, epochs = 20)

logit = model(X_train)

f_x = tf.nn.sigmoid(logit)

X_test = test_df[["MonthlyCharges", "TotalCharges","ViewingHoursPerWeek",
                    "AverageViewingDuration","ContentDownloadsPerMonth","UserRating",
                    "SupportTicketsPerMonth","WatchlistSize"
                    ]]

logit2 = model(X_test)

f2_x = tf.nn.sigmoid(logit2)

Y_test = []
for P in f2_x:
    if P >= 0.5:
        P = 1
        Y_test.append(P)
    else:
        P = 0
        Y_test.append(P)

dict_df = {"CustomerID": test_df["CustomerID"],
        "predicted_probability": f2_x.tolist()
        }

new_df = pd.DataFrame(dict_df)

print(f"Model's AUC score: {roc_auc_score(Y_train.tolist(), f_x.tolist())}")


def plot_roc_curve(Y_train, f_x):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(Y_train, f_x)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(Y_train, f_x)





