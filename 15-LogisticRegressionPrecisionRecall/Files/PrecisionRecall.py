import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
import matplotlib.pyplot as plt

loans = pd.read_csv('LoansData01.csv')

target = 'safe_loans'
features = loans.columns[loans.columns!=target]

x = loans[features]
y = loans[target]

simple_logistic = linear_model.LogisticRegression(C=10e10)
simple_logistic.fit(x,y);
simple_eval = model_selection.cross_val_score(simple_logistic, x, y, cv=10)
print("Simple Logistic Regression\t%3.2f\t%3.2f" % (np.average(simple_eval), np.std(simple_eval)))

logistic_l1 = linear_model.LogisticRegression(penalty='l1')
logistic_l1.fit(x,y)
eval_l1 = model_selection.cross_val_score(logistic_l1, x, y, cv=10)
print("Logistic Regression (L1)  \t%3.2f\t%3.2f" % (np.average(eval_l1), np.std(eval_l1)))

logistic_l2 = linear_model.LogisticRegression(penalty='l2')
logistic_l2.fit(x,y)
eval_l2 = model_selection.cross_val_score(logistic_l2, x, y, cv=10)
print("Logistic Regression (L2)  \t%3.2f\t%3.2f" % (np.average(eval_l2), np.std(eval_l2)))


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

def PrintConfusionMatrix(model, true_y, predicted_y, positive=1, negative=-1):
    # true_positives = sum(predicted_y==1 and true_y==1)
    cm = confusion_matrix(y,yp)
    print("\t"+str(model.classes_[0])+"\t\t"+str(model.classes_[1]))
    print(str(model.classes_[0]) + "\t",cm[0][0],"\t",cm[0][1])
    print(str(model.classes_[1]) + "\t",cm[1][0],"\t",cm[1][1])

yp = simple_logistic.predict(x);
yp_l1 = logistic_l1.predict(x);
yp_l2 = logistic_l2.predict(x);

PrintConfusionMatrix(simple_logistic, y, yp)

precision_score(y, yp)

print(confusion_matrix(y,yp))

###
### AUC Precision-Recall Curve
###

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y, y)

###
### ROC Curve
###


