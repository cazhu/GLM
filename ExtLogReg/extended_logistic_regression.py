import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.feature_selection import chi2


# Load dataset
df = pd.read_csv('heights_weights_genders.csv')
df['Gender'] = 1 * df['Gender'] == 'Male'
x_train, x_test, y_train, y_test = train_test_split(df[['Height', 'Weight']], df['Gender'], test_size=0.25, random_state=0)

# variable selection
chi2_stats, p_val = chi2(x_train, y_train)
print('chi2 statistics of each feature: \n', chi2_stats)
print('p-values of each feature: \n', p_val)

# Run model
clf = linear_model.LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# p-value
"""
https://stackoverflow.com/questions/25122999/scikit-learn-how-to-check-coefficients-significance
"""

p = clf.predict_proba(x_train)
n = len(p)
m = len(clf.coef_[0]) + 1
coefs = np.concatenate([clf.intercept_, clf.coef_[0]])
x_full = np.matrix(np.insert(np.array(x_train), 0, 1, axis = 1))
ans = np.zeros((m, m))
for i in range(n):
    ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
print('ans: \n', ans)
vcov = np.linalg.inv(np.matrix(ans))
print('vcov: \n', vcov)
se = np.sqrt(np.diag(vcov))
t =  coefs/se
p = (1 - norm.cdf(abs(t))) * 2
print('p-value: \n', p)



# statsmodel
import statsmodels.api as sm
model = sm.Logit(y_train, sm.add_constant(x_train)).fit(disp=0)
print(model.summary())
y_pred_prob = model.predict(sm.add_constant(x_test))

# Confusion matrix
cm = confusion_matrix(y_test, [1 if x >= 0.5 else 0 for x in y_pred_prob])
print('Confusion matrix: \n', cm)

# ROC AUC
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5, 5))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_curve.png')

# Accuracy
accuracy = clf.score(x_test, y_test)
print('Accuracy:', accuracy)

# ROC AUC
y_pred_prob = clf.predict_proba(x_test)
y_pred_prob = [x[1] for x in y_pred_prob]
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5, 5))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_curve.png')
