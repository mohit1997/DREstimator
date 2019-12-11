import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import json

param_file = 'vowel.params'
filepath = 'vowel.data'

np.random.seed(0)

def parse_data(param_file, filepath):
  with open(filepath) as fp:
    line = fp.readline()
    X = []
    Y = []
    cnt = 1
    sep_dic = {'ecoli': None, 'yeast': None, 'glass': ',', 'letter-recognition': ',', 'sonar': ',', 'vowel': None}
    kind = filepath.split('.')[0]
    if kind != "letter-recognition":
        while line:
            x = line.split(sep_dic[kind])[1:-1]
            x = [float(i) for i in x]
            y = line.split(sep_dic[kind])[-1]
            X.append(x)
            Y.append(y)
            #  print("Ex {}: {} -> {}".format(cnt, x, y))
            line = fp.readline()
            cnt += 1
    else:        
        while line:
            x = line.split(sep_dic[kind])[1:]
            x = [float(i) for i in x]
            y = line.split(sep_dic[kind])[0]
            X.append(x)
            Y.append(y)
            #  print("Ex {}: {} -> {}".format(cnt, x, y))
            line = fp.readline()
            cnt += 1

  X = np.array(X)
  vals = list(set(Y))
  vals.sort()
  print("Classes :{}".format(vals))

  char2id_dict = {c: i for (i,c) in enumerate(vals)}
  id2char_dict = {i: c for (i,c) in enumerate(vals)}

  params = {'char2id_dict':char2id_dict, 'id2char_dict':id2char_dict}
  with open(param_file, 'w') as f:
      json.dump(params, f, indent=4)

  labels = np.array([char2id_dict[c] for c in Y])
  OnehotLabels = np.zeros((len(labels), len(vals)))
  OnehotLabels[np.arange(len(labels)), labels] = 1.0

  return X, labels, OnehotLabels, len(vals)

def get_pratially_revealed_labels(num_classes, true_labels):
  poLabels = -np.ones((len(true_labels), num_classes))
  random_indices = np.random.choice(num_classes, len(true_labels))
  poLabels[np.arange(len(true_labels)), random_indices] = true_labels[np.arange(len(true_labels)), random_indices]
  return poLabels

class MyPredictor:
  def __init__(self, class_num, loss):
    self.class_num = class_num
    self.loss = loss

  def predict_proba(self, features):
    return np.stack((1 - self.loss * np.ones((len(features))), self.loss * np.ones((len(features)))), axis=-1)

def prob2loss(probs):
  return probs[:, -1]

def loss_estimator(features, labels, num_classes):
  # Create Dictionary of loss estimators
  estimators = {}
  for i in range(num_classes):
    loss_i = 1 - labels[:, i]
    unique_vals = np.unique(loss_i)
    if len(unique_vals) > 1:
      clf = LogisticRegression(C=5.0, solver='liblinear', penalty='l1', max_iter=1000, tol=1e-4)
      estimators[i] = clf
      clf.fit(features, loss_i)
      # print(clf.score(features, loss_i))
    else:
      clf = MyPredictor(class_num=i, loss=loss_i)
      estimators[i] = clf
  return estimators


X, Y, OhotY, num_classes = parse_data(param_file, filepath)

def partial_loss_estimator(features, labels, num_classes):
  # Create Dictionary of loss estimators
  polabels = get_pratially_revealed_labels(num_classes, labels)
  estimators = {}
  for i in range(num_classes):
    idx = labels[:, i] != -1
    loss_i = 1 - labels[:, i][idx].copy()
    features_i = features[idx].copy()
    unique_vals = np.unique(loss_i)
    if len(unique_vals) > 1:
      clf = LogisticRegression(C=1.0, solver='lbfgs', penalty='l2', max_iter=1000, tol=1e-4)
      estimators[i] = clf
      clf.fit(features, loss_i)
      # print(clf.score(features, loss_i))
    else:
      clf = MyPredictor(class_num=i, loss=unique_vals[0])
      estimators[i] = clf
  return estimators, polabels

def DM_impute(features, estimators, num_classes):
  loss = np.zeros((len(features), num_classes))
  for j in range(num_classes):
    loss[:, j:j+1] = np.array([prob2loss(estimators[j].predict_proba(features[i:i+1])) for i in range(len(features))])

  return 1 - loss
  # for i,a in enumerate(actions):
  #   loss.append(estimator[a].predict(features[i:i+1]))

def IPS_impute(features, polabels, num_classes):
  loss = np.zeros((len(features), num_classes))
  for j in range(num_classes):
    loss[:, j] = np.array([num_classes*(1 - polabels[i, j]) if polabels[i, j] != -1 else 0 for i in range(len(features))])
  return 1 - loss

def DR_impute(features, polabels, num_classes, estimators):
  loss = np.zeros((len(features), num_classes))
  for j in range(num_classes):
    loss[:, j:j+1] = np.array([num_classes*(1 - polabels[i, j] - prob2loss(estimators[j].predict_proba(features[i:i+1]))) + prob2loss(estimators[j].predict_proba(features[i:i+1])) 
    if polabels[i, j] != -1 
    else prob2loss(estimators[j].predict_proba(features[i:i+1]))
    for i in range(len(features))])
  return 1 - loss

def rmse(predictions, targets):
  return np.sqrt(((predictions - targets) ** 2).mean())

reps = 50
losses = {'DM': [], 'IPS': [], 'DR': [], 'True': []}
for time in range(reps):
  X_train, X_test, y_train, y_test, oh_train, oh_test = train_test_split(
      X, Y, OhotY, test_size=0.30)
  clf = SGDClassifier(max_iter=1000, tol=1e-3)
  estimators, polabels = partial_loss_estimator(X_train, oh_train, num_classes)
  y_train_imputed = DM_impute(X_train, estimators, num_classes)
  clf.fit(X_train, np.argmax(y_train_imputed, axis=1))
  losses['DM'].append(1 - clf.score(X_test, y_test))
  y_train_imputed = IPS_impute(X_train, polabels, num_classes)
  clf.fit(X_train, np.argmax(y_train_imputed, axis=1))
  losses['IPS'].append(1 - clf.score(X_test, y_test))
  y_train_imputed = DR_impute(X_train, polabels, num_classes, estimators)
  clf.fit(X_train, np.argmax(y_train_imputed, axis=1))
  losses['DR'].append(1 - clf.score(X_test, y_test))
  # clf.fit(X_train, y_train)
  # losses['True'].append(1 - clf.score(X_test, y_test))
  print("Time {}/{}".format(time+1, reps), end='\r')



def bias_var(pred, label):
  return np.mean(pred - label), np.var(pred - label)


dm_out = np.array(losses['DM'])
ips_out = np.array(losses['IPS'])
dr_out = np.array(losses['DR'])
# target = np.array(losses['True'])

print("DM: Error {:4f}".format(np.mean(dm_out)))
print("IPS: Error {:4f}".format(np.mean(ips_out)))
print("DR: Error {:4f}".format(np.mean(dr_out)))

# print("DM: Bias {:.4f} Variance {:.4f}".format(*bias_var(dm_out, target)))
# print("IPS: Bias {:.4f} Variance {:.4f}".format(*bias_var(ips_out, target)))
# print("DR: Bias {:.4f} Variance {:.4f}".format(*bias_var(dr_out, target)))
