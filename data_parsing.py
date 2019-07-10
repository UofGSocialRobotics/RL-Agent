import itertools
import os
import csv

from sklearn.ensemble import RandomForestClassifier

import config
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def build_raw_dataset():
    rapport_dict, groups_dict = load_rapport_and_groups_dict()
    with open(config.TRAINING_DIALOGUE_PATH + "raw_dataset.csv", mode='w', newline='') as csv_file:
        dataset = csv.writer(csv_file, delimiter=',')
        tmp_row = []
        for root, dirs, files in os.walk(config.TRAINING_DIALOGUE_PATH):
            for name in files:
                turn = 1
                if name.endswith(".csv") and "dataset" not in name and "rapport" not in name:
                    with open(config.TRAINING_DIALOGUE_PATH + name, mode='rt') as csv_file:
                        interaction = csv.reader(csv_file, delimiter=',')
                        for row in interaction:
                            if turn == 1:
                                tmp_row.append(row[1].replace("_full_dialog.pkl", ""))
                                tmp_row.append(rapport_dict[row[1].replace("_full_dialog.pkl", "")])
                                tmp_row.append(groups_dict[row[1].replace("_full_dialog.pkl", "")])
                                tmp_row.extend(vectorize(row[9]))
                            else:
                                tmp_row.extend(vectorize(row[4]))
                                tmp_row.extend(vectorize(row[6]))
                                tmp_row.extend(vectorize(row[9]))
                            turn += 1
                    while turn < 16:
                        tmp_row.extend(vectorize("blank"))
                        turn += 1
                    dataset.writerow(tmp_row)
                    tmp_row = []


def build_count_dataset():
    rapport_dict, groups_dict = load_rapport_and_groups_dict()
    with open(config.TRAINING_DIALOGUE_PATH + "count_dataset.csv", mode='w', newline='') as csv_file:
        dataset = csv.writer(csv_file, delimiter=',')
        tmp_row = []
        ack_cs = [0, 0, 0, 0, 0]
        agent_cs = [0, 0, 0, 0, 0]
        user_cs = [0, 0, 0, 0, 0]
        for root, dirs, files in os.walk(config.TRAINING_DIALOGUE_PATH):
            for name in files:
                turn = 1
                if name.endswith(".csv") and "dataset" not in name and "rapport" not in name:
                    with open(config.TRAINING_DIALOGUE_PATH + name, mode='rt') as csv_file:
                        interaction = csv.reader(csv_file, delimiter=',')
                        for row in interaction:
                            if turn == 1:
                                tmp_row.append(row[1].replace("_full_dialog.pkl", ""))
                                tmp_row.append(rapport_dict[row[1].replace("_full_dialog.pkl", "")])
                                group = groups_dict[row[1].replace("_full_dialog.pkl", "")]
                                if group in '1' or group in '2' or group in '3':
                                    tmp_row.append(0)
                                else:
                                    tmp_row.append(1)
                            ack_cs = count(row[4], ack_cs)
                            agent_cs = count(row[6], agent_cs)
                            user_cs = count(row[9], user_cs)
                            turn += 1
                    tmp_row.extend(ack_cs)
                    tmp_row.extend(user_cs)
                    tmp_row.extend(agent_cs)
                    dataset.writerow(tmp_row)
                    tmp_row = []
                    ack_cs = [0, 0, 0, 0, 0]
                    agent_cs = [0, 0, 0, 0, 0]
                    user_cs = [0, 0, 0, 0, 0]


def load_rapport_and_groups_dict():
    rapport_dict = {}
    groups_dict = {}
    with open(config.TRAINING_DIALOGUE_PATH + "rapport_groups.csv", mode='rt') as csv_file:
        interaction = csv.reader(csv_file, delimiter=',')
        for row in interaction:
            rapport_dict[row[0]] = row[2]
            groups_dict[row[0]] = row[1]
        return rapport_dict, groups_dict


def vectorize(cs):
    if "NONE" in cs:
        vector = [1]
    elif "HE" in cs:
        vector = [2]
    elif "SD" in cs:
        vector = [3]
    elif "PR" in cs:
        vector = [4]
    elif "VSN" in cs:
        vector = [5]
    elif "blank" in cs:
        vector = [0, 0, 0]
    else:
        vector = [0]
    return vector


def one_hot_vectorize(cs):
    if "NONE" in cs:
        vector = [1, 0, 0, 0, 0]
    elif "HE" in cs:
        vector = [0, 1, 0, 0, 0]
    elif "SD" in cs:
        vector = [0, 0, 1, 0, 0]
    elif "PR" in cs:
        vector = [0, 0, 0, 1, 0]
    elif "VSN" in cs:
        vector = [0, 0, 0, 0, 1]
    elif "blank" in cs:
        vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        vector = [0, 0, 0, 0, 0]
    return vector


def count(cs, list):
    if "NONE" in cs:
        list[0] += 1
    elif "HE" in cs:
        list[1] += 1
    elif "SD" in cs:
        list[2] += 1
    elif "PR" in cs:
        list[3] += 1
    elif "VSN" in cs:
        list[4] += 1
    return list


def get_data(dataset_name):
    dataset = pd.read_csv(config.TRAINING_DIALOGUE_PATH + dataset_name, index_col=False, header=None)
    X = dataset.drop([0, 1, 2], axis=1)
    y = dataset[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #std_scale = preprocessing.StandardScaler().fit(X)
    #df_std = std_scale.transform(X)
    return X_train, X_test, y_train, y_test

def compute_reg_scores(regressors_list, X_train, X_test, y_train, y_test):
    for name, reg in regressors_list.items():
        if name == 'Lin' or name == 'Las' or name == 'Rid' or name in 'MLP':
            grid_values = {}
        #elif name in 'MLP':
        #    grid_values = [{'learning_rate': ["constant", "invscaling", "adaptive"],
                            #            'max_iter': [200, 500, 800, 1000, 1500],
                            #            'hidden_layer_sizes': [(10), (10,2), (5,), (5,2), (9,), (9,2)],
        #                    'alpha': [0.0001, 1e-5, 0.01, 0.001],
        #                    'solver': ['lbfgs', 'sgd', 'adam'],
        #                    'learning_rate_init': [0.001, 0.01, 0.1, 0.2, 0.3],
        #                    'activation': ["logistic", "relu", "tanh"]}]

        results = GridSearchCV(reg, param_grid=grid_values, cv=3, iid=False, scoring='neg_mean_squared_error')
        results.fit(X, y)
        # if name in 'MLP':
        # print(name + ": " + str(results.best_estimator_.hidden_layer_sizes))
        print(name + ": " + str(results.best_score_))

def compute_clf_scores(classifiers_list, X_train, X_test, y_train, y_test):
    final_scores = {}
    for name, clf in classifiers_list.items():
        if name == 'Log':
            grid_values = {'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000]}
        elif name in 'SVM':
            grid_values = [
                {'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000], 'kernel': ['linear']},
                {'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
            ]
        elif name in 'KNN':
            grid_values = {"n_neighbors": [2, 3, 4], "metric": ["euclidean", "cityblock"]}
        elif name in 'DT':
            grid_values = {'criterion': ['gini', 'entropy'],
                           'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]}
        elif name in 'Gau':
            grid_values = {}
        elif name in 'RFC':
            grid_values = {'n_estimators': [200, 300, 400, 500, 600, 700], 'max_features': ['auto', 'sqrt', 'log2']}
        elif name in 'MLP':
            grid_values = [{'learning_rate': ["constant", "invscaling", "adaptive"],
                            #            'max_iter': [200, 500, 800, 1000, 1500],
                            #            'hidden_layer_sizes': [(10), (10,2), (5,), (5,2), (9,), (9,2)],
                            'alpha': [0.0001, 1e-5, 0.01, 0.001],
                            'solver': ['lbfgs', 'sgd', 'adam'],
                            'learning_rate_init': [0.001, 0.01, 0.1, 0.2, 0.3],
                            'activation': ["logistic", "relu", "tanh"]}]

        results = GridSearchCV(clf, param_grid=grid_values, cv=10, iid=False, scoring='f1_weighted')
        results.fit(X_train, y_train)
        # if name in 'MLP':
        # print(name + ": " + str(results.best_estimator_.hidden_layer_sizes))


        print(name + ": " + str(results.best_score_))
        y_pred = results.best_estimator_.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
        final_scores[name] = [str(results.best_score_), confusion_matrix(y_test, y_pred), precision, recall, fscore, support]

        # print('Best C:', results.best_estimator_.C)
    for key, score in final_scores.items():
        print(key + " - F1 score train: " + str(score[0]))
        print(key + " - F1 score test: " + str(sum(score[4])/len(score[4])))
        print(key + " - confusion matrix:\n " + str(score[1]))

def get_reg_models():
    Lin = LinearRegression()
    Rid = Ridge()
    Las = Lasso()
    MLP = MLPRegressor(hidden_layer_sizes=10, max_iter=100)

    reg_dict = {'Lin': Lin, 'Rid': Rid, 'Las': Las, 'MLP': MLP}
    return reg_dict

def get_clf_models():
    MLP = MLPClassifier(hidden_layer_sizes=10, max_iter=100)
    Log = LogisticRegression(solver='liblinear', multi_class='ovr')
    SVM = SVC(gamma='scale')
    KNN = KNeighborsClassifier()
    Gau = GaussianNB()
    DT = DecisionTreeClassifier()
    RFC = RandomForestClassifier(n_estimators=100)

    clf_dict = {'MLP': MLP, 'Log': Log, 'SVM': SVM, 'KNN': KNN, 'Gau': Gau, 'DT': DT, 'RFC': RFC}
    return clf_dict


def cross_validate():
    dataset_name = "count_dataset.csv"
    X_train, X_test, y_train, y_test = get_data(dataset_name)

    clf_models = get_clf_models()
    reg_models = get_reg_models()

    compute_clf_scores(clf_models, X_train, X_test, y_train, y_test)
    #compute_reg_scores(reg_models, X_train, X_test, y_train, y_test)




if __name__ == '__main__':
    build_count_dataset()
    cross_validate()
