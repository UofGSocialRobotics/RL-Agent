import os
import csv
import pickle
import ml_models

from keras.optimizers import Adam
from numpy import array

from scipy import stats
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import config
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense


def build_task_model():
    dataset = config.RECO_ACCEPTANCE_DATASET
    cross_validate(dataset)

def build_social_model():
    dataset = "reciprocity_dataset.csv"
    print(dataset)
    cross_validate(dataset)

def get_task_data(dataset_name):
    dataset = pd.read_csv(dataset_name, index_col=False, header=None)
    X = dataset.drop([4], axis=1)
    X = MinMaxScaler().fit_transform(X)
    y = dataset[4]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test

def cross_validate(dataset):
    X_train, X_test, y_train, y_test = get_data(dataset)
    #X_train, X_test, y_train, y_test = get_task_data(dataset)

    #clf_models = get_clf_models()
    reg_models = get_reg_models()

    #compute_clf_scores(clf_models, X_train, X_test, y_train, y_test)
    compute_reg_scores(reg_models, X_train, X_test, y_train, y_test)


#################################################################################################################
#################################################################################################################
##############                                                                                ###################
##############                  Off Line Rapport Estimator Training Functions                 ###################
##############                                                                                ###################
#################################################################################################################
#################################################################################################################

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

def build_reciprocity_dataset():
    rapport_dict, groups_dict = load_rapport_and_groups_dict()
    with open(config.TRAINING_DIALOGUE_PATH + "reciprocity_dataset.csv", mode='w', newline='') as csv_file:
        dataset = csv.writer(csv_file, delimiter=',')
        tmp_row = []
        rec_agent = [0, 0, 0, 0, 0]
        rec_user = [0, 0, 0, 0, 0]
        for root, dirs, files in os.walk(config.TRAINING_DIALOGUE_PATH):
            for name in files:
                turn = 1
                if name.endswith(".pkl.csv"):
                    with open(config.TRAINING_DIALOGUE_PATH + name, mode='rt') as csv_file:
                        interaction = csv.reader(csv_file, delimiter=',')
                        for row in interaction:
                            if turn == 1:
                                tmp_row.append(row[1].replace("_full_dialog.pkl", ""))
                                dialogue_name = row[1].replace("_full_dialog.pkl", "").split("\\")
                                tmp_row.append(rapport_dict[dialogue_name[1]])
                            else:
                                # Check which CS the agent used as an ack to reciprocate user non NONE CS
                                if "NONE" not in user_cs:
                                    rec_agent = increment_agent_rec(row[4], rec_agent)
                            user_cs = row[10]
                            # Check which CS from the agent was reciprocated by a non NONE CS from the user side
                            if "NONE" not in row[10]:
                                rec_user = count(row[7], rec_user)
                            turn += 1
                    tmp_row.extend(rec_user)
                    tmp_row.extend(rec_agent)
                    dataset.writerow(tmp_row)
                    tmp_row = []
                    rec_user = [0, 0, 0, 0, 0]
                    rec_agent = [0, 0, 0, 0, 0]


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

def increment_agent_rec(cs, list):
    if "NONE" in cs:
        list[1] += 1
    elif "SD" in cs:
        list[2] += 1
    elif "PR" in cs:
        list[3] += 1
    elif "VSN" in cs:
        list[4] += 1
    else:
        list[0] += 1
    return list

def count(agent_cs, list):
    if "NONE" in agent_cs:
        list[0] += 1
    elif "HE" in agent_cs:
        list[1] += 1
    elif "SD" in agent_cs:
        list[2] += 1
    elif "PR" in agent_cs:
        list[3] += 1
    elif "VSN" in agent_cs:
        list[4] += 1
    return list

def scale_data(data):
    data = array([data])
    data = MinMaxScaler().fit_transform(data.T)
    X = pd.DataFrame(data, columns = None)
    return X.T

def get_data(dataset_name):
    dataset = pd.read_csv(config.TRAINING_DIALOGUE_PATH + dataset_name, index_col=False, header=None)
    X = dataset.drop([0, 1, 2], axis=1)
    X = MinMaxScaler().fit_transform(X)
    y = dataset[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test

def compute_reg_scores(regressors_list, X_train, X_test, y_train, y_test):
    for name, reg in regressors_list.items():
        if name == 'Lin' or name == 'Las' or name == 'Rid' or name in 'MLP':
            grid_values = {}
        elif name in 'MLP':
            grid_values = [{'learning_rate': ["constant", "invscaling", "adaptive"],
                                        'max_iter': [200, 500, 800, 1000, 1500],
                                        'hidden_layer_sizes': [(10), (10,2), (5,), (5,2), (9,), (9,2)],
                            'alpha': [0.0001, 1e-5, 0.01, 0.001],
                            'solver': ['lbfgs', 'sgd', 'adam'],
                            'learning_rate_init': [0.001, 0.01, 0.1, 0.2, 0.3],
                            'activation': ["logistic", "relu", "tanh"]}]

        results = GridSearchCV(reg, param_grid=grid_values, cv=5, iid=False, scoring='neg_mean_squared_error')
        results.fit(X_train, y_train)
        if name in 'MLP':
            print(name + ": " + str(results.best_estimator_.hidden_layer_sizes))
        y_pred = results.best_estimator_.predict(X_test)
        print(name + ' Mean Absolute Error: ' + str(metrics.mean_absolute_error(y_test, y_pred)))
        print(name + ' Mean Squared Error: ' + str(metrics.mean_squared_error(y_test, y_pred)))
        print(name + ' Root Mean Squared Error: ' +  str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))
        print(name + ' r2: ' + str(metrics.r2_score(y_test, y_pred)))

        if name not in 'MLP':
            params = np.append(results.best_estimator_.intercept_, results.best_estimator_.coef_)
            newX = pd.DataFrame({"Constant": np.ones(len(X_test))}).join(pd.DataFrame(X_test))
            MSE = (sum((y_test - y_pred) ** 2)) / (len(newX) - len(newX.columns))

        # Note if you don't want to use a DataFrame replace the two lines above with
        # newX = np.append(np.ones((len(X),1)), X, axis=1)
        # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

            var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
            sd_b = np.sqrt(var_b)
            ts_b = params / sd_b

            p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

            sd_b = np.round(sd_b, 3)
            ts_b = np.round(ts_b, 3)
            p_values = np.round(p_values, 3)
            params = np.round(params, 4)

            myDF3 = pd.DataFrame()
            myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilites"] = [params, sd_b, ts_b,
                                                                                                     p_values]
            print(myDF3)

        if "Rid" in name:
            filename = config.RAPPORT_ESTIMATOR_MODEL_TOSCORE1
            filename2 = config.RAPPORT_ESTIMATOR_TEST_MODEL
            pickle.dump(results.best_estimator_, open(filename, 'wb'))


            loaded_model = pickle.load(open(filename, 'rb'))
            result = loaded_model.score(X_test, y_test)
            print("pickle prediction: " + str(result))
            loaded_model = pickle.load(open(filename2, 'rb'))
            result = loaded_model.score(X_test, y_test)
            print("pickle prediction 2: " + str(result))



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

        if "MLP" in name:
            filename = config.RECO_ACCEPTANCE_MODEL
            pickle.dump(results.best_estimator_, open(filename, 'wb'))

            loaded_model = pickle.load(open(filename, 'rb'))
            result = loaded_model.score(X_test, y_test)
            print("pickle prediction: " + str(result))

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


def rapport_estimator_validate():
    dataset_name = "reciprocity_dataset.csv"
    cross_validate(dataset_name)



#################################################################################################################
#################################################################################################################
##############                                                                                ###################
##############                          OnLine Rapport Estimation Functions                   ###################
##############                                                                                ###################
#################################################################################################################
#################################################################################################################

def estimate_rapport(data):
    filename = config.RAPPORT_ESTIMATOR_MODEL
    loaded_model = pickle.load(open(filename, 'rb'))
    data = scale_data(data)
    result = loaded_model.predict(data)
    return result

def get_rapport_reward(rapport_score, none_ratio):
    #if "P" in user_type:
    #   reward = none_ratio *100
    #else:
    reward = (rapport_score/7) * 100
    #if rapport_score <4:
    #    reward = -50
    #elif rapport_score <5:
    #    reward = 0
    #elif rapport_score <6:
    #    reward = 50
    #elif rapport_score <7:
    #    reward = 100
    return reward
    #Todo CHange Reward function so that introduction gives some rapport?

if __name__ == '__main__':
    #build_reciprocity_dataset()
    #cross_validate()
    build_task_model()

def test_re():
    data = []
    data.extend([3, 3, 2, 2, 0])
    data.extend([2, 3, 0, 3, 1])
    rapport = ml_models.estimate_rapport(data)
    print("Rapport: " + str(rapport))

#################################################################################################################
#################################################################################################################
##############                                                                                ###################
##############                                  DQN Functions                                 ###################
##############                                                                                ###################
#################################################################################################################
#################################################################################################################

def _build_DQN_model(state_size, action_size):
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate))
    return model

