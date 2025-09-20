# Libraries for Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# Libraries for resutls reporting
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score,roc_auc_score, f1_score


def build_baselines(X_train, y_train):
    # Pick baseline models
    uniform_clf = DummyClassifier(strategy='uniform',random_state=0)
    mode_clf = DummyClassifier(strategy='most_frequent',random_state=0)
    prior_clf = DummyClassifier(strategy='prior',random_state=0)


    # Baseline models training
    uniform_clf.fit(X_train,y_train)
    print('Created uniform classifier!')

    mode_clf.fit(X_train,y_train)
    print('Created most frequent/mode classifier!')

    prior_clf.fit(X_train,y_train)
    print('Created prior classifier!')

    return uniform_clf, mode_clf, prior_clf


def build_LRModel(X_train, y_train):
    # Init Logisitic regression
    # LR_clf = LogisticRegression(solver="lbfgs",penalty='none',max_iter=1000,random_state=0)
    LR_clf = LogisticRegression(penalty='none',max_iter=2500)

    # Model training
    LR_clf.fit(X_train,y_train)
    print('Trained logistic regression!')
    return LR_clf
