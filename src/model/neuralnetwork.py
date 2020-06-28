from sklearn.neural_network import MLPClassifier
import time
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def clf_report(clf, X_train, y_train, X_test, y_test):
    start = time.time()
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_weighted')
    print("Cross validation F1 scores:", scores)
    print("Average cross validation F1:", scores.mean())
    print(f"Cross validation time: {time.time()-start} seconds")
    
    start = time.time()
    clf.fit(X_train, y_train)
    print(f"Time taken to train on full training set: {time.time() - start} seconds\n")
    print(classification_report(y_test, clf.predict(X_test), labels=None, target_names=['Sober', 'Intoxicated'], sample_weight=None, digits=2, output_dict=False, zero_division='warn'))
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_pred, y_test)
    print(f"Overall accuracy: {cm.diagonal().sum()/np.sum(cm)}")
    print(f"Sober accuracy: {cm[0,0]/cm[:,0].sum()}")
    print(f"Intoxicated accuracy: {cm[1,1]/cm[:,1].sum()}")
    return y_pred

def get_predictions(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=1337)
    print(f"Distribution of Intoxicated class in the dataset is {np.round(y.mean(), 3) * 100}%")
    clf = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100, 50), learning_rate='invscaling',
              learning_rate_init=0.001, max_fun=15000, max_iter=400,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=1337, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
    return clf_report(clf, X_train, y_train, X_test, y_test)