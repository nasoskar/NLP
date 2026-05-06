from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

class SVM_Classifier():
    def __init__(self):
        self.svc = svm.SVC(class_weight='balanced', random_state=42)
        self.parameters = {'kernel':['linear', 'rbf'], 'C':[0.01, 0.1, 1, 10, 100]}
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True, max_df=0.95)
        self.clf = GridSearchCV(self.svc, self.parameters, cv=5, scoring="f1_macro", n_jobs=-1, verbose=2)
    
    def fit(self, X_train, y_train):
        X = self.vectorizer.fit_transform(X_train)
        self.clf.fit(X, y_train)
        print("Best params:", self.clf.best_params_)
        print("Best macro F1:", self.clf.best_score_)

    def predict(self, X_test):
        X = self.vectorizer.transform(X_test)  
        return self.clf.predict(X)
    