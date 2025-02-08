from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_decision_tree(X_train, Y_train, random_state=42):
	clf = DecisionTreeClassifier(random_state=random_state)
	clf.fit(X_train, Y_train)
	return clf

def evaluate_model(clf, X_test, Y_test):
	Y_pred = clf.predict(X_test)
	accuracy = accuracy_score(Y_test, Y_pred)
	report = classification_report(Y_test, Y_pred)
	return Y_pred, accuracy, report