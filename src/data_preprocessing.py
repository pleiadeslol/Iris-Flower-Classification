import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
	df = pd.read_csv(file_path)
	return df

def split_features_target(df, target_column):
	X = df.drop(target_column, axis=1).values
	Y = df[target_column].values
	return X, Y

def split_train_test(X, Y, test_size=0.2, random_state=42):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
	return X_train, X_test, Y_train, Y_test