from pathlib import Path

from joblib import dump
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

clf = svm.SVC(gamma=0.001)

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)
clf.fit(X_train, y_train)

path = Path("./models")
path.mkdir(parents=True, exist_ok=True)
dump(clf, f"{path.absolute()}/sklearn_digits.joblib")
