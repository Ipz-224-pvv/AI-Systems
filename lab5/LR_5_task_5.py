!pip install -q scikit-learn numpy

import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = []
with open("/content/drive/MyDrive/Colab Notebooks/lab5/traffic_data.txt", "r") as f:
    for line in f:
        data.append(line.strip().split(','))
data = np.array(data)

label_encoders = []
X_encoded = np.empty(data.shape, dtype=float)
for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i].astype(float)
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(data[:, i])
        label_encoders.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

model = ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']
encoded_point = []
for i, item in enumerate(test_datapoint):
    if item.isdigit():
        encoded_point.append(int(item))
    else:
        encoder = label_encoders[i]
        encoded_point.append(int(encoder.transform([item])[0]))
encoded_point = np.array(encoded_point)

predicted = int(model.predict([encoded_point])[0])
print(f"Predicted traffic intensity: {predicted}")
