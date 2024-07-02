import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data_url = "http://bit.ly/w-data"
data = pd.read_csv(data_url) 

X = data[['Hours']]
Y = data[['Scores']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, Y_Train)

Y_pred = model.predict(X_team)

print('Mean Absolute Error:', mean_absolute_error(Y_test, Y_pred))
print('R2 Score:', r2_score(Y_test, Y_pred))

hours = [[9,25]]
predicted_score = model.predict(hours)
print(f'Predicted score for 9.25 hours/day: {predicted_score[0]}')
