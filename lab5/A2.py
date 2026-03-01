from sklearn.linear_model import LinearRegression

def perform_regression(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return model, y_train_pred, y_test_pred