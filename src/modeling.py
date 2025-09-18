import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Support Vector Regressor": SVR()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R²": r2_score(y_test, y_pred),
            "model": model
        }

    # Select best model by R²
    best_model_name = max(results, key=lambda x: results[x]["R²"])
    best_model = results[best_model_name]["model"]

    return results, best_model_name, best_model


#Train the actual model
def train_gradient_boosting(X_train, X_test, y_train, y_test):
    
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, y_train)

    # Predictions
    y_pred = gb_model.predict(X_test)
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "R²": r2
    }

    return gb_model, metrics, y_pred


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        return importance_df
    else:
        return None
