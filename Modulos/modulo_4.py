import pandas as pd
from sklearn.metrics import accuracy_score

def evaluar_modelo(modelo, X_test, y_test, feature_cols):
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Evaluación del Modelo ---")
    print(f"Accuracy: {accuracy:.2%}")
    importances = modelo.feature_importances_
    df_import = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values(by="importance", ascending=False)
    print("\n--- Variables Más Importantes ---")
    print(df_import[df_import["importance"] > 0].to_string())
    return df_import
