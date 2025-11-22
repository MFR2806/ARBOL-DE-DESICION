from modulo_1 import cargar_datos
from modulo_2 import preprocesar_datos
from modulo_3 import entrenar_modelo
from modulo_4 import evaluar_modelo
from modulo_5 import graficar_importancia, graficar_arbol

def main():
    df = cargar_datos("shopping_behavior_updated (1).csv")
    if df is None:
        return
    X, y, feature_cols = preprocesar_datos(df)
    modelo, X_train, X_test, y_train, y_test = entrenar_modelo(X, y)
    df_import = evaluar_modelo(modelo, X_test, y_test, feature_cols)
    df_import = df_import[df_import["importance"] > 0]
    graficar_importancia(df_import)
    graficar_arbol(modelo, feature_cols)

if __name__ == "__main__":
    main()
