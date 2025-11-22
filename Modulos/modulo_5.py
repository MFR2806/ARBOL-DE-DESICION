import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def graficar_importancia(df_import):
    plt.figure(figsize=(10, 6))
    plt.barh(df_import['feature'], df_import['importance'])
    plt.title("Importancia de Variables")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("subscription_feature_importance.png")
    plt.clf()

def graficar_arbol(modelo, feature_cols):
    plt.figure(figsize=(25, 12))
    plot_tree(
        modelo,
        feature_names=feature_cols,
        class_names=['No Subscribed', 'Subscribed'],
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.title("Árbol de Decisión")
    plt.savefig("subscription_tree.png")
    plt.clf()
