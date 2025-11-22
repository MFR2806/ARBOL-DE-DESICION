
# Decision Tree para Suscripción de Clientes

Este repositorio contiene un proyecto en **Python** que implementa un modelo de clasificación supervisada usando un **árbol de decisión**. El objetivo es predecir si un cliente se suscribirá según sus características (edad, género, monto de compra, frecuencia, calificación, historial de compras, etc.).

---

##  Estructura del Proyecto
proyecto_arbol/
│── data_loader.py
│── preprocess.py
│── model_train.py
│── evaluate.py
│── visualize.py
│── main.py
│── requirements.txt


\* Cada módulo tiene una responsabilidad bien definida:

- `data_loader.py`: carga los datos desde un archivo CSV.  
- `preprocess.py`: realiza limpieza, codificación y preparación de variables.  
- `model_train.py`: divide los datos en conjunto de entrenamiento y prueba, y entrena el árbol de decisión.  
- `evaluate.py`: evalúa el modelo (accuracy e importancia de las variables).  
- `visualize.py`: genera gráficos (importancia de variables y visualización del árbol).  
- `main.py`: orquesta la ejecución de todos los módulos.

---

## Uso

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tu-usuario/decision-tree-subscription.git
  
2. Coloca el dataset en la raíz del proyecto con el nombre shopping_behavior_updated (1).csv (o cambia la ruta en main.py según corresponda).

3.Crea un entorno virtual (recomendado) e instala los requerimientos:

python3 -m venv venv
source venv/bin/activate        # En Linux / MacOS
venv\Scripts\activate           # En Windows

pip install -r requirements.txt

4. Ejecuta el proyecto:
   python main.py

5. Al terminar, se generarán dos gráficos:

subscription_feature_importance.png — importancia de variables

subscription_tree.png — visualización del árbol de decisión

 Resultados

Se imprime en consola el accuracy del modelo.

Se muestra la importancia de cada variable usada por el modelo.

Se guardan gráficos para poder analizarlos (árbol + barras de importancia).
