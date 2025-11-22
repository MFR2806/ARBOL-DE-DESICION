# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 14:55:33 2025

@author: maria
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

try:
  
    df = pd.read_csv("shopping_behavior_updated (1).csv")

  
    df.dropna(subset=['Subscription Status', 'Frequency of Purchases'], inplace=True)


    df["Subscription Status"] = df["Subscription Status"].map({"No": 0, "Yes": 1})
    

    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    

    freq_dummies = pd.get_dummies(df["Frequency of Purchases"], prefix="Freq", drop_first=True)
    

    df = pd.concat([df, freq_dummies], axis=1)


    feature_cols = [
        'Age',
        'Gender',
        'Purchase Amount (USD)',
        'Review Rating',
        'Previous Purchases',
    ]
    

    feature_cols.extend(freq_dummies.columns)

    X = df[feature_cols]           
    y = df["Subscription Status"]   

 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
    

    modelo.fit(X_train, y_train)

 
    y_pred = modelo.predict(X_test)
    

    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- Evaluación del Modelo (Predecir Suscripción) ---")
    print(f"Variables usadas: {feature_cols}")
    print(f"Accuracy (Precisión) del modelo: {accuracy:.2%}")


    
    importances = modelo.feature_importances_
    

    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    })
    

    top_features = feature_importance_df.sort_values(by='importance', ascending=False)
    top_features = top_features[top_features['importance'] > 0] 
    
    print("\n--- Variables Más Importantes ---")
    print(top_features.to_string())
    

    plt.figure(figsize=(10, 6))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel("Importancia de la Variable")
    plt.ylabel("Variable")
    plt.title("Importancia de Variables para Predecir la Suscripción")
    plt.gca().invert_yaxis() 
    plt.tight_layout() 
    
   
    plt.savefig("subscription_feature_importance.png", dpi=300)
    print("Gráfico 'subscription_feature_importance.png' guardado.")
    plt.clf() 


    
  
    plt.figure(figsize=(25, 12)) 
    
    plot_tree(
        modelo,
        feature_names=feature_cols,    
        class_names=['No Subscribed', 'Subscribed'],  
        filled=True,                    
        rounded=True,                   
        fontsize=8                      
    )
    
    plt.title("Árbol de Decisión: ¿Qué clientes se suscriben?", fontsize=20)
    
    # Guardar el gráfico del árbol
    plt.savefig("subscription_tree.png", dpi=300) # dpi alto para buena resolución
    print("Gráfico 'subscription_tree.png' guardado.")


except FileNotFoundError:
    print("Error: No se encontró el archivo 'shopping_behavior_updated (1).csv'.")
except Exception as e:
    print(f"Ocurrió un error: {e}")