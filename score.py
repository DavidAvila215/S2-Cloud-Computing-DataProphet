import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """
    Evalúa un modelo de clasificación y muestra un informe de clasificación junto con la curva de precisión-recall.
    
    Parámetros:
    - model: Modelo de clasificación entrenado.
    - X_test: Conjunto de datos de prueba (features).
    - y_test: Etiquetas reales de los datos de prueba.
    """
    y_pred = model.predict(X_test)  # Predicciones del modelo
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidad de la clase positiva

    # Reporte de clasificación
    print(classification_report(y_test, y_pred))
    
    # Calcular precisión y recall para la curva
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap_score = average_precision_score(y_test, y_proba)
    
    # Graficar la curva de precisión-recall
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {ap_score:.2f}', color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.show()
