from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Separar las características (X) y la variable objetivo (y)
X = df_humedades_filtered.drop(['cerrado_ganado'], axis=1)
y = df_humedades_filtered['cerrado_ganado']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo con class_weight ajustado
model = RandomForestClassifier(class_weight={0: 1, 1: 5}, random_state=42)
model.fit(X_train, y_train)

# Predecir las probabilidades de la clase 1
y_prob = model.predict_proba(X_test)[:, 1]

# Ajustar el umbral a 0.4 para mejorar el equilibrio entre precisión y recall
umbral = 0.4
y_pred = (y_prob > umbral).astype(int)

# Generar el reporte de clasificación
print("Reporte de Clasificación con umbral ajustado a 0.4:")
print(classification_report(y_test, y_pred, target_names=['Clase 0', 'Clase 1']))

# Mostrar la matriz de confusión
print("Matriz de Confusión con umbral ajustado a 0.4:")
print(confusion_matrix(y_test, y_pred))


