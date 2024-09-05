import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Carregar os dados e o modelo
with open('census.pkl', 'rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

knn_census = KNeighborsClassifier(n_neighbors=10)
knn_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = knn_census.predict(X_census_teste)

# Calcular a matriz de confusão
cm = confusion_matrix(y_census_teste, previsoes)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', 
            xticklabels=knn_census.classes_, yticklabels=knn_census.classes_)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
st.pyplot(fig)

# Mostrar o relatório de classificação
st.text("Classification Report:")
st.text(classification_report(y_census_teste, previsoes))

# Mostrar a acurácia
accuracy = accuracy_score(y_census_teste, previsoes)
st.text(f"Accuracy: {accuracy:.2f}")

# Fórmula da Distância Euclidiana
st.subheader("Fórmula da Distância Euclidiana")
st.latex(r'''
    d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
''')
