import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Carregar os dados e o modelo
with open('census.pkl', 'rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

# Treinar o modelo KNN
k = 3
knn_census = KNeighborsClassifier(n_neighbors=k)
knn_census.fit(X_census_treinamento, y_census_treinamento)

# Função para prever a classe com base nos atributos preditores
def prever_classe(attributes):
    previsao = knn_census.predict([attributes])
    return previsao[0]

# Interface do Streamlit
st.title('Classificação com KNN')

st.header('Insira os atributos do cliente para previsão:')
atributo1 = st.number_input('Atributo 1', min_value=0.0)
atributo2 = st.number_input('Atributo 2', min_value=0.0)
atributo3 = st.number_input('Atributo 3', min_value=0.0)
atributo4 = st.number_input('Atributo 4', min_value=0.0)

if st.button('Prever Classe'):
    atributos = [atributo1, atributo2, atributo3, atributo4]
    resultado = prever_classe(atributos)
    st.write(f'A previsão para os atributos fornecidos é: {resultado}')

# Avaliação do modelo
previsoes = knn_census.predict(X_census_teste)
accuracy = accuracy_score(y_census_teste, previsoes)
conf_matrix = confusion_matrix(y_census_teste, previsoes)

st.subheader('Avaliação do Modelo')
st.write(f'Acurácia: {accuracy:.2f}')
st.write('Relatório de Classificação:')
st.text(classification_report(y_census_teste, previsoes))

# Exibir a matriz de confusão
st.subheader('Matriz de Confusão')
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Classe Prevista')
ax.set_ylabel('Classe Real')
st.pyplot(fig)

# Fórmulas de distância e KNN
st.subheader('Fórmulas do Algoritmo KNN')

# Fórmula da Distância Euclidiana
st.write("**Distância Euclidiana:**")
st.latex(r"D(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}")

# Fórmula da Distância de Manhattan
st.write("**Distância de Manhattan:**")
st.latex(r"D(x, y) = \sum_{i=1}^n |x_i - y_i|")

# Fórmula da Distância de Minkowski
st.write("**Distância de Minkowski:**")
st.latex(r"D(x, y) = \left(\sum_{i=1}^n |x_i - y_i|^p \right)^{1/p}")

# Fórmula para o KNN
st.write("**Fórmula para a Previsão com KNN:**")
st.latex(r"y = \text{classe mais frequente entre os k vizinhos mais próximos}")

# Rodar o aplicativo: use o comando 'streamlit run app.py' no terminal
