import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

# Carregar os dados e o modelo
with open('census.pkl', 'rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

# Configurar o classificador KNN
knn_census = KNeighborsClassifier(n_neighbors=10)
knn_census.fit(X_census_treinamento, y_census_treinamento)

# Previsões
previsoes = knn_census.predict(X_census_teste)

# Interface do Streamlit
st.title('Classificação com K-Nearest Neighbors (KNN)')

st.header('Resultados da Classificação')
st.write('Previsões:')
st.write(previsoes)
st.write('Valores reais:')
st.write(y_census_teste)

# Métricas de Avaliação
accuracy = accuracy_score(y_census_teste, previsoes)
st.subheader('Acurácia do Modelo')
st.write(f'Acurácia: {accuracy:.3f}')

# Matriz de Confusão
st.subheader('Matriz de Confusão')
cm = ConfusionMatrix(knn_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm_score = cm.score(X_census_teste, y_census_teste)
st.pyplot(cm.poof())  # Exibe a matriz de confusão

# Relatório de Classificação
st.subheader('Relatório de Classificação')
class_report = classification_report(y_census_teste, previsoes, output_dict=True)
st.write(classification_report(y_census_teste, previsoes))

# Fórmula da Distância Euclidiana
st.subheader('Fórmula da Distância Euclidiana')
st.latex(r'''
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
''')

# Fórmula Instanciada (Exemplo de cálculo)
st.subheader('Exemplo de Cálculo da Distância Euclidiana')

# Calcula a distância euclidiana entre dois pontos exemplo
def distancia_euclidiana(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

# Pontos exemplo (usando as primeiras amostras de X_teste para simplicidade)
ponto1 = X_census_teste[0]
ponto2 = X_census_teste[1]
distancia = distancia_euclidiana(ponto1, ponto2)

st.write(f'Ponto 1: {ponto1}')
st.write(f'Ponto 2: {ponto2}')
st.write(f'Distância Euclidiana: {distancia:.3f}')

# Exemplo de cálculo da fórmula instanciada
st.latex(r'''
\text{Para os pontos} \quad x = \left[{}''' + ', '.join(f'{val:.2f}' for val in ponto1) + r'''\right] \quad \text{e} \quad y = \left[{}''' + ', '.join(f'{val:.2f}' for val in ponto2) + r'''\right] \\
d(x, y) = \sqrt{(''' + ' + '.join([f'({p1:.2f} - {p2:.2f})^2' for p1, p2 in zip(ponto1, ponto2)]) + r''')}
''')

# Exibir a fórmula da distância euclidiana instanciada
st.write(f'Fórmula Instanciada: {distancia:.3f}')
