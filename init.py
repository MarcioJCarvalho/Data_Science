# matplotlib para graficos
# seabonr para graficos
# scikit-learn

# siência de dados 

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

tabela = pd.read_csv(r"D:\Faculdade\Intensivão de Python\AULA-04\DataScience\advertising.csv")
print(tabela)

# corelaçao de dados

# criar grafico
sns.heatmap(tabela.corr(), cmap="Wistia", annot=True)

# exibir grafico
plt.show()

# Toda inteligência artificial passa por 2 etapas, 1ª treino, 2ª teste
# 1º separar a base de dadoe em dados de treino e dados de teste

# separar nossos dados em x e y
y = tabela["Vendas"] # quem eu vou prever (calcular)
x = tabela[["TV", "Radio", "Jornal"]] # quem eu vou usar para previsão

# separar os dados em treino
# por padrão o train_test_split pega 80% para treino e 20% para teste
x_treino, x_teste, y_treino, y_teste = tts(x, y)

# criação da inteligência artificial
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# trainamento da inteligência artificial
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

# saber qual é o melhor modelo
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# teste da AI e avaliação do melhor modelo
# calcular o R² -> dis % que o nosso modelo consegue explicar o que acontece
print(metrics.r2_score(y_teste, previsao_arvoredecisao))
print(metrics.r2_score(y_teste, previsao_regressaolinear))

# visualização grafica das previsões
# a arvore de decisão é o melhor modelo

# como fazer as novas previsões
# 1º  importar um nova tabela com as informaçãoes de propaganda em TV, Rádio e Jornal
novos = pd.read_csv(r"D:\Faculdade\Intensivão de Python\AULA-04\DataScience\novos.csv")
                    
# 2º passa a nova tabela para o predict do seu modelo
previsao = modelo_arvoredecisao.predict(novos)
print(previsao)