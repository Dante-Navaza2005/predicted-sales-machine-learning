import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#Passo 1: Etendimento do Desafio

#Passo 2: Entendimento da Área/Empresa

#Passo 3: Extração/Obtenação de Dados

tabela = pd.read_csv("advertising.csv")

print(tabela)

#Passo 4: Ajuste de Dados (Tratamento/Limpeza)

#Passo 5: Analise Explanatoria

print(tabela.corr())

#criar um grafico

sns.heatmap(tabela.corr(), cmap="Blues", annot = True)

#exibe um grafio

plt.show()

#Passo 6: Modelagem + Algoritimos (Aqui que entra a inteligencia artificial se necessário)

y = tabela["Vendas"]

x = tabela[["TV", "Radio", "Jornal"]]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)

# importar a inteligencia artificial

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#criar a inteligencia artificial
 
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

#treinar a inteligencia artificial

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

from sklearn.metrics import r2_score

print(r2_score(y_teste, previsao_regressaolinear)) #0.907
print(r2_score(y_teste, previsao_arvoredecisao)) #0.96

#Passo 7: Interpretação de resultados

#Visualização Grafica das previsoes

tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsão Arvore Decisao"] = previsao_arvoredecisao
tabela_auxiliar["PRevisao Regressão Linear"] = previsao_regressaolinear
print(tabela_auxiliar)

sns.lineplot(data=tabela_auxiliar)
plt.show()

#Passo 8: criar uma nova previsao

nova_tabela = pd.read_csv("novos.csv")
previsao_arvoredecisao
previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)