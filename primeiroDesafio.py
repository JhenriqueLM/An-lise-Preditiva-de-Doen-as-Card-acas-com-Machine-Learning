Este projeto utiliza Machine Learning para prever doenças cardíacas com base em dados clínicos. O código pré-processa os dados, removendo outliers, normalizando variáveis ​​e separando-os para treinamento e teste. Modelos comparados: RandomForest, LogisticRegression, KNN e AdaBoost. O projeto Pandas,Matplotlib,Seaborne Scikit-Learn.




import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Essa funcao possibilita o reajuste de valores exagerados .
# @dados recebe a tabela de dados a ser reajustada 
# @coluna recebi o nome da coluna entre apas duplas
# @novaPOcentagem recebe uma pocentagem tendo 1 = 100% e 0 = 0% 
# @resposta recebi false para apenel mosta o novo valor para a substituicao e true para retorna esse vai que sera aproximado 
def corretorDeOutliers (dados,coluna,novaPocentagem, resposta):
    # print(coluna,novaPocentagem,resposta)
    dadosAtualizados = dados[coluna].quantile(novaPocentagem)
    #print(dadosAtualizados)
    if resposta == True:
        dadosAtualizados = dados[ dados[coluna] <  dados[coluna].quantile(novaPocentagem)]
        return dadosAtualizados



def escolhendoRedeNeural (dadosDoModelo, logicaDaRedeNeuralEMaxArvores):

    global desempenho
    global espectativa
    global relatorio

    # Excluindo da variavel @X a coluna (target) que ser adquirida pela variavel @y
    X = dadosDoModelo.drop(columns=["target"])
    # print(X)

    #@y recebendo a coluna (target) com as informacoes que iremos prever
    y= dadosDoModelo["target"]
    #print(y)

    # Definindo os dados de treinamento e teste
    # @test_size= 0.3 =  30% sera teste e os outros 70% treinamento 
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.3,random_state= 42)
    # Transformando os dados para ficarem no intervalo de 0 a 1
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)

    #Definindo o modelo neural de Random Forest , 
    #@n_estimators = 100 define o numero de arvores de decisao , quanto menor o valor mais raiodo e a resposta e mones precisa e o invesor pra quanto maior 
    redeNeural =logicaDaRedeNeuralEMaxArvores

    #Execulta o treinamento do modelo usando @X_train e com  y_train sendo valores reias 
    redeNeural.fit(X_train,y_train)

    #Execultando  previsoes no dados @X_test
    y_pred = redeNeural.predict(X_test)
    espectativa = f"Espectativa do Treino com rede neural : {redeNeural.score(X_train,y_train) * 100 :.0f}%"
    desempenho = f"Desempenho Real: {accuracy_score( y_test,y_pred) *100 :.0f}%"
    relatorio = classification_report( y_test,y_pred)
    matrix = confusion_matrix( y_test,y_pred)
    # print(desempenho)
    # print(espectativa)

    return matrix

sns.set_style('whitegrid') 
dadosBrutos = pd.read_csv("DisafioDoCoracaoPedroAlmir\heart.csv")
dadosBrutosNaoNulos = dadosBrutos.isnull().sum()
dadosBrutosDuplicados = dadosBrutos.duplicated().sum()
dadosBrutos = dadosBrutos.drop_duplicates()
dados = dadosBrutos.describe()
dadosCorrigindoTrestbps = corretorDeOutliers(dadosBrutos,"trestbps",0.95,True)
dadosCorrigindoChol= corretorDeOutliers(dadosCorrigindoTrestbps,"chol",0.97,True)
dadosCorrigindoThalach = corretorDeOutliers(dadosCorrigindoChol,"thalach",0.98,True)
dadosCorrigindoThalach = dadosCorrigindoChol[ dadosCorrigindoChol["thalach"] >  dadosCorrigindoChol["thalach"].quantile(0.005)]
dadosCorrigindoOldpeak  = corretorDeOutliers(dadosCorrigindoThalach,"oldpeak",0.9,True)
dadosCorrigindoCa = corretorDeOutliers(dadosCorrigindoOldpeak,"ca",0.95,True)
dadosDefinidos = dadosCorrigindoCa

Ran=escolhendoRedeNeural( dadosDefinidos, RandomForestClassifier(n_estimators = 1000))
plt.figure(figsize= (19,21))
plt.subplot(2,4,1)
sns.heatmap(Ran,annot=True)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz")
plt.subplots_adjust(bottom= 0.4)
plt.figtext(0.11,0.5,f"RandomForestClassifier:\n\n{espectativa}.\n\n{desempenho}.")
plt.figtext(0.11,0.31,f" Relatorio: \n{relatorio}")

Log =escolhendoRedeNeural( dadosDefinidos, LogisticRegression(max_iter = 1000))
plt.subplot(2,4,2)
sns.heatmap(Log,annot=True)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz")
plt.subplots_adjust(bottom= 0.4)
plt.figtext(0.33,0.5,f"LogisticRegression:\n\n{espectativa}.\n\n{desempenho}.")
plt.figtext(0.33,0.31,f" Relatorio: \n{relatorio}")

KNe =escolhendoRedeNeural( dadosDefinidos, KNeighborsClassifier(n_neighbors=5))
plt.subplot(2,4,3)
sns.heatmap(KNe,annot=True)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz")
plt.subplots_adjust(bottom= 0.4)
plt.figtext(0.53,0.5,f"KNeighborsClassifier:\n\n{espectativa}.\n\n{desempenho}.")
plt.figtext(0.53,0.31,f" Relatorio: \n{relatorio}")

Ada =escolhendoRedeNeural( dadosDefinidos, AdaBoostClassifier(n_estimators= 1000))
plt.subplot(2,4,4)
sns.heatmap(Ada,annot=True)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz")
plt.subplots_adjust(bottom= 0.4)
plt.figtext(0.73,0.5,f" AdaBoostClassifier:\n\n{espectativa}.\n\n{desempenho}.")
plt.figtext(0.73,0.31,f" Relatorio: \n{relatorio}")

plt.show()
