# scd-feature-reduction

## FEITO:

### SVM, KNN e RegLog com CV/K-Fold

- Usando: Features da planilha "unificada" (chamada de **"time"** nos arquivos), hrv_time (neurokit2) e wavelet, planilha + hrv_time, todos os 3

- usando só hrv_time ou wavelet sairam ruins

- hrv_frequency e hrv_nonlinear **não foram calculadas** por causa do custo computacional (memoria acima de 90% em pouco tempo)

- tem a biblioteca do `biosppy` tb que calcula e gasta menos memória, mas num holter demorou mais de 1hr e ainda nao tinha terminado

  > seria uma eternidade para calcular de todos os 176 holters

- Reducao com selectkbest e usando matriz de correlacao

- Explicação do metodo da correlacao: remove colunas que possui algum abs(corr_coef) > limiar

- do "time" em si foram os melhores, mas quando junta com hrv_time e wavelet tem/pode ter uma pequena melhora nos resultados, principalmente na parte de cross validation

  > em alguns modelos, o conjunto de teste não teve diferença significativa nas métricas

### IsolationForest, RandomForest, GradientBoosting e XGBoost

- Primeiro foi feito um grid search na "força bruta":

  1. separado em treino e teste de forma 80-20 e de forma estratificada
  2. usava as possíveis combinações com os dados de treino para treinar o modelo e testava no conjunto de teste e ficava com o melhor

- **problema**: usa o conjunto de teste para escolher os hiperparâmetros, podendo "overfittar" o conjunto de teste (aqueles parâmetros podem ficar muito bons para aquele conjunto de teste, mas nao garante a generalização)

- Criado código para a utilização do GridSearchCV e utilizar os mesmos hiperparâmetros acima

  > Separado os dados em treino e teste (80-20) e os dados de treino usado para cross validation / grid search

  > Depois do grid search, o modelo é treinado com os melhores hiperparametros e os 80% escolhido para treino, e então testado no conjunto de teste

- a escolha dos parâmetros do GridSearch podem performar não tão bem no conjunto de teste escolhido, mas pode generalizar bem outros dados

- **Teste no XGBoost**: K-fold CV durante o GridSearchCV deu resultado mais baixos nas métricas comparado com a versão "força bruta", mas a média das métricas do conjunto de validação no k-fold foram bem melhores que os dois, o que pode indicar uma melhor generalização (**será ?**)

- F1-score foi de 69.56, sendo que no cross-validation foi de 83.85 +- 8.39, e o força bruta de 78.76 %.
  > conjunto de teste pequeno, então as métricas podem ser mais "volateis"

## TODO (possui algumas sugestoes do deepseek e chatgpt):

- testar com `RFE`

  > criado, nao testado nos metodos baseados em arvores, mas na regressao logistica nao houve diferença comparado com o SelectKBest

- "Nested cross-validation"

- oversampling usando `SMOTE`

- tentar extrair de novo extrair features nao lineares e/ou da frequência de um holter longo com o `biosppy` e ver quanto tempo gasta (ou ver uma solução melhor para extrair elas)

- XGBoost é tão bom quanto GradientBoosting e muito mais rápido, o que será melhor na hora de aplicar o GridSearchCV

- No xgboost, usar **SHAP plot** para "explicar" predições

> Features extraídas (HRV, morfológicas, etc) podem ser mais discriminativas do que ECG puro para o risco de SCD em longo-termo.
