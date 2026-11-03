# Monitoramento de Transformadores com Deep Learning

Este projeto analisa dados horários de transformadores do setor elétrico com três objetivos principais:

- explorar o comportamento temporal da temperatura do óleo e das cargas;
- detectar falhas por meio de um autoencoder;
- prever a temperatura do óleo e classificar falhas com redes neurais.

O trabalho foi desenvolvido por Ana Carolina Dutra Ramos, Camila Gois de Jesus e Guilherme Magalhães Júnior.

## Notebook

- `Entrega_final.ipynb`

## Dataset

O notebook espera o arquivo abaixo no mesmo diretório:

```text
transformers.csv
```

O conjunto registrado possui 17.348 observações horárias, entre julho de 2016 e junho de 2018, e não apresenta valores ausentes ou datas duplicadas. Suas colunas são:

- `date`: data e hora da observação;
- `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL` e `LULL`: variáveis de carga do transformador;
- `OT`: temperatura do óleo;
- `class`: rótulo binário de operação normal ou falha.

A classe é fortemente desbalanceada:

| Classe | Quantidade | Proporção aproximada |
|---|---:|---:|
| Normal (`0`) | 17.175 | 99,0% |
| Falha (`1`) | 173 | 1,0% |

> O CSV não está incluído neste repositório. Antes de redistribuí-lo, verifique sua origem, licença e condições de uso.

## Etapas do projeto

### Análise exploratória

O notebook apresenta estatísticas descritivas, decomposição sazonal da temperatura do óleo, autocorrelação e correlações cruzadas entre `OT` e as variáveis de carga.

### Detecção de anomalias

O experimento combina:

1. criação de lags, médias móveis, volatilidade, diferenças e relações entre temperatura e carga;
2. remoção preliminar de outliers com Isolation Forest;
3. normalização Min-Max;
4. treinamento de um autoencoder apenas com observações normais;
5. classificação como anomalia quando o erro de reconstrução supera o percentil 99.

### Previsão de temperatura

Uma LSTM recebe as 24 horas anteriores das sete variáveis numéricas e prevê `OT` para os quatro horizontes seguintes.

### Classificação de falhas

O notebook testa quatro arquiteturas MLP com pesos de classe balanceados. Uma etapa adicional cria estatísticas históricas em janelas de 3, 6, 12 e 24 horas, totalizando 126 variáveis preditoras após a remoção das primeiras linhas incompletas.

Os cortes de treino e teste são cronológicos, preservando a ordem temporal dos dados.

## Resultados registrados

Os resultados abaixo foram extraídos da última execução salva no notebook e não foram recalculados durante a preparação deste repositório.

### Previsão da temperatura do óleo

| Horizonte | MAE | RMSE |
|---|---:|---:|
| t+1 hora | 0,5681 | 0,7631 |
| t+2 horas | 0,7407 | 1,0015 |
| t+3 horas | 0,8615 | 1,1679 |
| t+4 horas | 0,9724 | 1,3193 |

O erro aumenta de forma gradual conforme o horizonte de previsão se distancia.

### Detecção e classificação de falhas

- O autoencoder detectou 8 das 173 falhas, resultando em recall de aproximadamente `4,62%` para a classe de falha.
- As quatro MLPs sem engenharia temporal classificaram todas as amostras de teste como normais, apesar da acurácia aparente de `97%`.
- Com engenharia de atributos e limiar padrão de `0,5`, a MLP continuou sem detectar falhas.
- Ao reduzir o limiar para `0,1`, foram detectadas 6 das 103 falhas do período de teste, com precisão de `3%` e recall de `6%`.

Esses resultados mostram que a previsão de temperatura funcionou consideravelmente melhor do que a detecção de falhas. A acurácia global não deve ser usada isoladamente neste problema devido ao forte desbalanceamento.

## Como executar

1. Instale o Python 3.10 ou superior.
2. Coloque `transformers.csv` no mesmo diretório do notebook.
3. Instale as dependências:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow
```

4. Abra o notebook:

```bash
jupyter notebook Entrega_final.ipynb
```

O treinamento das redes pode ser realizado em CPU, mas uma GPU compatível com TensorFlow reduz o tempo necessário.

## Limitações e próximos passos

- O conjunto possui apenas 173 falhas, dificultando a aprendizagem da classe minoritária.
- O percentil 99 do erro de reconstrução não foi selecionado diretamente para maximizar recall, F1 ou PR-AUC.
- As arquiteturas MLP não conseguiram separar adequadamente as falhas no período final.
- O limiar de decisão deve ser escolhido em um conjunto de validação, não no teste.
- PR-AUC, recall e matriz de confusão são mais informativos que acurácia para este problema.
- Técnicas como focal loss, amostragem temporal, modelos de boosting e análise dos eventos de falha podem ser investigadas.
- Para uso operacional, modelos, scalers, limiares e versões das dependências devem ser persistidos.

## Tecnologias

- Python;
- pandas e NumPy;
- Matplotlib e Seaborn;
- scikit-learn;
- statsmodels;
- TensorFlow/Keras.

## Licença

Nenhuma licença foi definida para este projeto. Adicione um arquivo `LICENSE` antes de permitir reutilização ou redistribuição do código.
