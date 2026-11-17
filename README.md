# Histograma da Temperatura

Script para gerar um histograma da coluna de temperatura (primeira coluna) do CSV `weather_classification_data.csv`.

Uso:

```bash
python plot_temperature_histogram.py weather_classification_data.csv
```

Exemplos:

Salvar imagem png:

```bash
python plot_temperature_histogram.py weather_classification_data.csv --out temperatura.png
```

Alterar número de bins:

```bash
python plot_temperature_histogram.py weather_classification_data.csv --bins 50
```

Gerar o gráfico de barra empilhada de `Weather Type`:

```bash
python plot_weather_type_stacked.py
```

Instalação rápida:

```bash
pip install -r requirements.txt
```

Gerar os gráficos e as medidas de tendência central:

```bash
python generate_all_plots.py
```

O comando salva as medidas em `central_tendency.csv`. Para colunas numéricas,
o relatório inclui média, mediana e moda. Para colunas categóricas, inclui a
moda, que é a medida de tendência central apropriada para esse tipo de dado.

O mesmo comando também salva as medidas de dispersão em `dispersion.csv`:
mínimo, máximo, percentis 10, 25, 50, 75 e 90, amplitude, intervalo
interquartil, variância, desvio padrão, MAD (desvio absoluto mediano) e
coeficiente de variação para colunas numéricas (o CV só é calculado para
variáveis em escala de razão, sem valores negativos; por isso fica vazio para
`Temperature`, que está em Celsius e tem valores negativos). Além disso, salva em `plots/`
gráficos bivariados para todos os pares de características: dispersões para
pares numéricos, boxplots para pares numérico-categóricos e mapas de calor de
contagens para pares categórico-categóricos.

Os gráficos `dispersion_boxplot.png` e `dispersion_violin.png` mostram a
dispersão e a distribuição de cada característica numérica. Os arquivos
`scatter_*.png`, `boxplot_*_by_*.png` e `heatmap_*_by_*.png` mostram as
associações entre pares de variáveis conforme seus tipos.

## Pré-processamento

```bash
python preprocess.py
```

Implementa o pipeline da Aula 06b/07a, na ordem recomendada. Todo parâmetro
aprendido dos dados (percentis, medianas, média/desvio) é calculado apenas no
treino e aplicado no teste, para evitar vazamento de dados.

1. **Split treino/teste** estratificado 80/20 (`random_state=42`).
2. **Anonimização**: não necessária (sem identificadores ou dados pessoais).
3. **Limpeza**: padronização de strings, remoção de duplicatas e de instâncias
   com mesmos atributos e classes diferentes (nenhuma encontrada), e valores
   fisicamente impossíveis convertidos em ausentes: `Humidity` e
   `Precipitation (%)` > 100, `Temperature` > 60 °C e `Atmospheric Pressure`
   fora de [870, 1085] hPa (~10,7% das instâncias).
4. **Outliers**: contagem pelos critérios IQR e z-score; capping nos percentis
   1 e 99 do treino.
5. **Codificação/transformação**: `Cloud Cover` ordinal (clear < partly cloudy
   < cloudy < overcast), `Season` e `Location` one-hot (k-1 colunas),
   `Weather Type` label encoding; `sqrt`/`log1p` nas colunas com assimetria > 1
   (`Wind Speed`, `Visibility (km)`).
6. **Engenharia de características**: não aplicada nesta etapa.
7. **Imputação** pela mediana.
8. **Balanceamento**: verificado; as classes já estão balanceadas (25% cada).
9. **Normalização** z-score (`StandardScaler`).

Saídas em `preprocessed/`: `train_preprocessed.csv`, `test_preprocessed.csv`,
`train_preprocessed_unscaled.csv`, estatísticas descritivas antes/depois,
`outliers.csv`, gráficos em `preprocessed/plots/` e o relatório
`RELATORIO_PREPROCESSAMENTO.md` (número de instâncias, de atributos,
distribuição de classes e o que mudou em cada etapa).

## Redução de dimensionalidade

```bash
python dimensionality_reduction.py
```

Usa os dados gerados por `preprocess.py` (Aulas 08a e 08b):

- **PCA**: coeficientes dos componentes, variância explicada e acumulada,
  scree plot; escolhe o menor número de componentes que explica >= 90% da
  variância.
- **Seleção de características**: filtro por informação mútua e wrapper
  Sequential Forward Selection com KNN (validação cruzada de 5 folds).
- **Comparação** da acurácia de um KNN com todas as características, com os
  componentes do PCA e com as características selecionadas.

Saídas em `dimensionality_reduction/`, incluindo o relatório
`RELATORIO_REDUCAO_DIMENSIONALIDADE.md`.
