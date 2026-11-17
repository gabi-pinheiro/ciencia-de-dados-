"""Redução de dimensionalidade dos dados pré-processados (Aulas 08a e 08b).

- Fusão de características: PCA (componentes, variância explicada e acumulada)
- Seleção de características:
  * filtro: informação mútua entre cada característica e a classe
  * wrapper: Sequential Forward Selection com KNN (SequentialFeatureSelector)
- Comparação do desempenho de um KNN (validação cruzada no treino) com todas
  as características, com os componentes escolhidos e com as selecionadas.

Rode antes: python preprocess.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from preprocess import OUT_DIR, RANDOM_STATE, TARGET, to_markdown

TRAIN_PATH = os.path.join(OUT_DIR, 'train_preprocessed.csv')
TEST_PATH = os.path.join(OUT_DIR, 'test_preprocessed.csv')
RESULTS_DIR = 'dimensionality_reduction'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
VARIANCE_THRESHOLD = 0.90
SFS_TOLERANCE = 0.002
N_NEIGHBORS = 5


def load(path: str):
    df = pd.read_csv(path)
    return df.drop(columns=TARGET), df[TARGET]


def run_pca(X_train: pd.DataFrame):
    pca = PCA(random_state=RANDOM_STATE).fit(X_train)
    names = [f'PC{i + 1}' for i in range(pca.n_components_)]
    loadings = pd.DataFrame(pca.components_, index=pd.Index(names, name='Componente'),
                            columns=X_train.columns)
    variance = pd.DataFrame({
        'Autovalor': pca.explained_variance_,
        'Variancia_explicada': pca.explained_variance_ratio_,
        'Variancia_acumulada': np.cumsum(pca.explained_variance_ratio_),
    }, index=pd.Index(names, name='Componente'))
    n_components = int(np.searchsorted(variance['Variancia_acumulada'].values, VARIANCE_THRESHOLD) + 1)
    return pca, loadings, variance, n_components


def save_scree_plot(variance: pd.DataFrame, n_components: int, path: str) -> None:
    positions = np.arange(1, len(variance) + 1)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions, variance['Variancia_explicada'], color='#66c2a5', label='Variancia explicada')
    axis.plot(positions, variance['Variancia_acumulada'], marker='o', color='#fc8d62',
              label='Variancia acumulada')
    axis.axhline(VARIANCE_THRESHOLD, color='gray', linestyle='--',
                 label=f'Limiar {int(VARIANCE_THRESHOLD * 100)}%')
    axis.axvline(n_components, color='#8da0cb', linestyle=':', label=f'm = {n_components}')
    axis.set_xticks(positions)
    axis.set_xlabel('Componente principal')
    axis.set_ylabel('Proporcao da variancia')
    axis.set_title('PCA - variancia explicada (scree plot)')
    axis.legend()
    axis.grid(axis='y', alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)


def save_loadings_heatmap(loadings: pd.DataFrame, path: str) -> None:
    figure, axis = plt.subplots(figsize=(12, 7))
    image = axis.imshow(loadings.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axis.set_xticks(range(len(loadings.columns)), labels=loadings.columns, rotation=40, ha='right')
    axis.set_yticks(range(len(loadings.index)), labels=loadings.index)
    for row in range(loadings.shape[0]):
        for column in range(loadings.shape[1]):
            axis.text(column, row, f'{loadings.iat[row, column]:.2f}', ha='center', va='center', fontsize=7)
    figure.colorbar(image, ax=axis, label='Coeficiente')
    axis.set_title('PCA - coeficientes dos componentes principais')
    figure.tight_layout()
    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)


def save_pca_scatter(X_pcs: np.ndarray, y: pd.Series, path: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    for label in sorted(y.unique()):
        mask = (y == label).values
        axis.scatter(X_pcs[mask, 0], X_pcs[mask, 1], s=8, alpha=0.4, label=str(label))
    axis.set_xlabel('PC1')
    axis.set_ylabel('PC2')
    axis.set_title('Instancias de treino nos dois primeiros componentes')
    axis.legend(title=TARGET)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)


def save_bar_plot(values: pd.Series, title: str, xlabel: str, path: str) -> None:
    values = values.sort_values()
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(values.index, values.values, color='#66c2a5')
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis='x', alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)


def cv_accuracy(X, y, cv) -> tuple[float, float]:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=N_NEIGHBORS), X, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    X_train, y_train = load(TRAIN_PATH)
    X_test, y_test = load(TEST_PATH)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    report = ['# Relatorio de reducao de dimensionalidade', '',
              'Gerado automaticamente por `dimensionality_reduction.py` a partir dos dados de',
              f'`{TRAIN_PATH}` (ja padronizados com z-score, por isso o PCA equivale a usar a',
              'matriz de correlacao). PCA e selecao sao ajustados somente no treino.']

    # PCA
    pca, loadings, variance, n_components = run_pca(X_train)
    loadings.to_csv(os.path.join(RESULTS_DIR, 'pca_coeficientes.csv'))
    variance.to_csv(os.path.join(RESULTS_DIR, 'pca_variancia.csv'))
    X_train_pcs = pca.transform(X_train)
    X_test_pcs = pca.transform(X_test)
    pc_columns = loadings.index[:n_components]
    pd.DataFrame(X_train_pcs[:, :n_components], columns=pc_columns).assign(**{TARGET: y_train.values}) \
        .to_csv(os.path.join(RESULTS_DIR, 'train_pca.csv'), index=False)
    pd.DataFrame(X_test_pcs[:, :n_components], columns=pc_columns).assign(**{TARGET: y_test.values}) \
        .to_csv(os.path.join(RESULTS_DIR, 'test_pca.csv'), index=False)
    save_scree_plot(variance, n_components, os.path.join(PLOTS_DIR, 'pca_scree.png'))
    save_loadings_heatmap(loadings, os.path.join(PLOTS_DIR, 'pca_coeficientes.png'))
    save_pca_scatter(X_train_pcs, y_train, os.path.join(PLOTS_DIR, 'pca_pc1_pc2.png'))

    top_features = {pc: ', '.join(loadings.loc[pc].abs().sort_values(ascending=False).index[:3])
                    for pc in pc_columns}
    report += ['', '## PCA', '',
               'Variancia explicada por componente:', '',
               to_markdown(variance.round(4)), '',
               f'- Escolha: m = {n_components} componentes, o menor numero cuja variancia acumulada',
               f'  atinge o limiar de {int(VARIANCE_THRESHOLD * 100)}% '
               f'({variance["Variancia_acumulada"].iloc[n_components - 1] * 100:.2f}%).',
               f'- Reducao: {X_train.shape[1]} -> {n_components} dimensoes.',
               '- Caracteristicas com maior peso (|coeficiente|) em cada componente escolhido:',
               *[f'  - {pc}: {features}' for pc, features in top_features.items()],
               '', 'Coeficientes (autovetores) dos componentes:', '',
               to_markdown(loadings.round(3)), '',
               '![Scree plot](plots/pca_scree.png)', '',
               '![Coeficientes](plots/pca_coeficientes.png)', '',
               '![PC1 x PC2](plots/pca_pc1_pc2.png)']

    # Seleção de características - filtro (informação mútua)
    mutual_info = pd.Series(mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE),
                            index=X_train.columns, name='Informacao_mutua').sort_values(ascending=False)
    mutual_info.to_csv(os.path.join(RESULTS_DIR, 'informacao_mutua.csv'))
    save_bar_plot(mutual_info, 'Informacao mutua com a classe (filtro)', 'Informacao mutua',
                  os.path.join(PLOTS_DIR, 'informacao_mutua.png'))

    # Seleção de características - wrapper (Sequential Forward Selection)
    sfs = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=N_NEIGHBORS),
                                    n_features_to_select='auto', tol=SFS_TOLERANCE,
                                    direction='forward', scoring='accuracy', cv=cv, n_jobs=-1)
    sfs.fit(X_train, y_train)
    selected = X_train.columns[sfs.get_support()].tolist()
    pd.Series(selected, name='Caracteristica').to_csv(
        os.path.join(RESULTS_DIR, 'caracteristicas_selecionadas.csv'), index=False)
    X_train[selected].assign(**{TARGET: y_train.values}).to_csv(
        os.path.join(RESULTS_DIR, 'train_selecionadas.csv'), index=False)
    X_test[selected].assign(**{TARGET: y_test.values}).to_csv(
        os.path.join(RESULTS_DIR, 'test_selecionadas.csv'), index=False)

    report += ['', '## Selecao de caracteristicas', '',
               '### Filtro: informacao mutua', '',
               'Mede a dependencia entre cada caracteristica e a classe (independe do classificador).', '',
               to_markdown(mutual_info.round(4).to_frame()), '',
               '![Informacao mutua](plots/informacao_mutua.png)', '',
               '### Wrapper: Sequential Forward Selection', '',
               f'- Algoritmo: `SequentialFeatureSelector(direction="forward")` com KNN (k={N_NEIGHBORS}).',
               '- Funcao de avaliacao: acuracia media em validacao cruzada estratificada de 5 folds.',
               f'- Criterio de parada: `n_features_to_select="auto"` com `tol={SFS_TOLERANCE}` '
               '(para quando adicionar uma caracteristica',
               f'  melhora a acuracia em menos de {SFS_TOLERANCE * 100:.1f} ponto percentual).',
               '- KNN foi escolhido por ser sensivel a caracteristicas irrelevantes (usa distancia),',
               '  o que torna a selecao mais util do que com um modelo de arvore.',
               f'- Caracteristicas selecionadas ({len(selected)} de {X_train.shape[1]}): {selected}']

    # Comparação
    comparisons = {
        f'Todas as caracteristicas ({X_train.shape[1]})': X_train.values,
        f'PCA ({n_components} componentes)': X_train_pcs[:, :n_components],
        f'Selecao SFS ({len(selected)} caracteristicas)': X_train[selected].values,
    }
    rows = []
    for name, X in comparisons.items():
        mean, std = cv_accuracy(X, y_train, cv)
        rows.append({'Conjunto': name, 'Acuracia_media_CV': round(mean, 4), 'Desvio_CV': round(std, 4)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(os.path.join(RESULTS_DIR, 'comparacao_knn.csv'), index=False)
    report += ['', '## Comparacao (KNN, validacao cruzada de 5 folds no treino)', '',
               to_markdown(comparison, index=False), '',
               'O conjunto de teste nao foi usado em nenhuma decisao; ele foi apenas transformado',
               '(PCA/selecao) e salvo para a etapa de modelagem.']

    report_path = os.path.join(RESULTS_DIR, 'RELATORIO_REDUCAO_DIMENSIONALIDADE.md')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('\n'.join(report) + '\n')
    print('\n'.join(report))
    print(f'\nRelatorio salvo em {report_path}')


if __name__ == '__main__':
    main()
