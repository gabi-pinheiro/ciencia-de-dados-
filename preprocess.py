"""Pipeline de pré-processamento do dataset weather_classification_data.csv.

Segue a ordem recomendada na Aula 06b/07a (slide "A ordem importa!"):
 1. Split treino / teste
 2. Anonimização e remoção de identificadores diretos
 3. Limpeza: duplicatas, inconsistências, padronização de strings
 4. Tratamento de outliers
 5. Codificação de variáveis categóricas / transformação de distribuição
 6. Engenharia de características (não aplicada nesta etapa)
 7. Imputação de valores ausentes
 8. Balanceamento de classes
 9. Normalização (escalonamento)

Todo parâmetro aprendido dos dados (percentis, medianas, média/desvio etc.)
é calculado SOMENTE no treino e depois aplicado ao teste, para evitar
vazamento de dados (data leakage).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

CSV_PATH = 'weather_classification_data.csv'
OUT_DIR = 'preprocessed'
PLOTS_DIR = os.path.join(OUT_DIR, 'plots')
TARGET = 'Weather Type'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Limites físicos plausíveis. Valores fora deles são erros (outliers tipo 1).
# - Umidade relativa e probabilidade de precipitação são porcentagens: [0, 100]
# - Temperatura: recorde mundial de calor ~56.7 C e de frio ~-89.2 C
# - Pressão ao nível do mar: recordes ~870 hPa (tufão) e ~1084 hPa
VALID_RANGES = {
    'Temperature': (-90.0, 60.0),
    'Humidity': (0.0, 100.0),
    'Wind Speed': (0.0, None),
    'Precipitation (%)': (0.0, 100.0),
    'Atmospheric Pressure': (870.0, 1085.0),
    'UV Index': (0.0, None),
    'Visibility (km)': (0.0, None),
}

CLOUD_COVER_ORDER = ['clear', 'partly cloudy', 'cloudy', 'overcast']
NOMINAL_COLUMNS = ['Season', 'Location']
ORDINAL_COLUMNS = {'Cloud Cover': CLOUD_COVER_ORDER}

CAPPING_PERCENTILES = (0.01, 0.99)
SKEW_THRESHOLD = 1.0
IMBALANCE_RATIO_THRESHOLD = 1.5


def to_markdown(df: pd.DataFrame, index: bool = True) -> str:
    """Converte um DataFrame em tabela Markdown (sem depender do pacote tabulate)."""
    if index:
        table = df.rename_axis(df.index.name or 'Caracteristica').reset_index()
    else:
        table = df
    header = '| ' + ' | '.join(str(c) for c in table.columns) + ' |'
    separator = '|' + '---|' * len(table.columns)
    rows = ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in table.itertuples(index=False)]
    return '\n'.join([header, separator, *rows])


def log_step(report: list, title: str, lines) -> None:
    report.append(f'\n## {title}\n')
    for line in lines:
        report.append(line)
    print(f'\n=== {title} ===')
    for line in lines:
        print(line)


def class_distribution(y: pd.Series) -> str:
    counts = y.value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(2)
    return ', '.join(f'{c}: {n} ({p}%)' for c, n, p in zip(counts.index, counts, pct))


def split_data(df: pd.DataFrame):
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def standardize_strings(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for column in X.select_dtypes(exclude='number').columns:
        X[column] = X[column].astype(str).str.strip().str.lower()
    return X


def remove_duplicates(X: pd.DataFrame, y: pd.Series):
    """Remove instâncias duplicadas e instâncias com mesmos atributos e classes diferentes."""
    data = X.copy()
    data[TARGET] = y.values
    n_before = len(data)

    data = data.drop_duplicates()
    n_exact = n_before - len(data)

    feature_columns = list(X.columns)
    conflicting = data.duplicated(subset=feature_columns, keep=False)
    n_conflicting = int(conflicting.sum())
    data = data[~conflicting]

    return data[feature_columns], data[TARGET], n_exact, n_conflicting


def mark_invalid_values(X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Substitui valores fisicamente impossíveis por NaN (serão imputados no passo 7)."""
    X = X.copy()
    counts = {}
    for column, (low, high) in VALID_RANGES.items():
        invalid = pd.Series(False, index=X.index)
        if low is not None:
            invalid |= X[column] < low
        if high is not None:
            invalid |= X[column] > high
        counts[column] = int(invalid.sum())
        X.loc[invalid, column] = np.nan
    return X, counts


def count_outliers(X: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = X[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        n_iqr = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        z = (values - values.mean()) / values.std()
        n_z = int((z.abs() > 3).sum())
        rows.append({'Caracteristica': column, 'Outliers_IQR': n_iqr, 'Outliers_zscore': n_z})
    return pd.DataFrame(rows)


def fit_capping(X: pd.DataFrame, columns) -> dict:
    low_q, high_q = CAPPING_PERCENTILES
    return {c: (X[c].quantile(low_q), X[c].quantile(high_q)) for c in columns}


def apply_capping(X: pd.DataFrame, limits: dict) -> pd.DataFrame:
    X = X.copy()
    for column, (low, high) in limits.items():
        X[column] = X[column].clip(lower=low, upper=high)
    return X


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for column, order in ORDINAL_COLUMNS.items():
        mapping = {category: code for code, category in enumerate(order)}
        unknown = set(X[column].dropna().unique()) - set(mapping)
        if unknown:
            raise ValueError(f'Categorias desconhecidas em {column}: {unknown}')
        X[column] = X[column].map(mapping).astype(float)

    for column in NOMINAL_COLUMNS:
        categories = sorted(X[column].dropna().unique())
        X[column] = pd.Categorical(X[column], categories=categories)
    return pd.get_dummies(X, columns=NOMINAL_COLUMNS, drop_first=True, dtype=float)


TRANSFORMS = {'log1p': np.log1p, 'sqrt': np.sqrt}


def fit_distribution_transforms(X: pd.DataFrame, columns) -> dict:
    """Escolhe log1p ou sqrt para colunas assimétricas à direita (o que deixar |skew| menor)."""
    chosen = {}
    for column in columns:
        values = X[column].dropna()
        if values.skew() <= SKEW_THRESHOLD or values.min() < 0:
            continue
        chosen[column] = min(TRANSFORMS, key=lambda name: abs(TRANSFORMS[name](values).skew()))
    return chosen


def apply_distribution_transforms(X: pd.DataFrame, transforms: dict) -> pd.DataFrame:
    X = X.copy()
    for column, name in transforms.items():
        X[column] = TRANSFORMS[name](X[column])
    return X


def save_comparison_plot(before: pd.DataFrame, after: pd.DataFrame, columns, path: str) -> None:
    figure, axes = plt.subplots(2, len(columns), figsize=(3 * len(columns), 8))
    for index, column in enumerate(columns):
        axes[0, index].boxplot(before[column].dropna(), patch_artist=True,
                               boxprops={'facecolor': '#fc8d62'})
        axes[0, index].set_title(column, fontsize=9)
        axes[1, index].boxplot(after[column].dropna(), patch_artist=True,
                               boxprops={'facecolor': '#66c2a5'})
        axes[1, index].set_title(column, fontsize=9)
    axes[0, 0].set_ylabel('Original (treino)')
    axes[1, 0].set_ylabel('Pre-processado (treino)')
    figure.suptitle('Distribuicoes antes e depois do pre-processamento')
    figure.tight_layout()
    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)


def save_class_plot(y_train: pd.Series, y_test: pd.Series, path: str) -> None:
    counts = pd.DataFrame({'Treino': y_train.value_counts(), 'Teste': y_test.value_counts()}).sort_index()
    axis = counts.plot(kind='bar', figsize=(8, 5), color=['#66c2a5', '#8da0cb'], edgecolor='white')
    axis.set_title('Distribuicao de classes (Weather Type)')
    axis.set_ylabel('Contagem')
    axis.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def describe(X: pd.DataFrame) -> pd.DataFrame:
    stats = X.describe().T
    stats['skew'] = X.skew()
    stats['nulos'] = X.isna().sum()
    return stats


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    report = ['# Relatorio de pre-processamento',
              '',
              'Gerado automaticamente por `preprocess.py`.']

    df = pd.read_csv(CSV_PATH)
    numeric_columns = df.drop(columns=TARGET).select_dtypes(include='number').columns.tolist()
    log_step(report, 'Dataset original', [
        f'- Instancias: {len(df)}',
        f'- Atributos (sem a classe): {df.shape[1] - 1}',
        f'- Valores ausentes: {int(df.isna().sum().sum())}',
        f'- Distribuicao de classes: {class_distribution(df[TARGET])}',
    ])

    # 1. Split treino / teste (estratificado para manter a proporção de classes)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_raw = X_train.copy()
    log_step(report, '1. Split treino / teste', [
        f'- Split estratificado {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} '
        f'(random_state={RANDOM_STATE})',
        f'- Treino: {len(X_train)} instancias | Teste: {len(X_test)} instancias',
    ])

    # 2. Anonimização
    log_step(report, '2. Anonimizacao', [
        '- O dataset contem apenas medicoes meteorologicas e categorias genericas de local',
        '  (inland/coastal/mountain); nao ha identificadores diretos, quasi-identificadores',
        '  nem atributos sensiveis de pessoas. Nenhuma supressao/tokenizacao foi necessaria.',
    ])

    # 3. Limpeza
    X_train, X_test = standardize_strings(X_train), standardize_strings(X_test)
    X_train, y_train, train_dup, train_conf = remove_duplicates(X_train, y_train)
    X_test, y_test, test_dup, test_conf = remove_duplicates(X_test, y_test)
    X_train, invalid_train = mark_invalid_values(X_train)
    X_test, invalid_test = mark_invalid_values(X_test)
    invalid_lines = [f'  - {c}: treino={invalid_train[c]}, teste={invalid_test[c]} '
                     f'(faixa valida {VALID_RANGES[c]})'
                     for c in VALID_RANGES if invalid_train[c] or invalid_test[c]]
    log_step(report, '3. Limpeza', [
        '- Strings categoricas padronizadas (strip + minusculas).',
        f'- Duplicatas exatas removidas: treino={train_dup}, teste={test_dup}',
        f'- Instancias com mesmos atributos e classes diferentes removidas: '
        f'treino={train_conf}, teste={test_conf}',
        '- Valores fisicamente impossiveis (outliers tipo 1, erro de medicao/digitacao)',
        '  substituidos por NaN para imputacao no passo 7:',
        *invalid_lines,
        f'- Instancias afetadas no treino: {int(X_train[numeric_columns].isna().any(axis=1).sum())} '
        f'({X_train[numeric_columns].isna().any(axis=1).mean() * 100:.2f}%). Como sao >5% das '
        'instancias, optou-se por imputar em vez de remover.',
    ])

    # 4. Outliers (valores possíveis, mas extremos): capping nos percentis 1-99 do treino
    outliers_before = count_outliers(X_train, numeric_columns)
    capping_limits = fit_capping(X_train, numeric_columns)
    X_train = apply_capping(X_train, capping_limits)
    X_test = apply_capping(X_test, capping_limits)
    outliers_after = count_outliers(X_train, numeric_columns)
    outliers = outliers_before.merge(outliers_after, on='Caracteristica', suffixes=('_antes', '_depois'))
    outliers.to_csv(os.path.join(OUT_DIR, 'outliers.csv'), index=False)
    log_step(report, '4. Tratamento de outliers', [
        '- Deteccao no treino pelos criterios do boxplot (1.5*IQR) e do z-score (|z|>3).',
        '- Valores plausiveis, mas extremos (outliers tipo 2), foram mantidos com capping',
        f'  nos percentis {int(CAPPING_PERCENTILES[0] * 100)} e {int(CAPPING_PERCENTILES[1] * 100)} '
        'do treino, preservando o numero de instancias e reduzindo a influencia dos extremos.',
        '',
        to_markdown(outliers, index=False),
        '',
        '- Limites usados: ' + ', '.join(f'{c}=[{lo:.2f}, {hi:.2f}]'
                                         for c, (lo, hi) in capping_limits.items()),
    ])

    # 5. Codificação e transformação de distribuição
    X_train, X_test = encode_features(X_train), encode_features(X_test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
    transforms = fit_distribution_transforms(X_train, numeric_columns)
    skew_before = X_train[list(transforms)].skew()
    X_train = apply_distribution_transforms(X_train, transforms)
    X_test = apply_distribution_transforms(X_test, transforms)
    skew_after = X_train[list(transforms)].skew()
    label_encoder = LabelEncoder().fit(y_train)
    y_train_encoded = pd.Series(label_encoder.transform(y_train), index=y_train.index, name=TARGET)
    y_test_encoded = pd.Series(label_encoder.transform(y_test), index=y_test.index, name=TARGET)
    log_step(report, '5. Codificacao e transformacao de distribuicao', [
        f'- Cloud Cover (ordinal): ordinal encoding {dict(enumerate(CLOUD_COVER_ORDER))}',
        f'- Season e Location (nominais): one-hot com k-1 colunas (drop_first) -> '
        f'{[c for c in X_train.columns if c.startswith(tuple(NOMINAL_COLUMNS))]}',
        f'- Weather Type (classe alvo): label encoding '
        f'{dict(enumerate(label_encoder.classes_))}',
        f'- Transformacao de distribuicao (log1p ou sqrt, a que deixar a assimetria mais proxima'
        f' de 0) nas colunas com assimetria > {SKEW_THRESHOLD} no treino: '
        + ', '.join(f'{c} -> {name} (skew {skew_before[c]:.2f} -> {skew_after[c]:.2f})'
                    for c, name in transforms.items()),
    ])

    # 6. Engenharia de características
    log_step(report, '6. Engenharia de caracteristicas', [
        '- Nao aplicada nesta etapa (sera vista mais a frente na disciplina).',
    ])

    # 7. Imputação (mediana: robusta e adequada para distribuições assimétricas)
    imputer = SimpleImputer(strategy='median')
    missing_before = int(X_train.isna().sum().sum())
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    log_step(report, '7. Imputacao de valores ausentes', [
        '- Imputacao simples pela mediana (ajustada no treino e aplicada no teste).',
        f'- Valores imputados no treino: {missing_before}; nulos restantes: '
        f'treino={int(X_train.isna().sum().sum())}, teste={int(X_test.isna().sum().sum())}',
        '- Medianas usadas (na escala ja transformada do passo 5): ' + ', '.join(f'{c}={v:.2f}' for c, v in zip(X_train.columns, imputer.statistics_)
                                         if c in numeric_columns),
    ])

    # 8. Balanceamento
    counts = y_train.value_counts()
    ratio = counts.max() / counts.min()
    balance_lines = [f'- Distribuicao de classes no treino: {class_distribution(y_train)}',
                     f'- Razao classe majoritaria/minoritaria: {ratio:.2f}']
    if ratio > IMBALANCE_RATIO_THRESHOLD:
        balance_lines.append('- Classes desbalanceadas: considerar subamostragem/SMOTE no treino.')
    else:
        balance_lines.append('- Classes ja balanceadas: balanceamento nao e necessario.')
    log_step(report, '8. Balanceamento de classes', balance_lines)

    # 9. Normalização (z-score)
    X_train_unscaled = X_train.copy()
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    log_step(report, '9. Normalizacao', [
        '- StandardScaler (z-score) ajustado no treino e aplicado no teste. Apos o capping os',
        '  outliers ja tem pouca influencia, entao a padronizacao z-score e adequada e deixa',
        '  todas as caracteristicas na mesma escala (necessario para KNN, SVM, PCA etc.).',
        f'- Media no treino ~ {X_train.mean().abs().max():.1e}, desvio ~ {X_train.std().mean():.3f}',
    ])

    # Saídas
    train_out = X_train.assign(**{TARGET: y_train_encoded})
    test_out = X_test.assign(**{TARGET: y_test_encoded})
    train_out.to_csv(os.path.join(OUT_DIR, 'train_preprocessed.csv'), index=False)
    test_out.to_csv(os.path.join(OUT_DIR, 'test_preprocessed.csv'), index=False)
    X_train_unscaled.assign(**{TARGET: y_train.values}).to_csv(
        os.path.join(OUT_DIR, 'train_preprocessed_unscaled.csv'), index=False)
    describe(X_train_raw[numeric_columns]).to_csv(os.path.join(OUT_DIR, 'describe_train_original.csv'))
    describe(X_train_unscaled).to_csv(os.path.join(OUT_DIR, 'describe_train_preprocessed_unscaled.csv'))
    describe(X_train).to_csv(os.path.join(OUT_DIR, 'describe_train_preprocessed.csv'))

    save_comparison_plot(X_train_raw, X_train_unscaled, numeric_columns,
                         os.path.join(PLOTS_DIR, 'boxplot_antes_depois.png'))
    save_class_plot(y_train, y_test, os.path.join(PLOTS_DIR, 'distribuicao_classes.png'))

    log_step(report, 'Resumo final', [
        f'- Instancias: {len(df)} -> treino {len(X_train)} + teste {len(X_test)} = '
        f'{len(X_train) + len(X_test)}',
        f'- Atributos: {df.shape[1] - 1} -> {X_train.shape[1]} (apos one-hot)',
        f'- Atributos finais: {list(X_train.columns)}',
        f'- Valores ausentes finais: {int(X_train.isna().sum().sum() + X_test.isna().sum().sum())}',
        f'- Distribuicao de classes treino: {class_distribution(y_train)}',
        f'- Distribuicao de classes teste: {class_distribution(y_test)}',
        '',
        'Estatisticas descritivas (treino, sem escalonamento) antes e depois:',
        '',
        '### Antes',
        '',
        describe(X_train_raw[numeric_columns])[['mean', 'std', 'min', '50%', 'max', 'skew']]
        .round(3).pipe(to_markdown),
        '',
        '### Depois (antes da normalizacao z-score)',
        '',
        describe(X_train_unscaled)[['mean', 'std', 'min', '50%', 'max', 'skew']].round(3).pipe(to_markdown),
        '',
        '![Boxplots antes e depois](plots/boxplot_antes_depois.png)',
        '',
        '![Distribuicao de classes](plots/distribuicao_classes.png)',
    ])

    report_path = os.path.join(OUT_DIR, 'RELATORIO_PREPROCESSAMENTO.md')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('\n'.join(report) + '\n')
    print(f'\nRelatorio salvo em {report_path}')


if __name__ == '__main__':
    main()
