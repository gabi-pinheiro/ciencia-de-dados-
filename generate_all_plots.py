import os
import re
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
from plot_temperature_histogram import plot_column


def safe_name(s: str) -> str:
    s = s.lower()
    s = s.replace('%', 'pct')
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s


def calculate_central_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate measures that are meaningful for each column type."""
    results = []

    for column in df.columns:
        series = df[column].dropna()
        modes = series.mode()
        row = {
            'Caracteristica': column,
            'Tipo': 'numerica' if pd.api.types.is_numeric_dtype(df[column]) else 'categorica',
            'Media': series.mean() if pd.api.types.is_numeric_dtype(df[column]) else pd.NA,
            'Mediana': series.median() if pd.api.types.is_numeric_dtype(df[column]) else pd.NA,
            'Moda': ', '.join(str(value) for value in modes),
        }
        results.append(row)

    return pd.DataFrame(results)


def calculate_dispersion(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate dispersion measures that are meaningful for numeric columns."""
    results = []

    for column in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        row = {
            'Caracteristica': column,
            'Tipo': 'numerica' if is_numeric else 'categorica',
            'Minimo': pd.NA,
            'Percentil_10': pd.NA,
            'Percentil_25': pd.NA,
            'Percentil_50': pd.NA,
            'Percentil_75': pd.NA,
            'Percentil_90': pd.NA,
            'Maximo': pd.NA,
            'Variancia': pd.NA,
            'Desvio_Padrao': pd.NA,
            'Amplitude': pd.NA,
            'Intervalo_Interquartil': pd.NA,
            'MAD': pd.NA,
            'Coeficiente_de_Variacao_pct': pd.NA,
        }

        if is_numeric:
            series = df[column].dropna()
            mean = series.mean()
            median = series.median()
            row.update({
                'Minimo': series.min(),
                'Percentil_10': series.quantile(0.10),
                'Percentil_25': series.quantile(0.25),
                'Percentil_50': median,
                'Percentil_75': series.quantile(0.75),
                'Percentil_90': series.quantile(0.90),
                'Maximo': series.max(),
                'Variancia': series.var(),
                'Desvio_Padrao': series.std(),
                'Amplitude': series.max() - series.min(),
                'Intervalo_Interquartil': series.quantile(0.75) - series.quantile(0.25),
                'MAD': (series - median).abs().median(),
                # CV so faz sentido em escala de razao (zero absoluto, valores >= 0);
                # ex.: temperatura em Celsius tem valores negativos e o CV nao e valido
                'Coeficiente_de_Variacao_pct': (
                    (series.std() / mean) * 100 if mean > 0 and series.min() >= 0 else pd.NA
                ),
            })

        results.append(row)

    return pd.DataFrame(results)


def generate_scatter_plots(df: pd.DataFrame, out_dir: str) -> None:
    """Save one scatter plot for every pair of numeric columns."""
    numeric_columns = df.select_dtypes(include='number').columns

    for x_column, y_column in combinations(numeric_columns, 2):
        plot_data = df[[x_column, y_column]].dropna()
        plt.figure(figsize=(8, 5))
        plt.scatter(plot_data[x_column], plot_data[y_column], alpha=0.35, edgecolors='none')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f'Dispersao: {x_column} x {y_column}')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        filename = f"scatter_{safe_name(x_column)}_{safe_name(y_column)}.png"
        output_path = os.path.join(out_dir, filename)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f'Grafico de dispersao salvo em {output_path}')


def generate_numeric_categorical_plots(df: pd.DataFrame, out_dir: str) -> None:
    """Save a boxplot for every numeric-categorical column pair."""
    numeric_columns = df.select_dtypes(include='number').columns
    categorical_columns = df.select_dtypes(exclude='number').columns

    for numeric_column in numeric_columns:
        for categorical_column in categorical_columns:
            plot_data = df[[numeric_column, categorical_column]].dropna().copy()
            plot_data[categorical_column] = plot_data[categorical_column].astype(str)
            categories = sorted(plot_data[categorical_column].unique(), key=str.lower)
            values = [
                plot_data.loc[plot_data[categorical_column] == category, numeric_column]
                for category in categories
            ]

            figure, axis = plt.subplots(figsize=(9, 5))
            axis.boxplot(values, tick_labels=categories, patch_artist=True,
                         boxprops={'facecolor': '#66c2a5'})
            axis.set_xlabel(categorical_column)
            axis.set_ylabel(numeric_column)
            axis.set_title(f'{numeric_column} por {categorical_column}')
            axis.tick_params(axis='x', rotation=35)
            axis.grid(axis='y', alpha=0.3)
            figure.tight_layout()

            filename = f"boxplot_{safe_name(numeric_column)}_by_{safe_name(categorical_column)}.png"
            output_path = os.path.join(out_dir, filename)
            figure.savefig(output_path, bbox_inches='tight')
            plt.close(figure)
            print(f'Boxplot bivariado salvo em {output_path}')


def generate_categorical_plots(df: pd.DataFrame, out_dir: str) -> None:
    """Save a count heatmap for every pair of categorical columns."""
    categorical_columns = df.select_dtypes(exclude='number').columns

    for x_column, y_column in combinations(categorical_columns, 2):
        plot_data = df[[x_column, y_column]].dropna().astype(str)
        counts = pd.crosstab(plot_data[y_column], plot_data[x_column])

        figure, axis = plt.subplots(figsize=(9, 6))
        image = axis.imshow(counts.values, cmap='YlGnBu', aspect='auto')
        axis.set_xticks(range(len(counts.columns)), labels=counts.columns, rotation=35, ha='right')
        axis.set_yticks(range(len(counts.index)), labels=counts.index)
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
        axis.set_title(f'Contagem: {y_column} x {x_column}')
        figure.colorbar(image, ax=axis, label='Contagem')

        for row_index in range(len(counts.index)):
            for column_index in range(len(counts.columns)):
                axis.text(column_index, row_index, str(counts.iat[row_index, column_index]),
                          ha='center', va='center', color='black', fontsize=8)

        figure.tight_layout()
        filename = f"heatmap_{safe_name(y_column)}_by_{safe_name(x_column)}.png"
        output_path = os.path.join(out_dir, filename)
        figure.savefig(output_path, bbox_inches='tight')
        plt.close(figure)
        print(f'Mapa de calor bivariado salvo em {output_path}')


def generate_bivariate_plots(df: pd.DataFrame, out_dir: str) -> None:
    """Generate plots for numeric and categorical pairs of columns."""
    generate_scatter_plots(df, out_dir)
    generate_numeric_categorical_plots(df, out_dir)
    generate_categorical_plots(df, out_dir)


def generate_distribution_plots(df: pd.DataFrame, out_dir: str) -> None:
    """Save boxplot and violin plot panels for every numeric column."""
    numeric_columns = df.select_dtypes(include='number').columns

    for plot_kind in ('boxplot', 'violin'):
        figure, axes = plt.subplots(3, 3, figsize=(14, 12))
        axes = axes.ravel()

        for axis, column in zip(axes, numeric_columns):
            values = df[column].dropna()
            if plot_kind == 'boxplot':
                axis.boxplot(values, orientation='vertical', patch_artist=True,
                             boxprops={'facecolor': '#66c2a5'})
            else:
                axis.violinplot(values, orientation='vertical', showmeans=True)
            axis.set_title(column)
            axis.set_ylabel('Valor')
            axis.grid(axis='y', alpha=0.3)

        for axis in axes[len(numeric_columns):]:
            axis.set_visible(False)

        figure.suptitle(f'Medidas de dispersao - {plot_kind}', fontsize=16)
        figure.tight_layout()
        output_path = os.path.join(out_dir, f'dispersion_{plot_kind}.png')
        figure.savefig(output_path, bbox_inches='tight')
        plt.close(figure)
        print(f'Grafico de {plot_kind} salvo em {output_path}')


def main():
    csv_path = 'weather_classification_data.csv'
    out_dir = 'plots'
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    central_tendency = calculate_central_tendency(df)
    central_tendency_path = 'central_tendency.csv'
    central_tendency.to_csv(central_tendency_path, index=False)
    print(f'Medidas de tendencia central salvas em {central_tendency_path}')
    print(central_tendency.to_string(index=False))

    dispersion = calculate_dispersion(df)
    dispersion_path = 'dispersion.csv'
    dispersion.to_csv(dispersion_path, index=False)
    print(f'Medidas de dispersao salvas em {dispersion_path}')
    print(dispersion.to_string(index=False))

    generate_bivariate_plots(df, out_dir)
    generate_distribution_plots(df, out_dir)

    for i, col in enumerate(df.columns):
        col_lower = col.lower()
        filename = f"{i:02d}_{safe_name(col)}.png"
        out_path = os.path.join(out_dir, filename)

        # decide chart type
        if pd.api.types.is_numeric_dtype(df[col]):
            kind = 'hist'
            order = None
        else:
            kind = 'bar'
            order = None

        # special ordering for cloud cover
        if 'cloud' in col_lower:
            desired = ['clear', 'partly cloudy', 'cloudy', 'overcast']
            # map to exact existing labels
            idx_lower = {v.lower(): v for v in df[col].astype(str).unique()}
            requested = [idx_lower[d] for d in desired if d in idx_lower]
            order = ','.join(requested) if requested else None

        # special ordering for season
        if 'season' in col_lower:
            desired = ['Spring', 'Summer', 'Autumn', 'Winter']
            idx = {v: v for v in df[col].astype(str).unique()}
            requested = [d for d in desired if d in idx]
            order = ','.join(requested) if requested else None

        # special ordering for weather type (optional sensible order)
        if 'weather' in col_lower:
            desired = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
            idx_lower = {v.lower(): v for v in df[col].astype(str).unique()}
            requested = [idx_lower[d.lower()] for d in desired if d.lower() in idx_lower]
            order = ','.join(requested) if requested else None

        print(f'Gerando {kind} para coluna {i}: {col} -> {out_path}')
        try:
            plot_column(csv_path, i, bins=30, output_path=out_path, kind=kind, order=order)
        except Exception as e:
            print(f'Falha ao gerar gráfico para {col}: {e}')


if __name__ == '__main__':
    main()
