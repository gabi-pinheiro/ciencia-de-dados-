import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_column(csv_path, column_index=0, bins=30, output_path=None, kind='hist', order=None):
    df = pd.read_csv(csv_path)
    try:
        series_raw = df.iloc[:, int(column_index)]
    except Exception:
        print('Erro ao acessar a coluna especificada.', file=sys.stderr)
        raise
    col_name = df.columns[int(column_index)] if df.columns is not None else 'Coluna'

    if kind == 'hist':
        series = pd.to_numeric(series_raw, errors='coerce').dropna()

        # adapt bins for integer-like small-range data (e.g., UV Index)
        unique_vals = np.unique(series)
        try:
            is_integer_like = np.all(np.mod(unique_vals, 1) == 0)
        except Exception:
            is_integer_like = False

        if is_integer_like and len(unique_vals) <= bins:
            mn = int(np.nanmin(series))
            mx = int(np.nanmax(series))
            bins_to_use = np.arange(mn - 0.5, mx + 1.5, 1)
        else:
            if series.nunique() < bins:
                bins_to_use = max(3, series.nunique())
            else:
                bins_to_use = bins

        plt.figure(figsize=(8, 5))
        cmap = plt.get_cmap('plasma')
        plt.hist(series, bins=bins_to_use, color=cmap(0.6), edgecolor='white')
        plt.xlabel(col_name)
        plt.ylabel('Frequência')
        plt.title(f'Histograma de {col_name}')
        plt.grid(axis='y', alpha=0.75)

    elif kind == 'bar':
        s = series_raw.astype(str).str.strip()
        counts = s.value_counts()

        # Determine ordering
        if order:
            requested = [o.strip() for o in order.split(',') if o.strip()]
            # keep only requested categories that actually appear
            ordered = [r for r in requested if r in counts.index]
            remaining = [c for c in counts.index if c not in ordered]
            categories = ordered + remaining
        else:
            # sensible default for Cloud Cover
            if 'cloud' in col_name.lower():
                desired = ['clear', 'partly cloudy', 'cloudy', 'overcast']
                idx_lower = {c.lower(): c for c in counts.index}
                ordered = [idx_lower[d] for d in desired if d in idx_lower]
                remaining = [c for c in counts.index if c not in ordered]
                categories = ordered + remaining
            else:
                categories = counts.index.tolist()

        counts = counts.reindex(categories).fillna(0)
        plt.figure(figsize=(8, 5))
        cmap = plt.get_cmap('tab20')

        # color mappings for sensible palettes
        if 'season' in col_name.lower():
            season_map = {'Spring': '#00A86B', 'Summer': '#FFD700', 'Autumn': '#FF8C00', 'Winter': '#003366'}
            colors = [season_map.get(cat, cmap(i % 20)) for i, cat in enumerate(counts.index)]
        elif 'cloud' in col_name.lower():
            cloud_map = {'clear': '#FFD966', 'partly cloudy': '#9BC2E6', 'cloudy': '#6D9EEB', 'overcast': '#3D85C6'}
            colors = [cloud_map.get(cat.lower(), cmap(i % 20)) for i, cat in enumerate(counts.index)]
        else:
            colors = [cmap(i % 20) for i in range(len(counts))]

        plt.bar(counts.index, counts.values, color=colors, edgecolor='white')
        plt.xlabel(col_name)
        plt.ylabel('Contagem')
        plt.title(f'Contagem de {col_name}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

    else:
        raise ValueError('Tipo de gráfico desconhecido: use "hist" ou "bar"')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f'Gráfico salvo em {output_path}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plotar coluna de um CSV (histograma ou gráfico de barras)')
    parser.add_argument('csv', help='Caminho para o arquivo CSV')
    parser.add_argument('--col', default=0, help='Índice da coluna a usar (padrão 0)', type=int)
    parser.add_argument('--bins', default=30, help='Número de bins do histograma', type=int)
    parser.add_argument('--kind', default='hist', choices=['hist', 'bar'], help='Tipo de gráfico: "hist" ou "bar"')
    parser.add_argument('--order', help='Ordem desejada para categorias (vírgula-separadas) ao usar --kind bar', default=None)
    parser.add_argument('--out', help='Caminho para salvar o gráfico como imagem (png, pdf, ...)', default=None)
    args = parser.parse_args()
    plot_column(args.csv, args.col, args.bins, args.out, kind=args.kind, order=args.order)


if __name__ == '__main__':
    main()
