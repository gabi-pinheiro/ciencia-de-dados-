import os
import pandas as pd
import matplotlib.pyplot as plt


def main(csv_path='weather_classification_data.csv', out_path='plots/weather_type_stacked.png'):
    df = pd.read_csv(csv_path)
    col = 'Weather Type'
    counts = df[col].value_counts()

    labels = counts.index.tolist()
    sizes = counts.values

    # use a broader, vibrant palette
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    left = 0
    total = sizes.sum()
    for i, (label, size) in enumerate(zip(labels, sizes)):
        frac = size / total
        ax.barh(0, frac, left=left, color=colors[i], edgecolor='white')
        # annotate in the center of the segment if wide enough, else outside
        cx = left + frac / 2
        if frac >= 0.05:
            # choose white or black text for contrast
            r, g, b, _ = colors[i]
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = 'white' if luminance < 0.6 else 'black'
            ax.text(cx, 0, f"{label}\n{size}", va='center', ha='center', color=text_color, fontsize=9)
        else:
            ax.text(left + frac + 0.005, 0, f"{label} ({size})", va='center', ha='left', fontsize=9)
        left += frac

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.set_title('Distribuição de Weather Type')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Gráfico salvo em {out_path}')


if __name__ == '__main__':
    main()
