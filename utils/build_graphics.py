import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_boxplots(df, cols_per_row=3):
    numeric_cols = df.select_dtypes(include="number").columns
    num_cols = len(numeric_cols)
    num_rows = -(-num_cols // cols_per_row)  # округление вверх

    fig, axes = plt.subplots(
        num_rows, cols_per_row, figsize=(6 * cols_per_row, 4 * num_rows)
    )
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i], color="skyblue")
        axes[i].set_title(col)
        axes[i].grid(True)

    # Отключаем пустые сабплоты
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def a(data):
    numeric_df = data.select_dtypes(include=[np.number])

    n_cols = 2
    n_rows = int(np.ceil(len(numeric_df.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

    axes = axes.flatten()

    for i, col in enumerate(numeric_df.columns):
        df = numeric_df[col].dropna().astype(float)

        x_min, x_max = df.min(), df.max()
        y_max = df.value_counts(bins=10).max()

        axes[i].hist(df, bins=20, color="skyblue", edgecolor="black")
        axes[i].set_title(f"Histogram of {col}")
        axes[i].set_xlim(x_min - 1, x_max + 1)
        axes[i].set_ylim(0, y_max + 5)
        axes[i].grid(True)

    # Удалим пустые графики если столбцов меньше, чем подграфиков
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
