"""Plotting and reporting helpers shared by the analysis scripts.

show_report lists the metadata of false negative and false positive samples for a
binary prediction. plot_pca standardizes the expression matrix, runs PCA and draws
every pair of components with the explained variance ratio on the axes.
plot_scatter_boxplot draws expression of one gene per group with an overlaid strip plot
and group means, adding a combined disease-and-cancer group when both are present.
plot_panel_boxplots draws a grouped boxplot of a whole gene panel, optionally z-scored
per gene and ordered by supplied model coefficients. plot_roc_curve draws a ROC curve
with its AUC. plot_split_balance and plot_group_overview compare the composition of
data splits or sample groups in terms of class, sex, age, tumour stage and sequencing
depth.
"""

import os

import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
import pandas as pd
from itertools import combinations
import math


from utilz.constans import HEALTHY, DISEASE, CANCER

def show_report(y_pred, y_test_encoded, dataset, le):
    y_true = y_test_encoded
    y_pred_s = pd.Series(y_pred, index=y_true.index, name="y_pred")

    eval_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_s})
    class_map = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Mapowanie klas:", class_map)

    results = {}

    for cls, cls_code in class_map.items():
        if len(class_map) == 2 and cls_code != 0:
            TP_idx = eval_df.index[(eval_df.y_true == cls_code) & (eval_df.y_pred == cls_code)]
            FP_idx = eval_df.index[(eval_df.y_true != cls_code) & (eval_df.y_pred == cls_code)]
            FN_idx = eval_df.index[(eval_df.y_true == cls_code) & (eval_df.y_pred != cls_code)]
            TN_idx = eval_df.index[(eval_df.y_true != cls_code) & (eval_df.y_pred != cls_code)]

            results[cls] = {
                "TP": TP_idx,
                "FP": FP_idx,
                "FN": FN_idx,
                "TN": TN_idx,
            }

            for key in ["FN", "FP"]:
                print(f"\n--- {key} samples metadata ---")
                for idx in results[cls][key]:
                    sample_meta = dataset.meta.loc[idx]
                    print(f"{key} - Sample ID: {idx}, Metadata:")
                    print(sample_meta["Group"], sample_meta["Sex"], sample_meta["Age"], sample_meta["Stage"])
                    print("---")

    return

def plot_pca(X, y_encoded, n_components, le):
    viz_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=n_components, svd_solver="full", random_state=42))
    ])

    X_pca = viz_pipe.fit_transform(X)
    evr = viz_pipe.named_steps["pca"].explained_variance_ratio_
    classes = le.classes_
    y_int = y_encoded.values

    pairs = list(combinations(range(n_components), 2))
    n_pairs = len(pairs)
    n_cols = min(3, n_pairs)
    n_rows = math.ceil(n_pairs / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx // n_cols][idx % n_cols]
        for k, cls in enumerate(classes):
            mask = (y_int == k)
            ax.scatter(
                X_pca[mask, i], X_pca[mask, j],
                label=str(cls), alpha=0.5, s=45
            )
        ax.set_xlabel(f"PC{i+1} ({evr[i]*100:.1f}% wariancji)")
        ax.set_ylabel(f"PC{j+1} ({evr[j]*100:.1f}% wariancji)")
        ax.set_title(f"PC{i+1} vs PC{j+1}")
        ax.legend(title="Klasa", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
    for idx in range(n_pairs, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("PCA – wszystkie pary składowych", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

def plot_scatter_boxplot(X, y, gene_name):
    unique_groups = np.unique(y)
    has_disease = DISEASE in unique_groups
    has_cancer = CANCER in unique_groups

    if has_disease and has_cancer:
        disease_mask = y == DISEASE
        cancer_mask = y == CANCER
        combined_mask = disease_mask | cancer_mask
        X_combined = X[combined_mask]
        y_combined = np.full(len(X_combined), 'Disease and Cancer')
        X_plot = np.concatenate([X, X_combined])
        y_plot = np.concatenate([y, y_combined])
    else:
        X_plot = X
        y_plot = y
    plt.figure(figsize=(10, 10))

    ax = sns.boxplot(y=X_plot, x=y_plot,
                     width=0.5,
                     linewidth=2)

    sns.stripplot(y=X_plot, x=y_plot,
                  color='black',
                  alpha=0.5,
                  size=5,
                  jitter=0.2)

    for i, group in enumerate(np.unique(y_plot)):
        mask = y_plot == group
        mean_val = np.mean(X_plot[mask])
        ax.hlines(mean_val, i - 0.25, i + 0.25,
                  colors='black', linewidth=2, linestyle = '--',
                  label='Średnia' if i == 0 else '')

    plt.title(f'Analiza ekspresji genu: {gene_name}',
              fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Poziom ekspresji', fontsize=12)
    plt.xlabel('Etykieta', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_panel_boxplots(X, y, gene_map, gene_coef=None, group_order=None, showfliers=False, zscore=True, save_path=None):
    genes = [g for g in gene_map if g in X.columns]
    gene_order = [gene_map[g] for g in genes]
    sub = X[genes].astype(float)
    if zscore:
        z = (sub - sub.mean()) / sub.std(ddof=0).replace(0, 1)
    else:
        z = sub.copy()
    z.columns = gene_order
    z = z.reset_index(drop=True)
    z['__grupa__'] = np.asarray(y)
    long = z.melt(id_vars='__grupa__', var_name='gen', value_name='z')

    if group_order is None:
        present = set(np.asarray(y))
        group_order = [g for g in [HEALTHY, DISEASE, CANCER] if g in present]

    display_order = gene_order
    if gene_coef is not None:
        display_order = sorted(gene_order, key=lambda s: gene_coef.get(s, float('inf')))

    plt.figure(figsize=(max(10, len(genes) * 1.1), 6))
    ax = sns.boxplot(data=long, x='gen', y='z', hue='__grupa__',
                     order=display_order, hue_order=group_order,
                     width=0.7, fliersize=2, showfliers=showfliers)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('Gen', fontsize=12)
    ax.set_ylabel('Ekspresja (z-score)' if zscore else 'Ekspresja', fontsize=12)
    ax.set_title('Ekspresja panelu biomarkerów w grupach',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Grupa', loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curve(X, y, title):
    fpr, tpr, thresholds = roc_curve(y, X)
    auc = roc_auc_score(y, X)
    print(f"ROC AUC = {auc:.3f}")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate (1 - specificity)")
    plt.ylabel("True Positive Rate (sensitivity)")
    plt.title("ROC curve for " + title)
    plt.legend()
    plt.grid(alpha=0.1)
    plt.show()


def plot_split_balance(splits: dict):
    COLORS = {'Train': '#6366f1', 'Test': '#22d3ee', 'Valid': '#f59e0b'}
    LABELS = {'Train': 'Treningowy', 'Test': 'Testowy', 'Valid': 'Walidacyjny'}
    names = list(splits.keys())

    all_y   = splits[names[0]][0]
    class_vals = sorted(all_y.unique())
    sex_vals   = sorted(splits[names[0]][1].unique())
    stage_all = pd.concat([splits[s][3].dropna() for s in names])
    stage_vals = sorted(stage_all.unique())

    class_counts = {s: splits[s][0].value_counts(normalize=True) for s in names}
    sex_counts   = {s: splits[s][1].value_counts(normalize=True) for s in names}
    age_data     = {s: splits[s][2].values for s in names}
    stage_counts = {s: splits[s][3].dropna().value_counts(normalize=True) for s in names}

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=["Rozkład klas", "Rozkład płci", "Rozkład wieku", "Rozkład stadium"],
        horizontal_spacing=0.12,
    )

    for s in names:
        fig.add_trace(go.Bar(
            name=LABELS.get(s, s), x=class_vals,
            y=[class_counts[s].get(c, 0) for c in class_vals],
            marker_color=COLORS[s], showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            name=LABELS.get(s, s), x=sex_vals,
            y=[sex_counts[s].get(sv, 0) for sv in sex_vals],
            marker_color=COLORS[s], showlegend=False,
        ), row=1, col=2)

        fig.add_trace(go.Box(
            name=LABELS.get(s, s), y=age_data[s],
            marker_color=COLORS[s], boxmean=True, showlegend=False,
        ), row=1, col=3)

        fig.add_trace(go.Bar(
            name=LABELS.get(s, s), x=stage_vals,
            y=[stage_counts[s].get(sv, 0) for sv in stage_vals],
            marker_color=COLORS[s], showlegend=False,
        ), row=1, col=4)

    fig.update_layout(
        barmode='group',
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        title={"text": ""},
        legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='center', x=0.5),
    )
    fig.update_yaxes(title_text="Udział", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Udział", tickformat=".0%", row=1, col=2)
    fig.update_yaxes(title_text="Wiek (lata)", row=1, col=3)
    fig.update_yaxes(title_text="Udział", tickformat=".0%", row=1, col=4)

    save_path = "split_balance.png"
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.write_image(save_path, scale=2)

    fig.show()


def _to_rgba(color: str, alpha: float = 0.7) -> str:
    c = color.strip()
    if c.startswith('#'):
        c = c.lstrip('#')
        if len(c) == 3:
            r, g, b = (int(ch * 2, 16) for ch in c)
        else:
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'
    if c.startswith('rgb('):
        body = c[c.index('(') + 1: c.rindex(')')]
        return f'rgba({body},{alpha})'
    return c


def plot_group_overview(splits: dict, colors: dict = None, title: str = None,
                        libsize_data: dict = None,
                        width: int = None, height: int = 420,
                        save_path: str = None, white_bg: bool = True,
                        show: bool = True):

    names = list(splits.keys())

    if colors is None:
        _palette = pc.qualitative.Plotly
        colors = {s: _palette[i % len(_palette)] for i, s in enumerate(names)}

    all_sex = pd.concat([splits[s][1] for s in names])
    sex_vals   = sorted(all_sex.unique())

    sex_counts   = {s: splits[s][1].value_counts(normalize=True) for s in names}
    sex_raw      = {s: splits[s][1].value_counts()               for s in names}
    age_data     = {s: splits[s][2].values                       for s in names}

    has_libsize = libsize_data is not None
    n_cols = 3 if has_libsize else 2
    subplot_titles = ["Rozkład płci", "Rozkład wieku"]
    if has_libsize:
        subplot_titles.append("Głębokość sekwencjonowania")

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.10,
    )

    for s in names:
        color = colors[s]
        color_a = _to_rgba(color, 0.7)
        y_sex_pct = [sex_counts[s].get(sv, 0) for sv in sex_vals]
        y_sex_cnt = [sex_raw[s].get(sv, 0)    for sv in sex_vals]
        labels    = [f"n={cnt}<br>({pct:.0%})"
                     for cnt, pct in zip(y_sex_cnt, y_sex_pct)]
        fig.add_trace(go.Bar(
            name=s, x=sex_vals,
            y=y_sex_pct,
            text=labels,
            textposition='outside',
            marker_color=color_a,
            legendgroup=s,
            showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Box(
            name=s, y=age_data[s],
            line=dict(color=color),
            fillcolor=color_a,
            marker_color=color,
            boxmean=True,
            legendgroup=s,
            showlegend=False,
        ), row=1, col=2)

        if has_libsize and s in libsize_data:
            fig.add_trace(go.Box(
                name=s, y=libsize_data[s],
                line=dict(color=color),
                fillcolor=color_a,
                marker_color=color,
                boxmean=True,
                legendgroup=s,
                showlegend=False,
            ), row=1, col=3)

    if width is None:
        width = 1100 if has_libsize else 850
    layout_kwargs = dict(
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.10,
                    xanchor='center', x=0.5,
                    font=dict(size=13)),
        margin=dict(t=70, l=55, r=20, b=50),
        width=width,
        height=height,
    )
    if title is not None:
        layout_kwargs['title'] = {"text": title}
    if white_bg:
        layout_kwargs.update(
            paper_bgcolor='white', plot_bgcolor='white',
            font=dict(color='black'),
            title_font=dict(color='black'),
        )

    fig.update_layout(**layout_kwargs)
    fig.update_yaxes(title_text="Udział",      tickformat=".0%", title_standoff=4, row=1, col=1)
    fig.update_yaxes(title_text="Wiek (lata)", title_standoff=4, row=1, col=2)
    if has_libsize:
        fig.update_yaxes(title_text="log10(Lib.size)", title_standoff=4, row=1, col=3)
    if white_bg:
        axis_style = dict(
            showline=True, linecolor='black', linewidth=1,
            tickcolor='black', tickfont=dict(color='black'),
            title_font=dict(color='black'),
            gridcolor='lightgray',
            zerolinecolor='lightgray',
        )
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
        for ann in fig.layout.annotations:
            ann.font = dict(color='black', size=ann.font.size or 14)
        for tr in fig.data:
            if isinstance(tr, go.Bar):
                tr.textfont = dict(color='black')

    if save_path is not None:
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_path.suffix.lower()
        if suffix == '.html':
            fig.write_html(str(save_path))
        else:
            fig.write_image(str(save_path), scale=2)
        print(f"[OK] zapisano: {save_path.resolve()}")

    if show:
        fig.show()
    return fig
