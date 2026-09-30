"""Boxplots of biomarker panel expression across the three sample groups.

Methodology:

1. Gene panel. The genes selected by 3_forward_selection.py are read from GENES_CSV.
   Gene symbols are taken from BIOMARKER_PANEL where available; other genes are
   labelled with their Ensembl ID.

2. Gene order. Genes are ordered along the x axis by their mean signed coefficient
   from stability selection (STABLE_GENES_CSV), so down-regulated genes appear
   before up-regulated ones.

3. Plot. The full dataset is loaded without splitting or merging labels, and one
   grouped boxplot of raw expression per panel gene is drawn, split by the original
   three groups (asymptomatic controls, pancreatic diseases, pancreatic cancer).
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import pandas as pd

from utilz.Dataset import load_dataset
from utilz.constans import BIOMARKER_PANEL
from utilz.helpers import plot_panel_boxplots

OUT_DIR = Path(__file__).stem
Path(OUT_DIR).mkdir(exist_ok=True)

meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"
GENES_CSV        = "3_forward_selection/forward_selection_genes.csv"
STABLE_GENES_CSV = "2_stable_selection_with_enet/stability_all_stable_genes.csv"

OUT_PNG_BOX = f"{OUT_DIR}/biomarker_panel_box.png"


panel_genes = pd.read_csv(GENES_CSV)['gene'].tolist()
gene_map = {g: BIOMARKER_PANEL.get(g, g) for g in panel_genes}

signed_coef = pd.read_csv(STABLE_GENES_CSV).set_index('gene')['mean_signed_coef']
gene_coef = {gene_map[g]: signed_coef[g] for g in panel_genes if g in signed_coef.index}

ds = load_dataset(data_path, meta_path, label_col="Group")
plot_panel_boxplots(
    ds.X, ds.y.values, gene_map,
    gene_coef=gene_coef,
    zscore=False,
    save_path=OUT_PNG_BOX,
)
