"""Boxplots of biomarker panel expression across the three sample groups.

Methodology: loads the full dataset without splitting or merging labels, and draws one
grouped boxplot of raw expression per panel gene, split by the original three groups
(asymptomatic controls, benign pancreatic diseases, pancreatic cancer). Genes are
ordered along the x axis by their mean signed coefficient from stability selection, so
down-regulated genes appear before up-regulated ones.
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

from utilz.Dataset import load_dataset
from utilz.constans import BIOMARKER_PANEL
from utilz.helpers import plot_panel_boxplots

OUT_DIR = Path(__file__).stem
Path(OUT_DIR).mkdir(exist_ok=True)

meta_path = r"../data/samples_pancreatic.xlsx"
data_path = r"../data/counts_pancreatic.csv"

OUT_PNG_BOX = f"{OUT_DIR}/biomarker_panel_box.png"

PANEL_COEF = {
    'PLD4':    -0.128,
    'ITGB3BP': -0.186,
    'FUT8':     0.222,
    'ACVR1':   -0.071,
    'ZDHHC4':   0.104,
    'PRELID2':  0.125,
    'TMEM63B':  0.202,
    'FABP4':    0.199,
    'DENND6B':  0.134,
    'DDX11L17':-0.130,
    'H2BC15':  -0.092,
}


ds = load_dataset(data_path, meta_path, label_col="Group")
plot_panel_boxplots(
    ds.X, ds.y.values, BIOMARKER_PANEL,
    gene_coef=PANEL_COEF,
    zscore=False,
    save_path=OUT_PNG_BOX,
)
