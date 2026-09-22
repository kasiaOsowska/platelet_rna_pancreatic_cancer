"""Shared constants: group labels, reference gene identifiers and the biomarker panel.

Group labels match the Group column of the sample metadata. Gene identifiers are
Ensembl IDs; they cover genes known from tissue studies of pancreatic cancer, genes
reported by the platelet PLA2Sig signature, and the genes with the smallest p-values
found in this dataset. BIOMARKER_PANEL maps the Ensembl IDs of the final panel to gene
symbols and its key order defines the order used in plots.
"""

HEALTHY = "Asymptomatic controls"
DISEASE = "Pancreatic diseases"
CANCER = "Pancreatic cancer"

KRAS = "ENSG00000133703"
TP53 = "ENSG00000141510"
SMAD4 = "ENSG00000141646"

BCAP31 = "ENSG00000185825"
ARL2 = "ENSG00000213465"
CFL1 = "ENSG00000172757"
MYL9 = "ENSG00000101335"

SCN1B = "ENSG00000105711"
MAGOHB = "ENSG00000111196"

BIOMARKER_PANEL = {
    "ENSG00000166428": "PLD4",
    "ENSG00000142856": "ITGB3BP",
    "ENSG00000160145": "KALRN",
    "ENSG00000186314": "PRELID2",
    "ENSG00000137216": "TMEM63B",
    "ENSG00000103528": "SYT17",
    "ENSG00000233822": "H2BC15",
    "ENSG00000170323": "FABP4",
    "ENSG00000033170": "FUT8",
    "ENSG00000115170": "ACVR1",
    "ENSG00000069535": "MAOB",
    "ENSG00000001036": "FUCA2",
}
