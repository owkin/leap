"""Define trainer parameters (data and split configs)."""

from typing import Literal


# Data config parameters
SOURCE_DOMAIN_STUDIES: dict[str, str | list[str]] = {
    "1": "DepMap_23Q4",
    "2": ["GDSC_2020_v2-8_2", "GDSC_2020_v1-8_2", "CCLE_2015", "CTRPv2_2015"],
    "3": "DepMap_23Q4",
    "4": ["GDSC_2020_v2-8_2", "GDSC_2020_v1-8_2", "CCLE_2015", "CTRPv2_2015"],
    "5": ["GDSC_2020_v2-8_2", "GDSC_2020_v1-8_2", "CCLE_2015", "CTRPv2_2015"],
}

SOURCE_DOMAIN_LABEL: dict[str, str] = {
    "1": "gene_dependency",
    "2": "AAC",
    "3": "gene_dependency",
    "4": "AAC",
    "5": "AAC",
}

LIST_OF_PERTURBATIONS: dict[str, str] = {
    "1": "perturbations_task_1",
    "2": "perturbations_task_2",
    "3": "perturbations_task_3",
    "4": "perturbations_task_4",
    "5": "perturbations_task_5",
}

FILTER_AVAILABLE_FINGERPRINTS: dict[str, bool] = {
    "1": True,
    "2": True,
    "3": True,
    "4": True,
    "5": False,
}

# IMPORTANT: in LEAP we actually use combat_depmap_gdsc_pdx (using combat for batch effect correction)
NORMALIZATION: dict[str, str] = {
    "1": "tpm",
    "2": "tpm",
    "3": "tpm",
    "4": "tpm",
    "5": "tpm",
}

# IMPORTANT: in LEAP we actually use most_variant_genes_intersection_depmap_gdsc_pdx
LIST_OF_GENES: dict[str, str] = {
    "1": "most_variant_genes",
    "2": "most_variant_genes",
    "3": "most_variant_genes",
    "4": "most_variant_genes",
    "5": "most_variant_genes",
}

TARGET_DOMAIN_STUDIES: dict[str, str | list[str] | None] = {
    "1": None,
    "2": None,
    "3": None,
    "4": None,
    "5": ["PDXE"],
}

TARGET_DOMAIN_LABEL: dict[str, str | None] = {
    "1": None,
    "2": None,
    "3": None,
    "4": None,
    "5": "minus_min_delta_tumor_volume",
}

# Split config parameters
TEST_SET_TYPE: dict[str, Literal["sample", "perturbation", "tissue", "transfer_learning"]] = {
    "1": "sample",
    "2": "sample",
    "3": "transfer_learning",
    "4": "transfer_learning",
    "5": "transfer_learning",
}

# Dimensionality of the fingerprint rpz model
FGPS_DIM: dict[str, int] = {
    "1": 256,
    "2": 256,
    "3": 256,
    "4": 256,
    "5": 10,  # not used
}
