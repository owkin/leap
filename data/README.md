# Data

Directory for storing the data required to run models.

> [!NOTE]
> Added data files are gitignored. Only this README and the pre-existing lists of genes, perturbations and tissues are tracked.

## Existing Files

The files traked here are lists of genes, perturbation and tissues that are useful to reproduce the different tasks described in the paper.

## Files to download

To run experiments, please download the following files from [DepMap](https://depmap.org/portal/data_page/?tab=allData):
- CRISPRGeneDependency.csv
- OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
- Model.csv

And [this file](https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2025.1.Hs/c2.all.v2025.1.Hs.json) from MsigDB. This file is the JSON bundle associated with the GCP (chemical and genetic perturbations) gene set.

Save all those files in this directory.
