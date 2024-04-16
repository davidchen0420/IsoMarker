# IsoMarker

## Main
IsoMarker is a Python3 software package designed to identify consensus gene expression markers using Differential Expression (PyDESeq2), AUC (WilcoxAUC), and machine learning explainability (SHAP).

## Description
Identifying gene expression markers of cell states is classically computed using differential expression approaches of transcriptomic data. However, differential expression often leads to large gene lists (>100s of genes) that may be redundant and introduce noise when collectively used to identify specific cell states.

To address the problem of nominating select, highly predictive, and minimal cell state marker gene lists, IsoMarker aims to identify differentially expressed, specific, and predictive gene lists for specific cell states using a consensus approach. Beyond cell state marker identification, IsoMarker can be broadly applied to tasks where nomination of specific, minimal markers from high dimensional data is required. The rich selection of built-in analytical and visualization functions makes IsoMarker an end-to-end tool for marker identification.

## Installation
```sh
pip install isomarker
```

## Tutorial
Coming soon...

## File Structure
<pre>
├── LICENSE       
├── README.md
├── isomarker.py                                                
├── setup.py

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.Please make sure to update tests as appropriate.

## Reference
Coming soon...
