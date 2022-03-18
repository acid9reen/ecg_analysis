# ECG analysis with pytorch
Package with some models and preprocessing tools for PTB-XL

## Prerequisites

### Data

[Download PTB_XL dataset](https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip)

Unzip archive and move the files into data/raw folder in such a way that you get the following structure:

```
data
└───raw
    │   .gitkeep
    │   example_physionet.py
    │   LICENSE.txt
    │   ptbxl_database.csv
    │   RECORDS
    │   scp_statements.csv
    │   SHA256SUMS.txt
    │
    ├───records100
    └───records500
```

### Dependencies

Install dependencies via following command:
```bash
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### To run notebooks
Install local package:
```bash
pip install .
```
