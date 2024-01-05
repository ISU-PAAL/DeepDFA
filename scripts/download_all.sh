#!/bin/bash

# Raw Big-Vul dataset
gdown 'https://drive.google.com/uc?id=1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X' -O DDFA/storage/external/MSR_data_cleaned.zip
unzip DDFA/storage/external/MSR_data_cleaned.zip -d DDFA/storage/external/

# LineVul version of Big-Vul dataset
gdown 'https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V' -O LineVul/data/MSR/test.csv
gdown 'https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw' -O LineVul/data/MSR/train.csv
gdown 'https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ' -O LineVul/data/MSR/val.csv

# DeepDFA preprocessed data
wget 'https://figshare.com/ndownloader/files/43917390' -O preprocessed_data.zip
unzip preprocessed_data.zip

# DeepDFA CFGs
wget 'https://figshare.com/ndownloader/files/43916550' -O before.zip
unzip before.zip -d DDFA/storage/processed/bigvul
