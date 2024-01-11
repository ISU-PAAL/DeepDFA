#!/bin/bash
set -e

# Raw Big-Vul dataset
curl -Lo MSR_data_cleaned.zip 'https://figshare.com/ndownloader/files/43990908'
unzip MSR_data_cleaned.zip -d DDFA/storage/external/

# LineVul version of Big-Vul dataset
curl -Lo MSR_LineVul.zip 'https://figshare.com/ndownloader/files/43991823'
unzip MSR_LineVul.zip -d LineVul/data/MSR

# DeepDFA preprocessed data
curl -Lo preprocessed_data.zip 'https://figshare.com/ndownloader/files/43991910'
unzip preprocessed_data.zip

# DeepDFA CFGs
curl -Lo before.zip 'https://figshare.com/ndownloader/files/43916550'
unzip before.zip -d DDFA/storage/processed/bigvul
