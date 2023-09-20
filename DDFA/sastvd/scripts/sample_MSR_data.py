import pandas as pd

import sastvd as svd

df = pd.read_csv(svd.external_dir() / "MSR_data_cleaned.csv")
seed = 0
sample_n_from_each_class = 100
cut_df = pd.concat(
    (
        df[df.vul == 0].sample(sample_n_from_each_class, random_state=seed),
        df[df.vul == 1].sample(sample_n_from_each_class, random_state=seed),
    )
)
print(cut_df.columns)
print(cut_df)
cut_df.to_csv(svd.external_dir() / "MSR_data_cleaned_SAMPLE.csv")
