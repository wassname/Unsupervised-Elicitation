# %%
from io import StringIO
import requests
import pandas as pd
from datasets import Dataset


url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/refs/heads/main/TruthfulQA.csv"
response = requests.get(url)
df = pd.read_csv(StringIO(response.text))
df = df.reset_index(names="question_id")
df

# %%
a = df[['question_id', 'Type', 'Category', 'Question', 'Best Answer', 'Source']].rename(columns={'Best Answer': 'Answer'})
a['label'] = 1
b = df[['question_id', 'Type', 'Category', 'Question', 'Best Incorrect Answer', 'Source']].rename(columns={'Best Incorrect Answer': 'Answer'})
b['label'] = 0
df_binary = pd.concat([a,b]).sort_values('question_id')
df_binary


repo_id = "wassname/truthful_qa_v2"

datasets = {
    "default": df,
    "binary": df_binary,
}
for name, ddf in datasets.items():
    ds = Dataset.from_pandas(ddf)
    ds.push_to_hub(
        repo_id=repo_id,
        config_name=name,
        split="validation",
    )
