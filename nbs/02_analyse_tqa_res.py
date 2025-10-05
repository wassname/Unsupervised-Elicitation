# %%
import numpy as np
import pandas as pd
from pathlib import Path
import json

data_dir = Path('../outputs/icm/truthfulqa/')
print('Config', (data_dir / 'icm_config.json').open().read())
df_res = pd.read_parquet(data_dir / 'icm_final_labels.parquet')
reasons = open(data_dir / 'reasoning.txt').read()
df_res = df_res.dropna(subset=['label'])
print(f'Label counts: {df_res.shape[0]}')
df_res

# also show top disagreements

# %%
# from src.data.truthfulqa import load_truthfulqa
# data = load_truthfulqa()

acc = (df_res['label'] == df_res['vanilla_label']).mean()
print(f'Accuracy against vanilla: {acc:.3f}')

# df_res = df_res.merge(data, left_on="uid", right_on='idx')



# %%
# summarise the reasoning log
from openrouter_wrapper.retry import openrouter_request_sync, ProviderError
import os
from dotenv import load_dotenv
load_dotenv()

import nest_asyncio
nest_asyncio.apply()

response = openrouter_request_sync(
    {
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "messages": [
            {"role": "user", "content": f"The following text is a log of the reasons given while labelling a dataset. The dataset has no context so the latter reasoning it more weighty than the early. Read the reasoning and summarise the two categories that the LLM ended up labelling with. Also how did it evolve, and was it consistent near the end of the log?:\n\n{reasons}"}
        ],
        "timeout": 120
    }
)
s = response['choices'][0]['message']['content']
print(s)

