from datasets import load_dataset
import pandas as pd
from collections import defaultdict
import ast


def convert_values_to_list(x):
    # turn into list
    s = x["values_aggregated"]
    v = ast.literal_eval(s)
    return {"values_aggregated": v}

def load_daily_dilemma(label_col = "label_Virtue/Truthfulness"):
    dataset = load_dataset("kellycyy/daily_dilemmas", split="test")
    dataset = dataset.map(convert_values_to_list)
    df_labels = load_labels(dataset).rename(columns=lambda x: f"label_{x}" if x != "dilemma_idx" else x)
    df = dataset.to_pandas()
    df = df.merge(df_labels, on='dilemma_idx')
    cols_label = [c for c in df_labels.columns if c.startswith('label_')]
    # flip labels on not_to_do
    df.loc[df['action_type'] == 'not_to_do', cols_label] *= -1
    dataset = df.to_dict(orient='records')
    
    
    
    data = []
    for item in dataset:
        # Unbiased prompt - just the situation and action
        prompt = f"Situation: {item['dilemma_situation']}\nAction: {item['action']}"
        
        # Map to A/B (unrelated group names)
        consistency_key = 'A' if item['action_type'] == 'to_do' else 'B'
        
        example = {
            'uid': item['idx'],
            'prompt': prompt,
            'vanilla_label': item[label_col],
            'consistency_id': item['dilemma_idx'],
            'consistency_key': consistency_key
        }
        data.append(example)

    
    return data



def load_labels(dd_dataset):
    ds_values = load_dataset("kellycyy/daily_dilemmas", split="test", name="Values")

    # moral tags
    moral_frameworks = ["WVS", "MFT", "Virtue", "Emotion", "Maslow"]

    value2framework_dicts = {}
    for framework in moral_frameworks:
        df_values = ds_values.to_pandas()[["value", framework]].dropna()
        value2framework_dict = df_values.set_index("value")[framework].to_dict()
        value2framework_dict = {k: f"{framework}/{v}" for k, v in value2framework_dict.items()}
        value2framework_dicts[framework] = value2framework_dict


    # make labels
    df_dilemma = dd_dataset.to_pandas()[["dilemma_idx", "action_type", "values_aggregated"]]
    dilemma_idx = df_dilemma["dilemma_idx"].unique()

    labels = []
    for d_idx in dilemma_idx:
        pos_values = (
            df_dilemma.query('dilemma_idx == @d_idx and action_type == "to_do"')["values_aggregated"].iloc[0].tolist()
        )
        neg_values = (
            df_dilemma.query('dilemma_idx == @d_idx and action_type == "not_to_do"')["values_aggregated"].iloc[0].tolist()
        )

        label = defaultdict(int)

        for framework in value2framework_dicts:
            value2framework_dict = value2framework_dicts[framework]
            virtues = sorted(set(value2framework_dict.values()))

            pos_virtues = [value2framework_dict[k] for k in pos_values if k in value2framework_dict]
            neg_virtues = [value2framework_dict[k] for k in neg_values if k in value2framework_dict]

            for p in pos_virtues:
                label[p] += 1
            for n in neg_virtues:
                label[n] -= 1

        labels.append(dict(dilemma_idx=d_idx, **label))



    df_labels = pd.DataFrame(labels).set_index("dilemma_idx")
    assert df_labels.index.is_unique
    return df_labels

