from datasets import load_dataset
from . import is_consistent

def load_truthfulqa():

    # Load larger HF dataset: Yik/truthfulQA-bool
    dataset = load_dataset("wassname/truthful_qa_v2", split="validation", name="binary")
    # semantic_anchor = "0 is false, 1 is true."
    data = []
    group_id = 0
    groups = {}
    for idx, item in enumerate(dataset):
        claim = f"Question: {item['Question']}\nClaim: {item['Answer']}"
        label = item['label']
        consistency_id = item['question_id']
        
        
        example = {
            'uid': idx,
            'prompt': claim,
            'vanilla_label': label,
            'consistency_id': item['question_id'], # this is e.g. the question_id
            'consistency_key': 'A' if label == 1 else 'B' # this are the true groups, watch out you don't leak or meta-use the labels
        }
        data.append(example)

    return data
