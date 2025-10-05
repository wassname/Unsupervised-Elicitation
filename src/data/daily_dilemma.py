from datasets import load_dataset

def load_daily_dilemma():
    dataset = load_dataset("kellycyy/daily_dilemmas", split="test")
    
    data = []
    for item in dataset:
        # Unbiased prompt - just the situation and action
        prompt = f"Situation: {item['dilemma_situation']}\nAction: {item['action']}"
        
        # Map to A/B (unrelated group names)
        label = 1 if item['action_type'] == 'to_do' else 0
        consistency_key = 'A' if item['action_type'] == 'to_do' else 'B'
        
        example = {
            'uid': item['idx'],
            'prompt': prompt,
            'vanilla_label': label,
            'consistency_id': item['dilemma_idx'],
            'consistency_key': consistency_key
        }
        data.append(example)
    
    return data
