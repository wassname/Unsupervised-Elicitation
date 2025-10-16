def is_consistent(group_uids, demos):
    """
    Check if labels in a consistency group satisfy the constraint.
    Returns True if consistent (or insufficient labels to judge).
    
    Generic version: same consistency_key -> same label, different keys -> opposite labels
    """
    if len(group_uids) < 2:
        return True
    
    # Get labeled items only
    labeled = [(uid, demos[uid]['label'], demos[uid]['consistency_key']) 
               for uid in group_uids if demos[uid].get('label', None) is not None]
    
    if len(labeled) < 2:
        return True  # Can't be inconsistent with <2 labels
    
    # Group by consistency_key
    key_labels = {}
    for uid, label, key in labeled:
        if key not in key_labels:
            key_labels[key] = []
        key_labels[key].append(label)
    
    # Rule 1: Same key must have same label
    for key, labels in key_labels.items():
        if len(set(labels)) > 1:
            return False  # Same key, different labels
    
    # Rule 2: Different keys must have opposite labels
    if len(key_labels) > 1:
        unique_labels_per_key = {k: list(set(v))[0] for k, v in key_labels.items()}
        if len(set(unique_labels_per_key.values())) < len(key_labels):
            return False  # Not all different
    
    return True
