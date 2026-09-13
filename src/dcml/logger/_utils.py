
def flatten_dict(d, parent_key='', sep='_'):
    """
    Flattens a nested dictionary into a flat dictionary.

    Args:
        d (dict): The dictionary to be flattened.
        parent_key (str, optional): The parent key prefix. Defaults empty.
        sep (str, optional): The separator between keys. Defaults to '_'.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
