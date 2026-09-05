import matplotlib.pyplot as plt
import numpy as np


# def get_metrics_table(scores, class_label_dict):
#     fig, ax = plt.subplots(figsize=(12, 4))
#     fig.patch.set_visible(False)
#     ax.axis = ('off')
#     ax.axis = ('tight')
#     row_labels = ["Precision", "Recall", "F1-Score"]
#     num_classes = len(class_label_dict)
#     col_labels = [class_label_dict[k]
#                   for k in range(num_classes)]
#     table = plt.table(np.round(scores, 3),
#                       rowLabels=row_labels, colLabels=col_labels,
#                       loc='center')
#     table.set_fontsize(14)
#     fig.tight_layout()
#     return fig


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
