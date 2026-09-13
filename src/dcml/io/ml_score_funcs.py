
import json
import pathlib

dcml_package_path = pathlib.Path(__file__).parents[1]


def load_ml_score_to_int(ml_score_to_int: dict = None):
    """If ml_score_to_int is not given, load the 'ml_score_to_int.json' file that contains all relevant
    ml_score_xyz definitions and their corresponding target integer.
    If ml_score_to_int

    Returns
    -------
    ml_score_names : dict
        Dict of `ml_score_xyz` names from the `ml_score_to_int.json` file or from the input argument.

    """
    if not ml_score_to_int:
        with open(dcml_package_path / "definitions" / "ml_score_to_int.json",
                  "r") as ff:
            ml_score_names = json.load(ff)
        print("--> Using definitions/ml_score_to_int.json file")
    else:
        ml_score_names = ml_score_to_int
    return ml_score_names
