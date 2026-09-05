
import re
import json
import pathlib
from glob import glob
import dclab
import numpy as np

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


# def match_annotation_files(rtdc_dir, recursive=True):
#     """Regex match all valid annotation files.
#     Matching the pattern '<some-prefix>ml_score_xyz.rtdc'.
#
#     Parameters
#     ----------
#     rtdc_dir : str or pathlib.Path
#         Location of the annotation files. Can be in nested folders.
#     recursive : bool
#         Set to True if you want to search in all folders under the given `path`
#
#     Returns
#     -------
#     annotation_files : list
#         List of annotation file paths
#
#     """
#     rtdc_dir = pathlib.Path(rtdc_dir)
#
#     r = re.compile('.*ml_score_[a-zA-Z0-9]{3}([_][0-9])?\.rtdc')  # noqa: W605
#     if recursive:
#         rtdc_dir = rtdc_dir / "**/*.rtdc"
#     else:
#         rtdc_dir = rtdc_dir / "*.rtdc"
#     annotation_files = glob(str(rtdc_dir), recursive=recursive)
#     annotation_files = [el for el in annotation_files if r.match(el)]
#     annotation_files = [pathlib.Path(x) for x in annotation_files]
#     print(f"The following annotation files have been found:"
#           f"\n{annotation_files}")
#     return annotation_files


# def populate_ctt_array(tmp_path, annotation_files, ml_score_names=None):
#     """Populate the ctt array and save it in a h5 file.
#
#     Parameters
#     ----------
#     tmp_path : str or pathlib.Path
#         Location of the temporary file.
#     annotation_files : list
#         list of annotation file paths
#     ml_score_names : dict, optional
#         A dictionary of ml_score_names. If set to None, will be loaded from
#         ``dcml/definitions/ml_score_to_int.json``
#     """
#
#     if ml_score_names is None:
#         ml_score_names = load_ml_score_to_int()
#
#     writer = dclab.rtdc_dataset.RTDCWriter(tmp_path)
#     ds_len = len(writer.h5file["events"]["area_um"])
#     ds_index_online = writer.h5file["events"]["index_online"]
#     ctt_array = np.zeros((ds_len,))
#
#     for anno_file in annotation_files:
#         print(f"Processing {anno_file}")
#         with dclab.new_dataset(anno_file) as ds:
#             for msn in ml_score_names:
#                 if msn in ds:
#                     # get array of index_online
#                     index_online = ds["index_online"]
#                     # get subarray of index_online, where msn==1
#                     index_online_msn = index_online[np.where(ds[msn] == 1)]
#                     index_online_msn = index_online_msn.astype(int)
#                     index_msn = np.where(
#                         np.isin(ds_index_online, index_online_msn))
#                     ctt_array[index_msn] = ml_score_names[msn]
#
#     # use RTDCWriter to write "ml_score_ctt" to original dataset in hdf5-format
#     print("Writing the ml_scores to file ...")
#     writer.store_feature("ml_score_ctt", ctt_array)
#     writer.h5file.close()
