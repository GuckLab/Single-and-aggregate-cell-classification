import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

import argparse
from src.dcml.preprocessing.io import apply_prediction_to_dir
from src.deepclassifier.get_device import get_available_device_num_workers
from src.dcml.utils.params import get_saved_params
from src.dcml.utils.data import create_single_dataset
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', type=str, required=True)
    parser.add_argument('--path_out', type=str, required=True)
    parser.add_argument('--model', type=str, help='model run_id', required=True)

    args = parser.parse_args()

    path_in = Path(args.path_in).resolve()
    path_out = Path(args.path_out).resolve()

    if not path_in.is_dir():
        raise NotADirectoryError(f"Input path does not exist or is not a directory: {path_in}")

    if path_out.exists() and not path_out.is_dir():
        raise NotADirectoryError(f"Output path exists and is not a directory: {path_out}")

    if path_in == path_out:
        raise ValueError("--path_in and --path_out must be different directories")

    try:
        path_out.relative_to(path_in)
    except ValueError:
        pass
    else:
        raise ValueError("--path_out must not be inside --path_in, otherwise reruns can reprocess generated prediction files")

    device, num_workers = get_available_device_num_workers()
    measurements_params = get_saved_params(args.model)
    train_dataset_std = measurements_params['create_dataset']['std']

    # compute dataset statistics
    abs_path_files = sorted(el for el in path_in.rglob("*.rtdc") if el.is_file())
    if not abs_path_files:
        raise FileNotFoundError(f"No .rtdc files found under input directory: {path_in}")

    dataset = create_single_dataset(
        hdf5_path=str(abs_path_files[0]), # relying on the first .rtdc file to compute the mean and std of the dataset
        required_data={"image": "image"},
        crop_size=measurements_params["create_dataset"]["crop_size"],
        correct_background=False,
        mean=None,  # forces computation of mean
        std=None,  # forces computation of std
        p99_compute=True, # forces computation of p99
        MTL=measurements_params["create_trainer"]["architecture"]["type"] == 'multitask'
    )

    apply_prediction_to_dir(path_in=path_in,
                            path_out=path_out,
                            model_path=args.model,
                            device=device,
                            num_workers=num_workers,
                            with_basin=False,
                            cleanup_predictions=False,
                            model_name="best_model_bal_acc",
                            mean = dataset.mean,  # train_dataset_mean,
                            std = train_dataset_std,  # dataset.std,
                            brightness_factor = None
    )

if __name__ == "__main__":
    main()
