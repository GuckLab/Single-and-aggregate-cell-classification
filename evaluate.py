import argparse

from dotenv import load_dotenv
from src.deepclassifier.evaluate_utils import evaluate_models, evaluate_models_on_gmm


# load env variables
load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', type=str, help='Input folder path')
    parser.add_argument('--path_out', type=str, default="", help='Output folder path for predictions on test set')
    parser.add_argument('--path_in_gmm', type=str, default="", help='Input folder path for GMM test data')
    parser.add_argument('--path_out_gmm_pred', type=str, default="", help='Output folder path for predictions on GMM set')
    parser.add_argument('--mlflow_gmm_folder_name', type=str, default="evaluation_on_gmm_measurements",
                        help='folder name within mlflow where gmm evaluation results will be stored')
    parser.add_argument('--model', type=str, help='model run_id')
    parser.add_argument('--batch_size', action='store_true', default=16)
    parser.add_argument('--rm_pred', action='store_true', default=False, help='removes folder with predictions after evaluation')

    args = parser.parse_args()



    # evaluate_model(args.model, args.path_in, path_out=args.path_out, model_name = "best_model_f1",
    #                batch_size=args.batch_size)
    # evaluate_model(args.model, args.path_in, model_name="best_model_f1", batch_size=args.batch_size)

    # evaluate_models(args.model, args.path_in, model_names=["best_model_f1",
    #                                                        "best_model_accuracy"], batch_size=args.batch_size)

    #evaluate_models(args.model, args.path_in, model_names=["best_model_f1"], batch_size=args.batch_size)

    evaluate_models(args.model, args.path_in, model_names=["best_model_bal_acc"], batch_size=args.batch_size)


    if args.path_in_gmm:

        # model evaluation on GMM data
        print("starting evaluation on GMM test dataset")

        evaluate_models_on_gmm(args.model, args.path_in_gmm, path_out_gmm_pred=args.path_out_gmm_pred,
                               model_names=["best_model_bal_acc"], remove_predictions=args.rm_pred,
                               mlflow_folder_name=args.mlflow_gmm_folder_name)