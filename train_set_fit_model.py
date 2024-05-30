import argparse
import os
import sys
from datetime import datetime
import pandas as pd
from datasets import Dataset
from datasets import DatasetDict
from setfit import SetFitModel, SetFitTrainer
from datasets import load_metric
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from sentence_transformers.losses import CosineSimilarityLoss
import yaml
from training_data_handler import TrainingDataHandler
import re

with open('setfit_config.yml', 'r') as f:
    config = yaml.safe_load(f)

tdh = TrainingDataHandler(config["random_state"])

# Load a SetFit model from Hub
MODEL_NAME_OR_PATH = config['model_name_or_path']


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_output_path(config):
    PATH_TO_TEMP_MODELS = os.path.join(os.path.dirname(__file__), config['output_path_to_temporary_models'])

    if not os.path.isdir(PATH_TO_TEMP_MODELS):
        os.makedirs(PATH_TO_TEMP_MODELS)

    return PATH_TO_TEMP_MODELS


def model_init(params):
    params = params or {}
    max_iter = params.get("max_iter", 100)
    solver = params.get("solver", "liblinear")
    params = {
        "head_params": {
            "max_iter": max_iter,
            "solver": solver,
        }
    }
    return SetFitModel.from_pretrained(MODEL_NAME_OR_PATH, **params)


def hp_space(trial):  # Training parameters
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_epochs": trial.suggest_int("num_epochs", 3, 20),
        "batch_size": trial.suggest_categorical("batch_size", [16]),
        "seed": trial.suggest_int("seed", 1, 5),
        "num_iterations": trial.suggest_categorical("num_iterations", [2, 10, 20]),
        "max_iter": trial.suggest_int("max_iter", 3, 20),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
    }


def training_with_hyperparams(category, metric, save_model):
    hyperparams = config['hyperparameter_search']

    label_list, label2id, id2label_converter, train_dataset, validation_dataset, test_dataset = tdh.prepare_data(
        category=category)

    # Create trainer
    setfit_trainer = SetFitTrainer(
        metric=metric,
        # metric_kwargs=average,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        model_init=model_init
    )

    best_run = setfit_trainer.hyperparameter_search(direction="maximize", hp_space=hp_space,
                                                    n_trials=hyperparams['n_trials'])

    setfit_trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)

    # Train and evaluate
    setfit_trainer.train()
    metrics_dict = setfit_trainer.evaluate()
    print("*** Evaluation ***")
    print(metrics_dict)

    if str2bool(save_model):
        PATH_TO_TEMP_MODELS = make_output_path(config=hyperparams)
        path_to_best_model = os.path.join(os.path.dirname(__file__), PATH_TO_TEMP_MODELS, category,
                                          re.sub('/', '_', MODEL_NAME_OR_PATH))
        if not os.path.exists(path_to_best_model):
            os.makedirs(path_to_best_model)
        setfit_trainer.model._save_pretrained(path_to_best_model)

    validation_dataset_as_dataframe = pd.DataFrame(validation_dataset)

    test_dataset_as_dataframe = pd.DataFrame(test_dataset)

    return setfit_trainer, best_run.hyperparameters, validation_dataset_as_dataframe, test_dataset_as_dataframe, id2label_converter


def training_without_hyperparams(category, metric, save_model):
    hyperparams = config['default_hyperparamters']

    label_list, label2id, id2label_converter, train_dataset, validation_dataset, test_dataset = tdh.prepare_data(
        category=category)

    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

    # , use_differentiable_head=True,
    #                                         head_params={"out_features": len(label_list)}

    # Create trainer
    setfit_trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        loss_class=CosineSimilarityLoss,
        metric=metric,
        metric_kwargs={'average': 'macro'},
        batch_size=hyperparams['batch_size'],
        num_iterations=hyperparams['num_iterations'],  # The number of text pairs to generate for contrastive learning
        num_epochs=hyperparams['num_epochs'],  # The number of epochs to use for contrastive learning
        # column_mapping={"sentence": "text", "label": "label"}  # Map dataset columns to text/label expected by trainer
    )

    # Train and evaluate
    setfit_trainer.train()
    metrics_dict = setfit_trainer.evaluate()
    print("*** Evaluation ***")
    print(metrics_dict)

    if str2bool(save_model):
        PATH_TO_TEMP_MODELS = make_output_path(config=hyperparams)
        path_to_best_model = os.path.join(os.path.dirname(__file__), PATH_TO_TEMP_MODELS, category,
                                          re.sub('/', '_', MODEL_NAME_OR_PATH))
        if not os.path.exists(path_to_best_model):
            os.makedirs(path_to_best_model)
        setfit_trainer.model._save_pretrained(path_to_best_model)

    validation_dataset_as_dataframe = pd.DataFrame(validation_dataset)

    test_dataset_as_dataframe = pd.DataFrame(test_dataset)

    return setfit_trainer, config, validation_dataset_as_dataframe, test_dataset_as_dataframe, id2label_converter


def training(category, metric, save_model=True, hyperparameter_search=False):
    if str2bool(hyperparameter_search):
        return training_with_hyperparams(category=category, metric=metric, save_model=save_model)

    if not str2bool(hyperparameter_search):
        return training_without_hyperparams(category=category, metric=metric, save_model=save_model)


def make_predictions(dataset_as_dataframe, trainer):
    predictions = trainer.model.predict_proba(dataset_as_dataframe.text.tolist())
    predicted_label = [id2label[x.numpy().argmax()] for x in predictions]

    dataset_as_dataframe['label'] = dataset_as_dataframe.label.apply(lambda x: id2label[int(x)])
    dataset_as_dataframe['predicted_label'] = predicted_label
    dataset_as_dataframe["model_name_or_path"] = MODEL_NAME_OR_PATH

    metrics = dict()
    metrics['accuracy'] = accuracy_score(dataset_as_dataframe.label.tolist(),
                                         dataset_as_dataframe.predicted_label.tolist(), normalize=True)
    metrics['macro-f1'] = f1_score(dataset_as_dataframe.label.tolist(),
                                   dataset_as_dataframe.predicted_label.tolist(), average="macro")
    metrics['micro-f1'] = f1_score(dataset_as_dataframe.label.tolist(),
                                   dataset_as_dataframe.predicted_label.tolist(), average="micro")
    metrics['macro-precision'] = precision_score(dataset_as_dataframe.label.tolist(),
                                                 dataset_as_dataframe.predicted_label.tolist(),
                                                 average="macro")
    metrics['micro-precision'] = f1_score(dataset_as_dataframe.label.tolist(),
                                          dataset_as_dataframe.predicted_label.tolist(), average="micro")
    metrics['macro-recall'] = recall_score(dataset_as_dataframe.label.tolist(),
                                           dataset_as_dataframe.predicted_label.tolist(), average="macro")
    metrics['micro-recall'] = recall_score(dataset_as_dataframe.label.tolist(),
                                           dataset_as_dataframe.predicted_label.tolist(), average="micro")
    metrics['matthews_corrcoef'] = matthews_corrcoef(dataset_as_dataframe.label.tolist(),
                                                     dataset_as_dataframe.predicted_label.tolist())

    metrics_df = pd.DataFrame([{'metric': metric, 'score': score} for metric, score in metrics.items()])

    return metrics_df, dataset_as_dataframe


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-cat', '--category',
                        help='Choose on which category you want to train.',
                        choices=['criteria_relevance'],
                        default='criteria_relevance')
    parser.add_argument('-sm', '--save_model', help='', default=True)
    parser.add_argument('-hps', '--hyperparameter_search', default=False,
                        help='Define if you want to perform hyperparameter search.')

    args = parser.parse_args()

    '''PATH_TO_TEMP_MODELS = os.path.join(PATH_TO_TEMP_MODELS, args.category)

    if not os.path.isdir(PATH_TO_TEMP_MODELS ):
        os.makedirs(PATH_TO_TEMP_MODELS )'''

    REPORTS_OUTPUT_PATH = f"{config['eval_output_path']}{str(args.hyperparameter_search)}__{args.category}"

    if not os.path.exists(REPORTS_OUTPUT_PATH):
        os.mkdir(REPORTS_OUTPUT_PATH)

    trainer, best_run_hyperparameters, validation_dataset_as_dataframe, test_dataset_as_dataframe, id2label = training(
        args.category, metric='f1', save_model=args.save_model, hyperparameter_search=args.hyperparameter_search)

    print('Making predictions on validation set.')
    metrics_df_validation_set, validation_dataset_as_dataframe = make_predictions(validation_dataset_as_dataframe,
                                                                                  trainer)

    print('Making predictions on test set.')
    metrics_df_test_set, test_dataset_as_dataframe = make_predictions(test_dataset_as_dataframe, trainer)

    time_stamp = datetime.now().isoformat()
    path_to_results_excel = f'{REPORTS_OUTPUT_PATH}/predictions_and_metrics_for_' + args.category + '_from_' + time_stamp + '.xlsx'
    with pd.ExcelWriter(path_to_results_excel) as writer:
        best_run_hyperparameters = pd.DataFrame([best_run_hyperparameters])
        best_run_hyperparameters.to_excel(writer, index=False, sheet_name="Hyperparameters")
        metrics_df_validation_set.to_excel(
            writer, index=False, sheet_name="scores_validation_set")
        validation_dataset_as_dataframe.to_excel(
            writer, index=False, sheet_name="predictions_validation_set")
        metrics_df_test_set.to_excel(
            writer, index=False, sheet_name="scores_test_set")
        test_dataset_as_dataframe.to_excel(
            writer, index=False, sheet_name="predictions_test_set")
        metrics_df_validation_set.rename(columns={'score': 'Validation', 'metric': 'Metric'}, inplace=True)
        metrics_df_validation_set['Test'] = metrics_df_test_set['score']
        metrics_df_validation_set['Test'] = metrics_df_validation_set['Test'].apply(lambda x: round(x * 100, 2))
        metrics_df_validation_set['Validation'] = metrics_df_validation_set['Validation'].apply(
            lambda x: round(x * 100, 2))
        metrics_df_validation_set.to_excel(
            writer, index=False, sheet_name="scores_validation_and_test_set")
