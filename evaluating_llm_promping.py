import argparse
import os
from copy import deepcopy
from statistics import mean
from typing import List, AnyStr, Dict

import nltk
import numpy as np
import pandas as pd
import setproctitle
import spacy
import torch
from helper import *
from llm_handler import LLMHandler
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from prompt_database import prompt_dict
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

nltk.download('wordnet')

spacy_nlp = spacy.load('de_core_news_lg')

rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)

config = get_main_config()


# relevance_predictor = RelevanceClassifier(config['output_path_relevance_criteria'])

class MultiScorer:

    def __init__(self, spacy_nlp):
        self.spacy_nlp = spacy_nlp

    def eval_json(self, ground_truth: dict, generated: dict):
        true_positives = 0
        for key, value in generated.items():
            if key in ground_truth and ground_truth[key] == value:
                true_positives += 1

        precision = true_positives / len(generated)
        recall = true_positives / len(ground_truth)

        if precision + recall == 0:
            return 0  # Avoid division by zero

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score, precision, recall

    def calculate_metrics(self, y_true: list, y_pred: list):

        if y_true == y_pred:
            return dict(precision_macro=1, recall_macro=1, f1_macro=1, precision_micro=1, recall_micro=1, f1_micro=1,
                        precision_weighted=1, recall_weighted=1, f1_weighted=1, accuracy_not_normalized=1,
                        accuracy_normalized=len(y_true))


        elif y_true != y_pred and y_pred == [[]]:
            return dict(precision_macro=0, recall_macro=0, f1_macro=0, precision_micro=0, recall_micro=0, f1_micro=0,
                        precision_weighted=0, recall_weighted=0, f1_weighted=0, accuracy_not_normalized=0,
                        accuracy_normalized=0)

        else:
            # Calculate Precision, Recall, and F1 Score
            precision_macro = precision_score(y_true, y_pred, average='macro')
            recall_macro = recall_score(y_true, y_pred, average='macro')
            f1_macro = f1_score(y_true, y_pred, average='macro')

            precision_micro = precision_score(y_true, y_pred, average='micro')
            recall_micro = recall_score(y_true, y_pred, average='micro')
            f1_micro = f1_score(y_true, y_pred, average='micro')

            precision_weighted = precision_score(y_true, y_pred, average='weighted')
            recall_weighted = recall_score(y_true, y_pred, average='weighted')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')

            accuracy_not_normalized = accuracy_score(y_true=y_true, y_pred=y_pred, normalize=False)
            accuracy_normalized = accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True)

            return {'precision_macro': precision_macro,
                    'recall_macro': recall_macro,
                    'f1_macro': f1_macro,
                    'precision_micro': precision_micro,
                    'recall_micro': recall_micro,
                    'f1_micro': f1_micro,
                    'precision_weighted': precision_weighted,
                    'recall_weighted': recall_weighted,
                    'f1_weighted': f1_weighted,
                    'accuracy_not_normalized': accuracy_not_normalized,
                    'accuracy_normalized': accuracy_normalized
                    }

    def eval_key_value_match(self, ground_truth: List[Dict], prediction: List[Dict], labels: List[AnyStr] = None):

        if labels is None:
            labels = ['kriterium', 'gewichtung', 'maxPunkte', 'zkNummer']

        ground_truth_selected = list()
        prediction_selected = list()

        for entry in ground_truth:
            label_ = ''
            for label in labels:
                if isinstance(entry, dict) and label in entry.keys():
                    label_ += label + '-' + entry[label] + '__'
            ground_truth_selected.append(label_)

        for entry in prediction:
            label_ = ''
            for label in labels:
                if isinstance(entry, dict) and label in entry.keys():
                    label_ += label + '-' + entry[label] + '__'
            prediction_selected.append(label_)

        # Create binary indicators for each string (treating each string as a label)
        all_labels = sorted(set(ground_truth_selected + prediction_selected))  # Get all unique labels
        y_true = [[1 if label in ground_truth_selected else 0 for label in all_labels]]
        y_pred = [[1 if label in prediction_selected else 0 for label in all_labels]]

        return self.calculate_metrics(y_true, y_pred)

    def evaluate_spans(self, ground_truth: List[Dict], prediction: List[Dict], labels: List[AnyStr] = None):

        all_results = dict()

        if labels is None:
            labels = ['kriterium', 'gewichtung', 'maxPunkte', 'zkNummer']

        for label in labels:
            ground_truth_selected = list()
            prediction_selected = list()

            for entry in ground_truth:
                if isinstance(entry, dict) and label in entry.keys():
                    ground_truth_selected.append(entry[label])

            for entry in prediction:
                if isinstance(entry, dict) and label in entry.keys():
                    prediction_selected.append(entry[label])

            # Create binary indicators for each string (treating each string as a label)
            all_labels = sorted(set(ground_truth_selected + prediction_selected))  # Get all unique labels
            y_true = [[1 if label in ground_truth_selected else 0 for label in all_labels]]
            y_pred = [[1 if label in prediction_selected else 0 for label in all_labels]]

            metrics = self.calculate_metrics(y_true, y_pred)

            all_results[label] = metrics

        return all_results


multi_scorer = MultiScorer(spacy_nlp=spacy_nlp)


def turn_rouge_score_to_dict(rouge_result):
    result_dict = dict()
    for score, values in rouge_result.items():
        entry = dict()
        for field in ['precision', 'recall', 'fmeasure']:
            entry[field] = getattr(values, field)
        result_dict[score] = entry

    return result_dict


def filter_values(extracted_infos, field):
    if isinstance(extracted_infos, str):
        return ""

    extracted_infos_filtered = list()

    for entry in extracted_infos:
        entry_new = dict()
        if field in entry.keys():
            entry_new[field] = entry[field]
        else:
            entry_new[field] = ''
        extracted_infos_filtered.append(entry_new)

    return extracted_infos_filtered


def apply_calculations(dataframe, label=None, as_text=False):
    if label is not None:
        dataframe['y_true'] = dataframe['y_true'].apply(lambda x: filter_values(x, label))
        dataframe['y_pred'] = dataframe['y_pred'].apply(lambda x: filter_values(x, label))

    dataframe['y_true_as_string'] = dataframe['y_true'].apply(lambda x: convert_criteria_info_to_string(x, as_text))
    dataframe['y_pred_as_string'] = dataframe['y_pred'].apply(lambda x: convert_criteria_info_to_string(x, as_text))

    if label is None:
        dataframe['key_value_eval'] = dataframe.apply(
            lambda row: multi_scorer.eval_key_value_match(row['y_true'], row['y_pred']),
            axis=1)

        dataframe[f"span_scores"] = dataframe.apply(
            lambda row: multi_scorer.evaluate_spans(row['y_true'], row['y_pred']),
            axis=1)

    return dataframe


def apply_llm(training_set_df: pd.core.frame.DataFrame, prompt_id: str,
              llm_handler: LLMHandler) -> pd.core.frame.DataFrame:
    all_scores = list()

    for row in training_set_df.to_dict(orient='records'):
        # if row['filename'] == '35_114328 AnbindLIWA AusschBaumeister 5_03  AGB WVZ.pdf':
        # if row["criteria_info"]:
        print(row['filename'])
        row['label_predicted'] = llm_handler.has_award_criteria(row['context'])
        result_entry = deepcopy(row)
        result_entry['prompt_id'] = prompt_id
        result_entry['error message'] = ''
        # TODO: Sometimes, I get the CUDA out of memory message
        pred_output_as_string = llm_handler.extract_award_criteria(text=row['context'], prompt=prompt_dict[prompt_id])
        try:
            pred_output_as_dict = json_extractor(pred_output_as_string)
        except Exception as e:
            print(e)
            result_entry['error message'] = e
            pred_output_as_dict = ''

        if isinstance(pred_output_as_dict, list):
            pred_output_as_dict = [normalize_criteria_representation(pred) for pred in pred_output_as_dict]
            pred_output_as_dict = [pred for pred in pred_output_as_dict if pred]
            result_entry['conversion_status'] = 'ok'
        else:
            result_entry['conversion_status'] = 'failed'
        print(pred_output_as_dict)
        result_entry['y_pred'] = pred_output_as_dict
        result_entry['pred_output_as_string'] = pred_output_as_string

        all_scores.append(result_entry)

    all_scores_df = pd.DataFrame(all_scores)

    return all_scores_df


def perform_all_evaluations(all_scores_df: pd.core.frame.DataFrame, output_folder: str, mode: str):
    with pd.ExcelWriter(f'{output_folder}/evaluation_results_{mode}.xlsx') as writer:

        all_scores_df = apply_calculations(all_scores_df, as_text=True)

        all_scores_df.to_excel(writer, index=False, sheet_name=f"results_on_all_labels")

        overview = list()

        for label in [None, 'zkNummer', 'kriterium', 'gewichtung', 'maxPunkte']:
            all_scores_df_copy = deepcopy(all_scores_df)

            all_scores_df_copy = apply_calculations(all_scores_df_copy, label=label, as_text=True)

            all_scores_df_copy.to_excel(writer, index=False, sheet_name=f"results_on_{label}")

        overview_df = pd.DataFrame(overview)
        overview_df.to_excel(writer, index=False, sheet_name=f"Overview_all")

        # Now only those examples where the conversion from text to dictionary worked
        overview = list()
        for label in [None, 'zkNummer', 'kriterium', 'gewichtung', 'maxPunkte']:
            all_scores_df_copy = deepcopy(all_scores_df)

            all_scores_df_copy = all_scores_df_copy[all_scores_df_copy.conversion_status == 'ok']

            all_scores_df_copy = apply_calculations(all_scores_df_copy, label=label, as_text=True)

            for prompt_id in prompt_dict.keys():
                all_scores_df_sub = all_scores_df_copy[all_scores_df_copy.prompt_id == prompt_id]
                if len(all_scores_df_sub) > 0:
                    item = dict()
                    item['label'] = label
                    item['pompt_id'] = prompt_id
                    overview.append(item)
        overview_df = pd.DataFrame(overview)
        overview_df.to_excel(writer, index=False, sheet_name=f"Overview_successful_conversion")


if __name__ == '__main__':

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    setproctitle.setproctitle('msv3 - evaluating_llm_promping_new')

    parser = argparse.ArgumentParser()

    parser.add_argument('-mnp', '--model_name_or_path',
                        default="VAGOsolutions/SauerkrautLM-Mixtral-8x7B-Instruct")
    parser.add_argument('-fp', '--filepath', help='Insert the path of the file you want to analyse.')
    parser.add_argument('-op', '--output_path', default="./evaluation_results_paper_final_1/")
    parser.add_argument('-m', '--mode', default=None, choices=['has_criteria', 'has_no_criteria'],
                        help="Define whether you want to evaluate only examples that have criteria (has_criteria) or have no criteria (has_no_criteria).")
    parser.add_argument('-ns', '--number_of_samples', type=int, default=None,
                        help="Define the number of samples you want to evaluate.")
    parser.add_argument('-pi', '--prompt_id', type=str, default=None,
                        help="Define which prompt you want to use.")

    args = parser.parse_args()

    llm_handler = LLMHandler(model_name_or_path=args.model_name_or_path)

    for prompt_id, prompt in prompt_dict.items():
        if args.prompt_id is None or prompt_id == args.prompt_id:
            OUTPUT_PATH = f"{args.output_path}/{args.model_name_or_path}/{prompt_id}"
            if not os.path.exists(OUTPUT_PATH):
                os.makedirs(OUTPUT_PATH)
            all_scores_df = list()
            training_set_df = pd.read_json('./datasets/text_classification_dataset.jsonl', lines=True)
            # We do this below, to see how the json used to look without normalization.
            training_set_df['y_true'] = training_set_df['criteria_info'].apply(
                lambda pred_true: [normalize_criteria_representation(pred) for pred in pred_true])
            training_set_df['context_length'] = training_set_df.context.apply(lambda x: len(x))
            if args.mode is not None:
                training_set_df = training_set_df[training_set_df.label == args.mode]
            if args.number_of_samples is not None:
                training_set_df = training_set_df[:args.number_of_samples]
            df = apply_llm(llm_handler=llm_handler, prompt_id=prompt_id, training_set_df=training_set_df)
            all_scores_df.append(df)
            torch.cuda.empty_cache()

            all_scores_df = pd.concat(all_scores_df)

            perform_all_evaluations(all_scores_df=all_scores_df, output_folder=OUTPUT_PATH, mode="all")

            mode_choices = next(action for action in parser._actions if '--mode' in action.option_strings)
            mode_choices = mode_choices.choices

            for mode in mode_choices:
                print(f'Making evaluation for mode {mode}')
                all_scores_df_subset = all_scores_df[all_scores_df.label == mode]
                try:
                    perform_all_evaluations(all_scores_df=all_scores_df_subset, output_folder=OUTPUT_PATH, mode=mode)
                except Exception as e:
                    print(e)
