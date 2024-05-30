import argparse
import os
import sys
from pprint import pprint
from typing import Literal
import torch
import re
from rapidfuzz import fuzz
import fitz
from setfit import SetFitModel
from helper import extract_text_from_pdf, LanguageIdentification, remove_all_empty_spaces, unzip_folder, \
    unzip_everything, replace_umlaute, get_main_config, str2bool
from PIL import Image
import shutil
from traceback import print_exc
import time
from templates import ClassifierPrediction

current_path = os.path.dirname(__file__)

from training_data_handler import TrainingDataHandler

tdh = TrainingDataHandler()

config = get_main_config()


OUTPUT_PATH = config['paths']['output_path_relevance_criteria']


class RelevanceClassifier:

    def __init__(self, model_name_or_path: str = None,
                 category: Literal['criteria_relevance'] = 'criteria_relevance') -> None:

        if category not in ['criteria_relevance']:
            print('Error. Category can only be one of these values: ', 'criteria_relevance')

        self.category = category

        training_data = tdh.load_dataset(self.category)
        all_data = training_data["all"]

        label2id = dict()
        id2label = dict()

        label_list = sorted(list(all_data.label.unique()))
        for i, l in enumerate(label_list):
            label2id[l] = i
            id2label[i] = l

        self.label_list = label_list
        self.label2id = label2id
        self.id2label = id2label

        # path_to_best_model = os.path.join(current_path, "../temporary_models/" + self.category)

        if model_name_or_path is None:
            self.model_name_or_path = config['models']['winning_criteria_binary_classifier']
        else:
            self.model_name_or_path = model_name_or_path

        self.finetuned_model = SetFitModel.from_pretrained(self.model_name_or_path)
        # self.finetuned_model.to('mps')

    def predict(self, text: str) -> ClassifierPrediction:
        # Constraint, because texts that are too short will not provide enough information to make a distinction
        if not bool(re.search(r'\w', text)) or len(text) < 2:
            return ClassifierPrediction(predicted_class="has_no_criteria", probability=1.0)
        else:
            output = self.finetuned_model.predict_proba([text])[0]
            pprint(output)
            prediction = output.argmax()
            if torch.is_tensor(prediction):
                prediction = prediction.item()
            prediction_label = self.id2label[prediction]
            if prediction_label == 'has_criteria' and output[prediction] > 0.8:
                return ClassifierPrediction(predicted_class="has_criteria", probability=output[prediction])

            return ClassifierPrediction(predicted_class="has_no_criteria", probability=1-output[prediction])


if __name__ == '__main__':

    '''for subfolder in ['pdfs', 'images', 'txt', 'too_similar']:
        final_path = f"{OUTPUT_PATH}/{subfolder}"
        if not os.path.exists(final_path):
            os.makedirs(final_path)'''

    parser = argparse.ArgumentParser()

    parser.add_argument('-fp', '--filepath', help='Insert the path of the file you want to analyse.')
    parser.add_argument('-c', '--check_if_in_training_data',
                        help='Check if the detected page has high similarity to the instances of the training dataset.',
                        default=False)

    args = parser.parse_args()

    args.check_if_in_training_data = str2bool(args.check_if_in_training_data)

    lang_dedector = LanguageIdentification()

    predictor = RelevanceClassifier()

    # Then you extract the content of the PDFs that you have already annotated.
    # You need them to check if the new documents are too similar.
    # We do not want to have always the same documents in the annotation dataset.

    already_annotated_documents = list()

    for root, dirs, files in os.walk('../annotation/curated_annotations/PDFs/files'):
        for file in files:
            if str(file).endswith('pdf'):
                full_path = os.path.join(root, file)
                pages = extract_text_from_pdf(full_path)
                document = ' '.join(pages)
                already_annotated_documents.append(document)

    for root, dirs, files in os.walk(args.filepath):
        for file in files:
            if str(file).endswith('zip'):
                full_path = os.path.join(root, file)
                try:
                    unzip_everything(full_path)
                except Exception as e:
                    print(e)
                    print(full_path)

    if args.filepath.endswith('pdf'):
        pages = extract_text_from_pdf(args.filepath)
        hits = list()
        for n, page in enumerate(pages):
            label = predictor.predict(page)
            if label == 'has_criteria':
                hits.append(label)
            print(f'Page {n + 1}: ', label)
        if len(hits) > 0:
            os.system(f'open "{args.filepath}"')
    else:
        hits = dict()
        for root, dirs, files in os.walk(args.filepath):
            for file in files:
                if str(file).endswith('pdf'):
                    full_path = os.path.join(root, file)
                    hits[full_path] = list()
                    print(f'Processing {full_path}')
                    try:
                        pages = extract_text_from_pdf(full_path)
                        for n, page in enumerate(pages):
                            to_be_added = True
                            lang = lang_dedector.predict_lang(remove_all_empty_spaces(page))[0][0]
                            if lang == '__label__de':
                                label = predictor.predict(page)
                                if label == 'has_criteria':
                                    if args.check_if_in_training_data:
                                        for aa_doc in already_annotated_documents:
                                            for aa_page in aa_doc:
                                                if fuzz.ratio(aa_page, page) > 70.0:
                                                    to_be_added = False
                                    if to_be_added:
                                        file_new = full_path.split('/')
                                        subfolder = replace_umlaute('_'.join(full_path.split('/')[:-1]))
                                        subfolder = replace_umlaute(subfolder)
                                        subfolder = re.sub(r'\.', '', subfolder)
                                        print('SUBFOLDER:', subfolder)
                                        if not os.path.exists(os.path.join(OUTPUT_PATH, subfolder)):
                                            os.makedirs(os.path.join(OUTPUT_PATH, subfolder))
                                        file_new = file_new[-1]
                                        file_new = file_new[:-4]
                                        file_new = re.sub(' ', '_', file_new)
                                        # if bool(re.search(r'(Zuschlag|Kriteri)', page, re.IGNORECASE)) or bool(re.search(r'ZK', page)):
                                        hits[full_path].append(n)
                                        print(f'Page {n + 1}: ', label)
                                        file_handle = fitz.open(full_path)
                                        current_pdf_page = file_handle[n]
                                        page_img = current_pdf_page.get_pixmap()
                                        full_path_to_img = f'{OUTPUT_PATH}/{subfolder}/{file_new}.jpg'
                                        page_img.save(full_path_to_img)
                                        img = Image.open(full_path_to_img)
                                        shutil.copy(full_path, f'{OUTPUT_PATH}/{subfolder}/{file_new}.pdf')
                                        with open(f'{OUTPUT_PATH}/{subfolder}/{file_new}.txt', 'w') as f:
                                            print(page, file=f)
                                        # img.show()
                                        # time.sleep(5)
                            else:
                                print(f'That is not German, but {lang}.')
                        already_annotated_documents.append(pages)


                    except:
                        print_exc()

        pprint(hits)
