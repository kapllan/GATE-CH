import json as js
import os
import re
import signal
import sys
import zipfile
from ast import literal_eval
from datetime import datetime, timedelta
from pathlib import Path
from pathlib import Path
from traceback import print_exc

import PyPDF2
import dateparser
import fasttext
import fitz
import tabula
import yaml
from difflib import SequenceMatcher
from PIL import Image

ENTRY_TEMPLATE = {
    "zkNummer": "",
    "kriterium": "",
    "gewichtung": "",
    "maxPunkte": ""
}


def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def slice_into_chunks(lst, chunk_size):
    """Slices a list into a list with smaller lists."""

    chunks = []  # Initialize an empty list to hold the chunks
    for i in range(0, len(lst), chunk_size):
        chunks.append(lst[i:i + chunk_size])  # Append sliced chunks to the list
    return chunks


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_main_config() -> dict:

    current_path = os.path.dirname(__file__)

    path_to_config_file = os.path.join(current_path, './config.yml')

    with open(path_to_config_file, 'r') as f:
        config = yaml.safe_load(f)

    paths = config['paths']
    for key, path in paths.items():
        paths[key] = os.path.join(current_path, path)

    return config


def handle_sigint(sig, frame):
    print('SIGINT received, terminating.')
    sys.exit()


def handle_timeout(sig, frame):
    raise TimeoutError('took too long')


def unzip_folder(path_to_zip_file: str):
    directory_to_extract_to = path_to_zip_file[:-4]

    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)

    return directory_to_extract_to


def unzip_everything(path_to_zip_file: str):
    if str(path_to_zip_file).endswith('zip'):
        directory_to_extract_to = unzip_folder(path_to_zip_file)
        for root, dirs, files in os.walk(directory_to_extract_to):
            for filename in files:
                if filename.endswith('zip'):
                    filename_full_path = os.path.join(root, filename)
                    unzip_everything(filename_full_path)
        return directory_to_extract_to


def get_reader(pdf_file: str):
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGALRM, handle_timeout)
    try:
        signal.alarm(5)
        reader = PyPDF2.PdfReader(pdf_file)
        return reader
    except TimeoutError as exc:
        print(f'Error occurred in this file {pdf_file}.')
        print(exc)


def get_number_of_pages(pdf_file: str):
    reader = get_reader(pdf_file)

    number_pf_pages = [n for n in range(0, len(reader.pages))]

    return number_pf_pages


def extract_text_from_pdf(pdf_file: str):
    content = []

    with fitz.open(pdf_file) as doc:
        for page in doc:
            content.append(page.get_text())

    return content


def extract_pdf_page_as_image(pdf_path: str, page: int, dpi: int = 300) -> Image:
    with fitz.open(pdf_path) as doc:
        page: fitz.Page = doc.load_page(page)
        pixel_map: fitz.Pixmap = page.get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", [pixel_map.width, pixel_map.height], pixel_map.samples)


def extract_text_from_pdf_1(pdf_file: str):
    content = []

    try:
        # TODO: Fix this error that sometimes occurs: PdfReadError: EOF marker not found
        reader = get_reader(pdf_file)

        number_pf_pages = [n for n in range(0, len(reader.pages))]

        for n in number_pf_pages:
            try:
                page_content = reader.pages[n].extract_text()
                content.append(page_content)
            except:
                print_exc()

        return content
    except:
        print_exc()
        return content


def extract_tables_from_pdf(pdf_file: str):
    try:
        pages = get_number_of_pages(pdf_file)
    except:
        pages = []

    all_tables = list()

    for p in pages:
        try:
            table = tabula.read_pdf(pdf_file, pages=p)
            if len(table) > 0:
                entry = dict()
                entry['file'] = pdf_file
                entry['page_number'] = p
                entry['tables'] = table
                all_tables.append(entry)
        except Exception as e:
            print(e)
            print(pdf_file)

    return all_tables


def remove_empty_strings(text):
    text_new = re.sub(r'\n', ' ', text)
    text_new = re.sub(r'\s+', ' ', text_new)
    return text_new


def remove_all_empty_spaces(text):
    text_new = remove_empty_strings(text)
    text_new = re.sub(r'\s+', '', text_new)
    return text_new


def replace_umlaute(filename):
    """
    It seems Inception has problems with umlaute in filenames.
    :param filename:
    :return:
    """

    filename_new = re.sub(r'ä', 'ae', filename)
    filename_new = re.sub(r'ö', 'oe', filename_new)
    filename_new = re.sub(r'ü', 'ue', filename_new)
    filename_new = re.sub(r'ß', 'ss', filename_new)

    return filename_new


def json_extractor(sample_string):
    sample_string = re.sub(r'`', '', sample_string)
    sample_string = re.sub(r"null", "''", sample_string)
    # Remove linebreaks caused by splitting up compounds with -
    sample_string = re.sub(r"-\n", "", sample_string)
    sample_string = re.sub(r"\n", " ", sample_string)
    sample_string = re.sub(r"\] \]", "]", sample_string)
    sample_string = re.sub(r"\]\]", "]", sample_string)
    regex = r"(\[[^\[\]]*\])"
    if len(re.findall(regex, sample_string)) > 0:
        sample_string = re.findall(regex, sample_string)[0]
    sample_string = literal_eval(sample_string)
    # Make sure the list contains only dictionaries
    sample_string = [x for x in sample_string if isinstance(x, dict)]

    return sample_string


class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = "../models/lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=2)  # returns top 2 matching languages
        return predictions


def remove_alpha_chars(input_string):
    if isinstance(input_string, str):
        return ''.join(char for char in input_string if bool(re.search(r'\d', char)))
    else:
        return str(input_string)

def normalize_criteria_representation(relation_item):
    """
    In my annotations, for zkNummer I annotated only the numbers.
    However, very often zkNummer looks like this ZK1, ZK2 etc.
    The LLM, therefore, might return ZK1, ZK2 etc. and not just the number.
    This is not wrong, however. Therefore, to make the evaluation fair,
    I strip all characters from zkNummer and keep only the digits.
    The same is done for maxPunkte and gewichtung.

    """

    '''for key in ENTRY_TEMPLATE.keys():
        if key not in relation_item.keys():
            relation_item[key] = ENTRY_TEMPLATE[key]'''

    for field in ["zkNummer", "gewichtung", "maxPunkte"]:
        if field in relation_item.keys():
            if relation_item[field]:
                relation_item[field] = str(remove_alpha_chars(relation_item[field])).strip()
    for field in ["kriterium", "zkNummer", "gewichtung", "maxPunkte"]:
        if field in relation_item.keys():
            if relation_item[field]:
                relation_item[field] = re.sub(r"\n-", "", relation_item[field])
                relation_item[field] = re.sub(r"-\n", "", relation_item[field])

    all_values = list(relation_item.values())
    all_values = [v for v in all_values if v]  # Keep onl values that are not empty strings
    if len(all_values) == 0:
        return []

    return relation_item


def get_criteria_representation(relation_item):
    list_of_criteria = list()
    for rel in relation_item:
        if rel['head_span']['label'] == 'kriterium':
            # entry = deepcopy(entry_template)
            # entry['kriterium'] = rel['head_span']['span']
            list_of_criteria.append(rel['head_span'])

    list_of_criteria_final = list()

    for item in list_of_criteria:
        entry = deepcopy(ENTRY_TEMPLATE)
        entry[item['label']] = item['span']
        for rel in relation_item:
            for field in ['gewichtung', 'maxPunkte', 'zkNummer']:
                if item == rel['head_span']:
                    if field == rel['child_span']['label']:
                        entry[field] = rel['child_span']['span']
                elif item == rel['child_span']:
                    if field == rel['head_span']['label']:
                        entry[field] = rel['head_span']['span']

        if entry not in list_of_criteria_final:
            entry = normalize_criteria_representation(entry)
            list_of_criteria_final.append(entry)

    return list_of_criteria_final


def convert_info_node_to_text(info_node):
    if isinstance(info_node, str) and not info_node:
        info_node = dict()
    else:
        info_node = literal_eval(info_node)
    text = ''
    lookup_dict = {'zkNummer': 'Zuschlagskriteriumnummer',
                   'kriterium': 'Zuschlagskriterium',
                   'gewichtung': 'Gewichtung',
                   'maxPunkte': 'Maximalpunkte'}
    if info_node:
        for label in ['zkNummer', 'kriterium', 'gewichtung', 'maxPunkte']:
            if label in info_node.keys():
                value = info_node[label]
                name = lookup_dict[label]
                line = f'{name} ist {value}. '
                text += line
        """else:
            text += f'Es gibt keine Information über {lookup_dict[label]}. '
    else:
        for label in ['zkNummer', 'kriterium', 'gewichtung', 'maxPunkte']:
            text += f'Es gibt keine Information zu {lookup_dict[label]}. '"""

    return text


def convert_criteria_info_to_string(criteria_info, as_text=False):
    if isinstance(criteria_info, str):
        return criteria_info

    criteria_info_as_string = list()

    for c_info in criteria_info:
        c_info = dict(sorted([x for x in c_info.items()], key=lambda x: x[0]))
        c_info = js.dumps(c_info)
        criteria_info_as_string.append(c_info)

    criteria_info_as_string = sorted(criteria_info_as_string)

    if as_text:
        if isinstance(criteria_info_as_string, list) and not criteria_info:
            criteria_info_as_string = [""]
        criteria_info_as_string = [convert_info_node_to_text(x) for x in criteria_info_as_string]
        criteria_info_as_string = sorted(criteria_info_as_string)
        criteria_info_as_string = ' '.join(criteria_info_as_string)
    else:
        criteria_info_as_string = [js.dumps(x) for x in criteria_info_as_string]
        criteria_info_as_string = sorted(criteria_info_as_string)
        criteria_info_as_string = js.dumps(criteria_info_as_string, ensure_ascii=False)

    return criteria_info_as_string


def get_dates_between(start_date_str, end_date_str):
    # Convert string representations to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    # Calculate the difference between end_date and start_date
    date_difference = end_date - start_date

    # Generate a list of dates between start_date and end_date
    date_list = [start_date + timedelta(days=x) for x in range(date_difference.days + 1)]

    # Convert datetime objects back to string representations
    date_strings = [date.strftime("%Y-%m-%d") for date in date_list]

    return date_strings


def clean_award_criteria_output(llm_output, separator='###'):
    llm_output_as_list = llm_output.split(separator)
    llm_output_as_list = [output.strip() for output in llm_output_as_list]
    llm_output_as_list = [output.strip() for output in llm_output_as_list if output]
    return llm_output_as_list


if __name__ == '__main__':
    LANGUAGE = LanguageIdentification()
    lang = LANGUAGE.predict_lang("Hej")
    print(lang)
