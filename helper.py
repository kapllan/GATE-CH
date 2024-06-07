import json as js
import os
import re
import signal
import sys
import zipfile
from ast import literal_eval
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from pathlib import Path
from traceback import print_exc

import PyPDF2
import dateparser
import fasttext
import fitz
import tabula
import yaml
from PIL import Image


def remove_alpha_chars(input_string):
    if isinstance(input_string, str):
        return ''.join(char for char in input_string if bool(re.search(r'\d', char)))
    else:
        return str(input_string)


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

    return text


def normalize_criteria_representation(relation_item):
    """
    In my annotations, for zkNummer I annotated only the numbers.
    However, very often zkNummer looks like this ZK1, ZK2 etc.
    The LLM, therefore, might return ZK1, ZK2 etc. and not just the number.
    This is not wrong, however. Therefore, to make the evaluation fair,
    I strip all characters from zkNummer and keep only the digits.
    The same is done for maxPunkte and gewichtung.

    """

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
