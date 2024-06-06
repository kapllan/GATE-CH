import argparse
from ast import literal_eval
from pprint import pprint
from traceback import print_exc
from typing import Union, Dict

import torch
import transformers
from helper import json_extractor
from log import get_logger
from peft import AutoPeftModelForCausalLM, PeftModel
from prompt_database import *
from sentence_transformers import SentenceTransformer, util
from torch.nn import DataParallel
from transformers import StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM

sent_model = SentenceTransformer.load("PM-AI/bi-encoder_msmarco_bert-base_german")
criteria_check_labels = ['ja', 'nein']
label_embeddings = sent_model.encode(criteria_check_labels)

logging = get_logger()

# special tokens used by llama 2 chat
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

prompt_template_mapper = {"jphme/Llama-2-13b-chat-german": {
    "begin": "[INST]",
    "end": "[/INST]"
},
    "VAGOsolutions/SauerkrautLM-Mixtral-8x7B-Instruct": {
        "begin": "[INST]",
        "end": "[/INST]"
    },
}


class EndTokenCriteria(StoppingCriteria):
    def __init__(self, end_token_id):
        self.end_token_id = end_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        token_id_most_likely = input_ids[:, -1].item()
        if token_id_most_likely == self.end_token_id:
            return True
        else:
            return False


class LLMHandler:

    def __init__(self, model_name_or_path: str, max_new_tokens=4096, temperature=0.0, do_sample=False,
                 repetition_penalty=1.1,
                 tokenizer_model: str = None):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        if tokenizer_model is None:
            self.tokenizer_model = model_name_or_path
        else:
            self.tokenizer_model = tokenizer_model

        config_class = self.find_config()
        self.model_config = config_class.from_pretrained(self.model_name_or_path)  # use_auth_token=hf_auth

        # initialize the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,  # True for flash-attn2 else False
            config=self.model_config,
            device_map='auto',
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_model,
            # use_auth_token=hf_auth
        )

        self.text_gen_pipeline = transformers.pipeline(
            model=self.model, tokenizer=self.tokenizer,
            return_full_text=False,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            temperature=self.temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=self.max_new_tokens,  # mex number of tokens to generate in the output
            repetition_penalty=self.repetition_penalty,  # without this output begins repeating
            do_sample=self.do_sample
            # stopping_criteria=stopping_criteria
        )

    def find_config(self):
        if 'mistral' in self.model_name_or_path.lower():
            return transformers.MistralConfig
        else:
            return transformers.AutoConfig

    def extract_award_criteria(self, text: str, prompt: str = None, convert_to_json: bool = False) -> Union[str, Dict]:
        if prompt is None:
            prompt = PROMPT_3
        final_command = prompt + text + ' ' + E_INST
        output = self.text_gen_pipeline(final_command)[0]["generated_text"]
        if convert_to_json:
            try:
                output = json_extractor(str(output))
            except Exception as e:
                logging.error("Could not extract a JSON from: {}".format(output))
                logging.error(e)
        return output

    def has_award_criteria(self, text: str, prompt: str = None) -> str:
        if prompt is None:
            prompt = prompt_check_if_award_criteria_1
        final_command = prompt + text + ' ' + E_INST
        output = self.text_gen_pipeline(final_command)[0]["generated_text"]
        output = output.lower().strip()
        if output.startswith("ja"):
            return 'has_criteria'
        elif output.startswith("nein"):
            return 'has_no_criteria'
        else:
            output_embedding = sent_model.encode([output])
            _index = util.semantic_search(output_embedding, label_embeddings)[0][0]['corpus_id']
            return criteria_check_labels[-_index]

    def __call__(self, instruction: str):
        return self.text_gen_pipeline(instruction)


if __name__ == '__main__':
    import json as js

    parser = argparse.ArgumentParser()

    parser.add_argument('-mnp', '--model_name_or_path', default="VAGOsolutions/SauerkrautLM-Mixtral-8x7B-Instruct",
                        help="Specify the model name.")
    parser.add_argument('-t', '--text', help="Insert your command here.")

    args = parser.parse_args()

    llm_handler = LLMHandler(args.model_name_or_path, max_new_tokens=512 * 4)

    COMMAND = B_INST + ' ' + args.text + ' ' + E_INST

    result = llm_handler.text_gen_pipeline(COMMAND)[0]["generated_text"]

    print(result)
