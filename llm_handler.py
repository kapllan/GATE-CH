# Sources:
# - https://www.pinecone.io/learn/llama-2/

import torch
import argparse
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM
from pprint import pprint
from helper import json_extractor
from prompt_database import *
from ast import literal_eval
from traceback import print_exc
from torch.nn import DataParallel
from log import get_logger
from typing import Union, Dict
from peft import AutoPeftModelForCausalLM, PeftModel
from sentence_transformers import SentenceTransformer, util

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
        if tokenizer_model is None:
            self.tokenizer_model = model_name_or_path
        else:
            self.tokenizer_model = tokenizer_model
        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        self.bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True
        )

        config_class = self.find_config()
        self.model_config = config_class.from_pretrained(self.model_name_or_path)  # use_auth_token=hf_auth

        # initialize the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,  # True for flash-attn2 else False
            config=self.model_config,
            # quantization_config=self.bnb_config,
            device_map='auto',
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            # bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # llm_int8_enable_fp32_cpu_offload=True
            # use_auth_token=hf_auth
        )

        self.model.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_model,
            # use_auth_token=hf_auth
        )

        # end_token_id_1 = self.tokenizer.encode(']', return_tensors='pt')[0][0].item()
        # end_token_id_2 = self.tokenizer.encode(']', return_tensors='pt')[0][1].item()
        # stopping_criteria = StoppingCriteriaList([EndTokenCriteria([end_token_id_1, end_token_id_2])])

        self.text_gen_pipeline = transformers.pipeline(
            model=self.model, tokenizer=self.tokenizer,
            return_full_text=False,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            temperature=self.temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=self.max_new_tokens,  # mex number of tokens to generate in the output
            repetition_penalty=self.repetition_penalty,  # without this output begins repeating
            do_sample=do_sample
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

    args = parser.parse_args()

    llm_handler = LLMHandler(args.model_name_or_path, max_new_tokens=512 * 4)

    TEXT = '''
            EK4 Qualitätsmanagement
            Die federführende Firma des Anbieters muss über eine ISO 9001-Zertifizierung oder eine vergleichbare Zertifzierung verfügen. Eine Kopie der Zertifzierung ist dem Angebot beizulegen.
            4.2 Zuschlagskriterien
            4.2.1 Grundsätze der Bewertung
            Die Angebote werden mit Zuschlagskriterien bewertet. Die Bewertungen werden mit Gewichtungen mul-tipliziert. Aus der Summe dieser Werte ergibt sich der Nutzwert des Angebots. Die Vergabe erfolgt an den Anbieter mit dem höchsten Nutzwert.
            Im Folgenden sind die Zuschlagskriterien und ihre Gewichtungen aufgeführt:
                Kriterien
                Gewichtung in % (G)
                Z1 Referenzen Schlüsselpersonen
                10 %
                Z1.1 Leiter Planung
                15 %
                21.2 Montageleiter
                15 %
                Z2 Referenz in BIM to field
                10 %
                Z3 Preis
                80 %
                Jedes Kriterium wird mit einer Note (N) zwischen 0 und 5 in Schritten von ganzen Punkten bewertet.
                Anschliessend werden pro Kriterium die Wertungen mit den Gewichtungen (G) multipliziert. Das Angebot mit der höchsten Punktzahl (P) erhält den Zuschlag.
                Bewertung Preis
                Die Bewertung des Preises erfolgt mit folgender Bewertungsmethode: Das Angebot mit dem tiefsten
                Pai ende domaine Das Angend Pl orien feuren mieten an mieten,
                vergeben und mit der Gewichtung multipliziert.
                        '''

    TEXT = '''

        Kriterium
        Gewichtung
        Investions- und Betrlebskosten
        50 %
        Fachliche Kompetenz
        30 %
        Firma: erstes Relerenzprojekt 17.50%
        Firma: zweites Referenzprojckt 17,50%
        Schlüsselperson: erstes Referenzprojek 17.50%

        Schlüsselperson: zweites Referenzprojekl 17.50%
        Liefertist 20 %
        Interventionszeit 10%
        Technische Kriterien
        20 %
        Garantierte Wartung 40%
        Anpassungen Hardware 20%
        Arpassungen Software 20%
        Anpassungen Aussenarage 20%

        118017
        Werklieferung
        Seite 6 von 15
        3.9 Bewertung der Zuschlagskriterien Die Zuschlagskriterien gemäss Anforderungen unter 4 werden
        wie folgt bewertet und gewichtet:
        Dogendet sie norme net Zuschlagskrienen (exil, Preis) nach
        d/Angabisewerung@nerager.mtopender Forain
        Dan ween D. Punake abyezogen Lase Beverung), Sar
        Preis sind Minuspunkte möglich. (Deckt eine 50%-Bandbreite der zu erwartenden Preise ab)

        Note bezogen auf Ertilung der bezogen aut Angaben und
        Kriterien
        Ausführung

        keine Angaben
        keine Angaben
        1
        unbrauchbar
        unbrauchbare Angaben
        2
        ungenügend
        ungenügender Bezug auf ausgeschriebene Arbeiten
        3
        genügend
        qualitativ genügend, Mindestanforderungen werden knapp erfüllt

        gut bis sehr gut
        qualitativ gut bis sehr gut

        ausgezeichnet
        qualitativ ausgezeichnet, hohe Innovation
        01.11.2000
        118017 K037 Ausschreibungsunterlagen

    '''
    chunk = 'Zuschlagskriterien'
    example = 'Preis'
    COMMAND = B_INST + f'''  Ich gebe dir einen Auszug aus einer Ausschreibung. 
                            Extrahiere folgende Informationen, sofern diese vorhanden sind: {chunk}. 
                            Setze vor und hinter jedem {chunk}, das du extrahierst, diese Zeichenkombination %&#. 
                            Zum Beispiel, wenn {example} als {chunk} genannt wird, gibst du folgendes aus: %&#{example}%&# 
                            WICHTIG: Gebe als Antwort nur die {chunk} und sage sonst nichts weiter! 
                            Hier ist der Auszug:\n\n''' + TEXT + ' ' + E_INST

    COMMAND = B_INST + f"""  Ich gebe dir einen Auszug aus einer Ausschreibung.
                            Hier ist der Auszug: \n\n {TEXT} \n\n
                            Welche Gewichtung wird für das Zuschlagskriterieum 'Anpassungen Aussenarage' angegeben?
                            Falls eine Gewichtung genannt wird, gib nur die Gewichtung an und sag nichts weiter.
                            Falls keine Gewichtung genannt wird, gibt folgendes Aus: 'Ich habe nichts gefunden'. 
                            """ + E_INST

    COMMAND = PROMPT_1 + TEXT + ' ' + E_INST

    result = llm_handler.text_gen_pipeline(COMMAND)[0]["generated_text"]

    print(result)

    # result = json_extractor(result)
    # with open("result.json", "w") as f:
    # js.dumps(result, f, ensure_ascii=False)
