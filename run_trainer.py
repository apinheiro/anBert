""" Treinamento do modelo

Este arquivo diz respeito ao processo de treinamento de um modelo BERT.
"""

from dataset import AnBertDataset
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import tensorflow as tf
from pathlib import Path

import nltk, datasets

if __name__ == '__main__':
    
    nltk.download('punkt')
    
    files = [str(x) for x in Path('./machado').glob("**/*.txt")]
    
    
    #dataset = datasets.Dataset.from_file('/machado')
    #print(dataset[0])
    
    # sentences = []
    # for f in files:
    #     with open(f, encoding="utf-8") as f:
    #         sentences += [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
    # # Removendo os textos que contém apenas o número romano ou início de capítulo.
    # sentences = [s for s in sentences if not (AnBertDataset.validation_roman_numbers(s) or s.lower().startswith('capítulo '))]
    
    # ds = datasets.Dataset.from_dict(mapping={"text" : sentences})
    
    # print(ds)
    
    model_checkpoint = "distilbert-base-uncased"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    ads = AnBertDataset(tokenizer, path = './machado/traducao')
    ads.load_dataset()

    tokenized_samples = ads.dataset["train"][:30]

    for idx, sample in enumerate(tokenized_samples["input_ids"]):
        print(f"'>>> Review {idx} length:'" + tokenizer.decode(sample))