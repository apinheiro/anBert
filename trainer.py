from argparse import ArgumentParser
import torch

from transformers import AutoModelForMaskedLM, BertTokenizer
from dataset import AnBertDataset
from param import parseArguments
import logging

# Definindo log para o script
# ############################

def getLogger(args: ArgumentParser) -> logging.Logger:
    logging.basicConfig(datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO if args.verbose_logging else logging.WARN,
                            format='########\n %(asctime)s: %(message)s')
    return logging.getLogger('monitor')

def getModel(args: ArgumentParser):
    log.info("Carregando modelo do Bert {0}.".format(args.bert_model))
    model = AutoModelForMaskedLM.from_pretrained(args.bert_model)
    
    device = "cpu"
    if not args.no_cuda:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    
    log.info("Baixnado tokenizer.")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    return model, tokenizer
    
# ###########################
# Programa principal
# ###########################

if __name__ == "__main__":
    args = parseArguments()
    log = getLogger(args)

    model, tokenizer = getModel(args)
    
    # Verificando se é um dataset a ser informado ou se é um diretório com arquivos.
    if args.train_dataset is None: 
        log.info("Carregando o dataset a partir do diretório {0}.".format(args.train_path))
        ads = AnBertDataset(tokenizer, 
                            path = args.train_path, 
                            block_size= args.max_seq_length, 
                            file=args.train_file)
        ads.load_dataset()
    else:
        ads = AnBertDataset(tokenizer, block_size=args.max_seq_length)
        ads.load_dataset_file(args.train_dataset)

    log.info("Gerando o dataset para modelos do tipo Label Masked.")
    tokenized_samples = ads.getLabelMaskedDataset(targets=['train','test'])
    
    