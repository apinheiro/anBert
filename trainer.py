from argparse import ArgumentParser
import torch

from transformers import AutoModelForMaskedLM, BertTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from dataset import AnBertDataset
from param import parseArguments
import logging
import psutil
from metrics import *

# Definindo log para o script
# ############################

def getDevice(args):
    device = "cpu"
    if not args.no_cuda:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device

def getLogger(args: ArgumentParser):
    logging.basicConfig(datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO if args.verbose_logging else logging.WARN,
                            format='########\n %(asctime)s: %(message)s')
    return logging.getLogger('monitor')

def getModel(args: ArgumentParser, log):
    log.info("Carregando modelo do Bert {0}.".format(args.bert_model))
    model = AutoModelForMaskedLM.from_pretrained(args.bert_model)
    model  = model.to(getDevice(args))
    
    log.info("Baixnado tokenizer.")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    return model, tokenizer
    
def getTrainParameters(args, model_name):
    return TrainingArguments(
        output_dir="tmp/output",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay= args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        save_strategy='no',
        logging_steps=logging_steps, 
        seed = args.seed,
        dataloader_num_workers=psutil.cpu_count(),
       #disable_tqdm=True
    )
    
# ###########################
# Programa principal
# ###########################

if __name__ == "__main__":
    args = parseArguments()
    log = getLogger(args)

    model, tokenizer = getModel(args, log)
    
    # Verificando se é um dataset a ser informado ou se é um diretório com arquivos.
    if args.train_dataset is None: 
        log.info("Carregando o dataset a partir do diretório {0}.".format(args.train_path))
        ads = AnBertDataset(tokenizer, 
                            path = args.train_path, 
                            block_size= args.max_seq_length, 
                            file=args.train_file)
        ads.load_dataset()
    else:
        log.info("Carregando dataset previamente cadastrado.")
        ads = AnBertDataset(tokenizer, block_size=args.max_seq_length)
        ads.load_dataset_file(args.train_dataset)
    
    
    log.info("Gerando o dataset para modelos do tipo Label Masked.")
    tokenized_samples = ads.getLabelMaskedDataset(targets=['train','test'], test_seq_length = args.eval_max_seq_length, valid_seq_length = args.eval_max_seq_length)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
    
    logging_steps = len(tokenized_samples["train"]) // args.batch_size
    model_name = args.bert_model.split("/")[-1]
    
    if args.do_train or args.do_eval:
        trainer = Trainer(
            model=model,
            args=getTrainParameters(args,model_name),
            train_dataset=tokenized_samples["train"],
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        if args.do_train:
            log.info("Inicializando o treinamento.")
            trainer.train()
            
            log.info("Finalizando o treinamento.")
        if args.do_eval:
            print_validate(
                calcule_validate(tokenized_samples['test'],
                                 model,
                                 args.eval_batch_size, 
                                 getDevice(args)),
                log)
        trainer.save_model(output_dir=f"{model_name}-an-bs{args.batch_size}-sl{args.max_seq_length}")

    