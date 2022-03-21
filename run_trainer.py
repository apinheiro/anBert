""" Treinamento do modelo

Este arquivo diz respeito ao processo de treinamento de um modelo BERT.
"""

from dataset import AnBertDataset
from transformers import AutoModelForMaskedLM, Trainer, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments
import tensorflow as tf
import numpy as np
from pathlib import Path
from argparse import Namespace, ArgumentParser

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from sklearn.metrics import accuracy_score, f1_score
import nltk, datasets, math, torch, logging, time

def getParametros():
    
    parser = ArgumentParser()
    # Model and hyperparameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Modelo do BERT para treinamento. Pode ser diretório ou um modelo")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for "
                        "uncased models, False for cased models.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after "
                        "WordPiece tokenization. Sequences longer than this "
                        "will be split into multiple spans, and sequences "
                        "shorter than this will be padded.")
    parser.add_argument("--max_position_embeddings", default=512, type=int,
                    help="Max embeddings to training.")
    parser.add_argument("-attentions_heads", default=12, type=int,
                        help="Número de camadas de atenção.")
    parser.add_argument("--hidden_layer", default=6,type=int,
                        help="Número de camadas hidden")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Tamanho do batch de treinamento.")
    parser.add_argument("--block_size", default=128, type=int,
                        help="Tamanho do bloco de texto para tratamento.")
    # General
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data "
                        "processing will be printed.")
    parser.add_argument('--override_cache',
                        action='store_true',
                        help='Override feature caches of input files.')

    # Training related
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_dataset", default=None,
                        type=str, help="JSON for training.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--classifier_lr',
                        type=float,
                        default=2e-5,
                        help='Learning rate of the classifier and CRF layers.')
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear "
                            "learning rate warmup for. E.g., 0.1 = 10%% "
                            "of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of"
                        " 32-bit")

    # Evaluation related
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the test set.")

    return parser.parse_args()

def validate(ds, model, batch_size):
    accuracy = 0.0
    f1score = 0.0
    
    tds = TensorDataset(torch.tensor(ds["input_ids"]), 
                        torch.tensor(ds["attention_mask"]), 
                        torch.tensor(ds["labels"]))
    
    validation_dl = DataLoader(
            tds, # The validation samples.
            sampler = SequentialSampler(tds), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
    
    for batch in validation_dl:
        
        b_labels = batch[2].to(device)
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        
        
        with torch.no_grad():
            result = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask,
                        labels=b_labels,
                        return_dict=True)

        
        logits = result.logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()
    
        predictions = np.argmax(logits, axis=-1).flatten()
        accuracy += accuracy_score(label_ids.flatten(),predictions)
        f1score += f1_score(label_ids.flatten(), predictions, average='macro')
              
    return accuracy / len(validation_dl), f1score / len(validation_dl)
    
if __name__ == '__main__':
    
    logging.basicConfig(datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        format='########\n %(asctime)s: %(message)s')
    
    log = logging.getLogger('monitor')

    log.info("Baixando o Punkt para separar frases.")
    nltk.download('punkt')
    args = getParametros()
     
    log.info("Carregando modelo do Bert {0}.".format(args.bert_model))
    model = AutoModelForMaskedLM.from_pretrained(args.bert_model)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
  
    log.info("Baixnado tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    
    log.info("Carregando o dataset a partir do diretório {0}.".format(args.train_dataset))
    ads = AnBertDataset(tokenizer, path = args.train_dataset, block_size= args.block_size)
    ads.load_dataset()
    
    log.info("Gerando o dataset para modelos do tipo Label Masked.")
    tokenized_samples = ads.getLabelMaskedDataset()
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    
    # Show the training loss with every epoch
    logging_steps = len(tokenized_samples["train"]) // args.batch_size
    model_name = args.bert_model.split("/")[-1]
    
    if args.do_train:
        log.info("Preparando o treinamento.")
        training_args = TrainingArguments(
            output_dir=f"{model_name}-finetuned-an",
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            fp16=args.fp16,
            logging_steps=logging_steps,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_samples["train"],
            eval_dataset=tokenized_samples["test"],
            data_collator=data_collator,
        )
        log.info("Inicializando o treinamento.")
        trainer.train()
        log.info("Finalizando o treinamento.")
    
        eval_results = trainer.evaluate()
        print(f">>> Loss: {math.exp(eval_results['eval_loss']):.2f}")
    
    t0 = time.time()
    log.info("iniciando a validação.")
    acc, f1 = validate(tokenized_samples['validate'], model, args.batch_size)
    log.info("Finalizando a validação.")
    t1 = time.time()
    
    log.info("Tempo total de validação: {0:.3f}s .".format(t1-t0))
    print("Acurácia: ", acc)
    print("F1 Score: ", f1)