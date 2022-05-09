""" Treinamento do modelo

Este arquivo diz respeito ao processo de treinamento de um modelo BERT.
"""

#import re
from dataset import AnBertDataset
from transformers import AutoModelForMaskedLM, Trainer, BertTokenizer, DataCollatorForLanguageModeling, TrainingArguments
#import tensorflow as tf
import numpy as np
#from pathlib import Path


from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk, torch, logging, time



def validate(ds, model, batch_size):
    accuracy = 0.0
    f1score = 0.0
    recall = 0.0
    loss = 0.0
    precision = 0.0
    
    t0 = time.time()
    
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
        loss += result.loss

        labels = b_labels.to('cpu').numpy().flatten()
        preds = np.argmax(logits, axis=-1).flatten()
        pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted',zero_division=0)
        acc = accuracy_score(labels, preds)
        
        f1score += f1
        recall += rc
        precision += pr
        accuracy += acc
        
    return {
        'eval_accuracy': accuracy/len(validation_dl),
        'eval_f1': f1score/len(validation_dl),
        'eval_precision': precision/len(validation_dl),
        'eval_recall': recall/len(validation_dl),
        'eval_loss': loss/len(validation_dl),
        'eval_runtime': (time.time()-t0)
    }
    
def print_validate(eval):
    
    results = ["Tempo total de validação: {:.3f}s .",
               "Acurácia: {:.3f}.", "F1 Score: {:.3f}.",
               "Recall: {:.3f}.", "Precisão: {:.3f}."]
    
    log.info("\n".join(results).format(eval["eval_runtime"],eval["eval_accuracy"],eval["eval_f1"],
                                       eval["eval_recall"],eval["eval_precision"]))

def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.argmax(-1).flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
if __name__ == '__main__':
    
    logging.basicConfig(datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        format='########\n %(asctime)s: %(message)s')
    
    log = logging.getLogger('monitor')

    log.info("Baixando o Punkt para separar frases.")
    
    parser = ArgumentParser()
    
    modelArguments(parser)
    
    
    nltk.download('punkt')
    args = getParametros()
     
    log.info("Carregando modelo do Bert {0}.".format(args.bert_model))
    model = AutoModelForMaskedLM.from_pretrained(args.bert_model)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
  
    log.info("Baixnado tokenizer.")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    
    if args.train_dataset is None: 
        log.info("Carregando o dataset a partir do diretório {0}.".format(args.train_path))
        ads = AnBertDataset(tokenizer, path = args.train_path, block_size= args.max_seq_length, file=args.train_file)
        ads.load_dataset()
    else:
        ads = AnBertDataset(tokenizer, block_size=args.max_seq_length)
        ads.load_file(args.train_dataset)

    log.info("Gerando o dataset para modelos do tipo Label Masked.")
    tokenized_samples = ads.getLabelMaskedDataset(target='train')
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # @TODO: Trocar este pedaço de código pela outra forma de treinamento.
    # Conforme está no caderno do google colab
    

    # Show the training loss with every epoch
    logging_steps = len(tokenized_samples["train"]) // args.batch_size
    model_name = args.bert_model.split("/")[-1]
    
    training_args = TrainingArguments(
        output_dir=f"{model_name}-finetuned-an",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.per_gpu_train_batch_size,
        fp16=args.fp16,
        num_train_epochs=args.num_train_epochs,
        save_strategy='no',
        logging_steps=logging_steps
    )
    
    if args.do_train or args.do_eval:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_samples["train"],
            eval_dataset=tokenized_samples["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        if args.do_train:
            log.info("Preparando o treinamento.")
            
            log.info("Inicializando o treinamento.")
            try:
                trainer.train()
            except:
                print("")
            
            #log.info("Finalizando o treinamento.")
            #validate(trainer.evaluate())
    
        if args.do_eval:
            log.info("iniciando a validação.")
            ads.block_size = args.eval_max_seq_length
            tokenized_samples = ads.getLabelMaskedDataset(target='train')
            print_validate(validate(tokenized_samples['validate'],model,args.eval_batch_size))
