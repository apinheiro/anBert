""" Treinamento do modelo

Este arquivo diz respeito ao processo de treinamento de um modelo BERT.
"""

from dataset import AnBertDataset
from transformers import AutoModelForMaskedLM, Trainer, AutoTokenizer, DataCollatorForLanguageModeling
import tensorflow as tf
from pathlib import Path

import nltk, datasets, math, torch

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
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
  
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    ads = AnBertDataset(tokenizer, path = './machado/traducao')
    ads.load_dataset()
    
    tokenized_samples = ads.getLabelMaskedDataset()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    
    from transformers import TrainingArguments

    batch_size = 16
    
    # Show the training loss with every epoch
    logging_steps = len(tokenized_samples["train"]) // batch_size
    model_name = model_checkpoint.split("/")[-1]

    training_args = TrainingArguments(
        output_dir=f"{model_name}-finetuned-imdb",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        #fp16=True,
        logging_steps=logging_steps,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_samples["train"],
        eval_dataset=tokenized_samples["test"],
        data_collator=data_collator,
    )
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")