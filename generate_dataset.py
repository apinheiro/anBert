from dataset import AnBertDataset
from transformers import  BertTokenizer
from param import parseArguments

import nltk 

if __name__ == "__main__":

    args = parseArguments()
    nltk.download('punkt')
    t = BertTokenizer.from_pretrained(args.bert_model)
    data = AnBertDataset(tokenizer=t, path=args.train_path)
    data.load_dataset(test_size=0.7, train_size=0.15, validate_size=0.15)
    
    data.save_dataset_file()
    print(data.dataset)
