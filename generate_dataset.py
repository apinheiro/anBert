from dataset import AnBertDataset
from transformers import  BertTokenizer
import nltk 

if __name__ == "__main__":

    nltk.download('punkt')
    t = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    data = AnBertDataset(tokenizer=t,path="./machado/romance")
    data.load_dataset()
    
    s = [len(h) for h in data.dataset["train"]["input_ids"]]
    
    print("número de tokens:")
    
    
    data.save_file()
    
    data = None
    data = AnBertDataset(tokenizer=t)
    data.load_file()
    
    print(data.dataset)
