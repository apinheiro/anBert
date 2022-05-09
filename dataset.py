import os, re, datasets
from pathlib import Path
from nltk.tokenize import sent_tokenize

from transformers.data import default_data_collator
from transformers import DataCollatorForLanguageModeling

class AnBertDataset(object):
    """AnBertDataset - Gerador de Dataset
    
    Esta classe tem por finalidade preparar o corpus para gerar as sentenças a 
    serem treinadas.
    
    A classe também produz um dataset com os textos para treinamento, avaliação e validação.

    """
    def __init__(self, tokenizer, path = None, file = None, block_size = 32):
        """ 
        Construtor da classe AnBertDataset

        Args:
            path (str, optional): Diretório contendo os arquivos a serem treinados.
            file (str, optional): Arquivo para treinamento único.
        """
        self.tokenizer = tokenizer
        self.path = path
        self.file = file
        self.block_size = block_size
        self.dataset = None
    
    def save_dataset_file(self, datasetName = "machado.ds"):
        self.dataset.save_to_disk(datasetName)
        
    def load_dataset_file(self, datasetName = "machado.ds"):
        self.dataset = datasets.DatasetDict.load_from_disk(datasetName)

    def load_dataset(self, train_size = 0.8 , test_size = 0.1, validate_size = 0.1):    
        # normalizando a distribuição dos textos.
        total = train_size + test_size + validate_size
        train_size = train_size / total
        test_size = test_size / total
        validate_size = validate_size / total
        
        # Verificando se os arquivos estão corretos.
        path = False if self.path is None else True
        if not path:
            assert self.file != None , "Indique um diretório ou arquivo para treinamento."
            assert os.path.isfile(self.file) , "Arquivo não encontrado."
        else:
            assert os.path.isdir(self.path) , "Diretório não localizado."
            
        # Lendo o conteúdo do texto informado ou dos textos de um diretório.
        textos = self.__load_path() if path else self.__load_file()
        
        sentences = []
        # Separando os textos em sentenças
        for t in textos:
            sentences += sent_tokenize(t)
        
        train, evaluate, validate = AnBertDataset.__split_dataset(sentences, train_size, test_size, validate_size)   
        self.__populate_dataset(train, evaluate, validate)
    
    def __load_path(self):
        files = [str(x) for x in Path(self.path).glob("**/*.txt")]
        sentences = []
        
        for file in files:
            sentences += self.__load_file(file = file)
        return sentences
        
    def __load_file(self, file = None):  
        file = file if file is not None else self.file
        sentences = []
        
        # Lendo todas as linhas do documento em que a linha não seja vazia.
        with open(file, encoding="utf-8") as f:
            text = f.read()
            text = text.replace("\n\n","#.#").replace("\n"," ").replace("#.#","\n")
            
            sentences += [line for line in text.splitlines() if (len(line) > 0 and not line.isspace())]
        
        # Removendo os textos que contém apenas o número romano ou início de capítulo.
        sentences = [str(s) for s in sentences if not (AnBertDataset.validation_roman_numbers(s) or s.lower().startswith('capítulo '))]
        return sentences
        
    def validation_roman_numbers(string):
        # Searching the input string in expression and
        # returning the boolean value
        return bool(re.search(r"^M{0,3}(C[M|D]|D?C{0,3})(X[C|L]|L?X{0,3})(I[X|V]|V?I{0,3})$",string.strip().upper()))
    
    def __populate_dataset(self, train, test, validate):
        ds_train = datasets.Dataset.from_dict(mapping = {"text": train})
        ds_test = datasets.Dataset.from_dict(mapping = {"text": test})
        ds_validate = datasets.Dataset.from_dict(mapping = {"text": validate})
        
        self.dataset = datasets.DatasetDict({"train": ds_train, "test": ds_test, "validate": ds_validate})
        
        self.dataset = self.dataset.map(
            self.__tokenize_function, batched=True, remove_columns=["text"]
        )
        
    def __split_dataset(dataset, train_size, eval_size, validate_size):
        size = len(dataset)
        
        x_train = dataset[:int(size * train_size)]
        x_eval = dataset[int(size * train_size): int(size * (train_size + eval_size))]
        x_validate = dataset[int(size * (train_size + eval_size)):int(size * (train_size + eval_size + validate_size))]
        
        return x_train, x_eval, x_validate
    
    def getLabelMaskedDataset(self, targets = []):
        self._targets = targets
        return self.dataset.map(self.__group_texts, batched=True)
    
    def getNextSentenceDataset(self, block_size = 32):
        return True
    
    def __group_texts(self, examples):
        
        meus_exemplos = {k: examples[k] for k in examples.keys() if k in self._targets}
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len
        
        resulta = {k : [] for k in meus_exemplos.keys()}
        
        for k in list(examples.keys()):
            for t in examples[k]:
                temp = [t[i:i+self.block_size] for i in range(0,len(t), self.block_size)]
                temp = [j + [0]*(self.block_size - len(j)) for j in temp]
                resulta[k] += temp
        
        resulta["labels"] = resulta["input_ids"].copy()
        return resulta
    
    def __tokenize_function(self, examples):
        result = self.tokenizer(examples["text"])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result