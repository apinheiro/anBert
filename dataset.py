import string
import torch, os, re, numpy
from pathlib import Path
from nltk.tokenize import sent_tokenize

class AnBertDataset(object):
    """AnBertDataset - Gerador de Dataset
    
    Esta classe tem por finalidade preparar o corpus para gerar as sentenças a 
    serem treinadas.
    
    A classe também produz um dataset com os textos para treinamento, avaliação e validação.

    """
    def __init__(self, path = None, file = None):
        """ 
        Construtor da classe AnBertDataset

        Args:
            path (str, optional): Diretório contendo os arquivos a serem treinados.
            file (str, optional): Arquivo para treinamento único.
        """
        self.path = path
        self.file = file
        self.dataset = {"text": [], "label":[]}

    def load_dataset(self, train_size = 0.8 , test_size = 0.1, validate_size = 0.1):    
        # normalizando a distribuição dos textos.
        total = train_size + test_size, validate_size
        train_size = train_size / total
        test_size = test_size / total
        validate_size = validate_size / total
        
        # Verificando se os arquivos estão corretos.
        path = False if self.path is None else True
        if not path:
            assert self.file == None , "Indique um diretório ou arquivo para treinamento."
            assert os.path.isfile(self.file) , "Arquivo não encontrado."
        
        assert os.path.isdir(self.path) , "Diretório não localizado."
            
        # Lendo o conteúdo do texto informado ou dos textos de um diretório.
        textos = self.__load_path() if path else self.__load_file()
        
        sentences = []
        # Separando os textos em sentenças
        _ = [sentences.append(sent_tokenize(t)) for t in textos]
        
        train, evaluate, validate = AnBertDataset.__split_dataset(sentences, train_size, test_size, validate_size)
        
        self.__populate_dataset(train, evaluate, validate)
    
    def __load_path(self):
        files = [str(x) for x in Path(self.args.valid_path).glob("**/*.txt")]
        sentences = []
        
        for file in files:
            sentences.append(file = self.__load_file(file))
        return sentences
        
    def __load_file(self, file = None):  
        file = file if file is not None else self.file
        sentences = []
        
        # Lendo todas as linhas do documento em que a linha não seja vazia.
        with open(file, encoding="utf-8") as f:
            sentences = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        # Removendo os textos que contém apenas o número romano ou início de capítulo.
        sentences = [s for s in sentences if not (self.__validation_roman_numbers(s) or s.lower().startswith('capítulo'))]
        return sentences
        
    def __validation_roman_numbers(self, string):
        # Searching the input string in expression and
        # returning the boolean value
        return bool(re.search(r"^M{0,3}(C[M|D]|D?C{0,3})(X[C|L]|L?X{0,3})(I[X|V]|V?I{0,3})$",string))
    
    def __populate_dataset(self, train, test, validate):
        output = [[t, "train"] for t in train] + [[t, "test"] for t in test] + [[t, "validate"] for t in validate] 
        numpy.random.shuffle(output)
        
        _ = [[self.dataset["text"].append(o[0]), self.dataset["label"].append(o[1])] for o in output]   

    
    def __split_dataset(dataset, train_size, eval_size, validate_size):
        size = len(dataset)
        
        x_train = dataset[:int(size * train_size)]
        x_eval = dataset[int(size * train_size): int(size * (train_size + eval_size))]
        x_validate = dataset[int(size * (train_size + eval_size)):int(size * (train_size + eval_size + validate_size))]
        
        return x_train, x_eval, x_validate