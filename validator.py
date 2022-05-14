
import logging
from dataset import AnBertDataset
from metrics import calcule_validate, print_validate
from param import parseArguments
from trainer import getDevice, getLogger, getModel


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
        
    log.info("iniciando a validação.")
    ads.block_size = args.eval_max_seq_length
    tokenized_samples = ads.getLabelMaskedDataset(targets=['validate'], valid_seq_length = args.eval_max_seq_length)
    print_validate(calcule_validate(tokenized_samples['validate'],model,args.eval_batch_size, getDevice(args)), log)