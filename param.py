from argparse import  ArgumentParser

def _modelArguments(parser: ArgumentParser):
    """ Modelo Arguments
    
    Basics arguments to train and evaluate the model.

    Args:
        parser (ArgumentParser): instance of ArgumentParser class.
    """
    
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
    
    parser.add_argument("--eval_max_seq_length", default=128, type=int,
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
    
    parser.add_argument("--batch_size", type=int,
                        help="Tamanho do batch de treinamento.")
    
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Tamanho do batch de treinamento.")

def _baseArguments(parser: ArgumentParser):
    """ Base arguments
    
    Arguments for basic config to training the model and evalueate also.

    Args:
        parser (ArgumentParser): argument parse class instance.
    """
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data "
                        "processing will be printed.")
    
    parser.add_argument('--override_cache',
                        action='store_true',
                        help='Override feature caches of input files.')
 
def _trainingArguments(parser: ArgumentParser):
    """Training Arguments
    
    Training of 

    Args:
        parser (ArgumentParser): instance of ArgumentParser class.
    """
    # Training related
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_path", default=None,
                        type=str, help="Path with train files.")
    parser.add_argument("--train_dataset",type=str, help="Path with pre-trained dataset.")
    parser.add_argument("--train_file", type=str, help="File to single training.")
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
    
def _infraArguments(parser: ArgumentParser):
    """ Infraestructure Arguments
    
    Arguments to set infra params and normalizations.

    Args:
    parser (ArgumentParser): instance of ArgumentParser class.
    """
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of"
                        " 32-bit")
    
def _evalArguments(parser: ArgumentParser):
    """Evaluation Arguments

    Args:
        parser (ArgumentParser): instance of ArgumentParser class.
    """
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the test set.")
        
def parseArguments():
    """ Arguments
    
    To configure a Bert Training and Evaluation process, this function returns the 
    argument parser to use in the parser.

    Returns:
        ArgumentParser: instance of ArgumentParser class.
    """
    parser = ArgumentParser()
    _modelArguments(parser)
    _trainingArguments(parser)
    _evalArguments(parser)
    _infraArguments(parser)
    _baseArguments(parser)

    return parser.parse_args()