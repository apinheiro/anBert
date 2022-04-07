import subprocess
import psutil, logging, time
from datetime import datetime
import torch

from dataset import AnBertDataset

modelo = "neuralmind/bert-base-portuguese-cased"

commands = ["python3", "run_trainer.py",
            "--bert_model={0}".format(modelo),
            "--train_dataset=./machado.ds",
            "--do_train",
            "--per_gpu_train_batch_size=1"]

FORMAT = '%(message)s'
file = datetime.now().strftime('monitor_%Y_%m_%d.log')
logging.basicConfig(filename=file,
                        filemode='a',
                        format=FORMAT,
                        level=logging.INFO)

monitor = logging.getLogger('monitor')

# Verificando o tipo de ambiente de treinamento.
if torch.cuda.is_available():      
    device = torch.device("cuda")
    monitor.info('There are %d GPU(s) available.' % torch.cuda.device_count())
    monitor.info('We will use the GPU:', torch.cuda.get_device_name(0))
    
    #gpu = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE)
    #monitor.log(gpu.communicate()[0].decode('utf-8'))

else:
    monitor.info('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


for batch in [256,128,64,32,16]:
    for block in [256,128,64,32,16]:
        
        if batch * block > 4097:
            continue
        
        este_comando = commands + ["--max_seq_length={0}".format(block), "--batch_size={0}".format(batch)]
        
        print("Executando o treinamento\n ********")
        print("Tokens por sentença: {0}.".format(block))
        print("Sentenças por Bloco de treinamento: {0}.".format(batch))
        print("Modelo pré-treinado: {0}.\n".format(modelo))
        

        with subprocess.Popen(este_comando, stdout=subprocess.PIPE, stderr=None) as running:
            monitor.info(running.communicate()[0].decode('utf-8'))
        
        print("Treinamento finalizado.\n*************")
