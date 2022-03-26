import subprocess
import psutil, logging, time
from datetime import datetime

modelo = "neuralmind/bert-base-portuguese-cased"

commands = ["python3", "run_trainer.py",
            "--bert_model={0}".format(modelo),
            "--train_dataset=./machado/traducao",
            "--do_train"]

FORMAT = '%(message)s'
file = datetime.now().strftime('monitor_%Y_%m_%d.log')
logging.basicConfig(filename=file,
                        filemode='a',
                        format=FORMAT,
                        level=logging.INFO)

monitor = logging.getLogger('monitor')

for batch in [4,8,16,32,64,128,256]:
    for block in [16,32,64,128,256]:
        
        este_comando = commands + ["--block_size={0}".format(block), "--batch_size={0}".format(batch)]
        
        print("Executando o treinamento\n ********")
        print("Tokens por sentença: {0}.".format(block))
        print("Sentenças por Bloco de treinamento: {0}.".format(batch))
        print("Modelo pré-treinado: {0}.\n".format(modelo))
        
        
        monitor.info("Executando o treinamento\n ********")
        monitor.info("Tokens por sentença: {0}.".format(block))
        monitor.info("Sentenças por Bloco de treinamento: {0}.".format(batch))
        monitor.info("Modelo pré-treinado: {0}.\n".format(modelo))
        
        with subprocess.Popen(commands, stdout=subprocess.PIPE, stderr=None) as running:
            monitor.info(running.communicate()[0].decode('utf-8'))
        
        print("Treinamento finalizado.\n*************")