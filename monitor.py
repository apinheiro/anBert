import psutil, logging, time
from datetime import datetime


def log_cpu(psu):
    try:
        cpu = p.cpu_times()
        return [cpu.user, cpu.system, cpu.children_user, cpu.iowait, psu.num_threads()]
    except:
        return [cpu.user, cpu.system, cpu.children_user, 0.0, psu.num_threads()]

def log_memory(psu):
    try:
        mem = psu.memory_info()
        return [mem.rss, mem.vms, mem.shared, mem.data]
    except:
        return [mem.rss, mem.vms, 0.0, 0.0]

def log_io(psu):
    try:
        io = p.io_counters()
        return [io.read_count, io.write_count, io.read_bytes, io.write_bytes]
    except:
        return [0,0,0,0]

if __name__ == '__main__':
    FORMAT = '%(asctime)s,%(message)s'
    file = datetime.now().strftime('monitor_%Y_%m_%d.csv')
    logging.basicConfig(filename=file,
                            filemode='a',
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG)

    monitor = logging.getLogger('monitor')
    with open(file,'w') as f:
        f.write('data,user,system,children_user,iowait,num_threads,rss_mem, vms_mem,shared_mem, data_mem, read_count_io,write_count_io,read_bytes_io,write_bytes_io')
    
    
    while True:
        try:
            itens = psutil.pids()
            
            for i in itens:
            
                p = psutil.Process(i)
                if 'run_trainer.py' in p.cmdline():
                    while p.status() == 'running':
                        a = datetime.now()
                        saida = []
                        saida += log_cpu(p)
                        saida += log_memory(p)
                        saida += log_io(p)
                        monitor.info(",".join([str(t) for t in saida]))
                        time.sleep(1.0 - ((datetime.now() - a).microseconds/10e8))
        except KeyboardInterrupt:
            print("Programa finalizado com sucesso!")
            break;
        except:
            itens = []