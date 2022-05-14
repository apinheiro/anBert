import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch, time

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.argmax(-1).flatten()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def print_validate(eval, logging):
    results = ["Tempo total de validação: {:.3f}s .",
               "Acurácia: {:.3f}.", "F1 Score: {:.3f}.",
               "Recall: {:.3f}.", "Precisão: {:.3f}."]
    
    logging.info("\n".join(results).format(eval["eval_runtime"],eval["eval_accuracy"],eval["eval_f1"],
                                       eval["eval_recall"],eval["eval_precision"]))

def calcule_validate(ds, model, batch_size, device='cpu'):
    accuracy = 0.0
    f1score = 0.0
    recall = 0.0
    loss = 0.0
    precision = 0.0
    
    t0 = time.time()
    
    tds = TensorDataset(torch.tensor(ds["input_ids"]), 
                        torch.tensor(ds["attention_mask"]), 
                        torch.tensor(ds["labels"]))
    
    validation_dl = DataLoader(
            tds, # The validation samples.
            sampler = SequentialSampler(tds), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )
    
    for step, batch in enumerate(validation_dl):
    
        # Progress update every 40 batches.
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(validation_dl), elapsed))
        
        b_labels = batch[2].to(device)
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        
        with torch.no_grad():
            result = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask,
                        labels=b_labels,
                        return_dict=True)

        
        logits = result.logits.detach().cpu().numpy()
        loss += result.loss

        labels = b_labels.to('cpu').numpy().flatten()
        preds = np.argmax(logits, axis=-1).flatten()
        pr, rc, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted',zero_division=0)
        acc = accuracy_score(labels, preds)
        
        f1score += f1
        recall += rc
        precision += pr
        accuracy += acc
        
    return {
        'eval_accuracy': accuracy/len(validation_dl),
        'eval_f1': f1score/len(validation_dl),
        'eval_precision': precision/len(validation_dl),
        'eval_recall': recall/len(validation_dl),
        'eval_loss': loss/len(validation_dl),
        'eval_runtime': (time.time()-t0)
    }