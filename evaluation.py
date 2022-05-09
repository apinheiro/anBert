from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import nltk, torch, logging, time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def _getDataloader(ds, parser):
    tds = TensorDataset(torch.tensor(ds["input_ids"]), 
                        torch.tensor(ds["attention_mask"]), 
                        torch.tensor(ds["labels"]))
    
    validation_dl = DataLoader(
            tds, # The validation samples.
            sampler = SequentialSampler(tds), # Pull out batches sequentially.
            batch_size = parser.eval_batch_size, # Evaluate with this batch size.
            shuffle=True
        )
    for i in validation_dl:
        yield i
    

def validate(ds, model, parser, device):
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
            batch_size = parser.eval_batch_size # Evaluate with this batch size.
        )
    
    for batch in _getDataloader(ds,parser):
        
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