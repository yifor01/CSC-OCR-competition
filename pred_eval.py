import pandas as pd
import numpy as np
from Levenshtein import distance

def metric_loss(true_txt,pred_txt):
    if true_txt==pred_txt:
        return [0, 0]
    pred_txt = pred_txt if pred_txt else ''
    max_dist = max(len(true_txt),len(pred_txt))
    loss2 = distance(true_txt,pred_txt) / max_dist
    return [1,loss2]

def mix_score(true_map,pred_map,verbose=True):
    '''
    input: 
        [{'label':'abcd','text':'abc'},....]
    '''
    res = [metric_loss(pred_map.copy().setdefault(_id,' '),t_label) for _id,t_label in true_map.items()]
    term1 = np.sum([x[0] for x in res])
    term2 = np.mean([x[1] for x in res])
    if verbose:
        print(f'error: {term1}, distance:{term2:.4f}, total:{term1+term2:.4f}')
    return term1+term2


def valid_eval2(pred_map,verbose=True):
    '''pred_map: {[id]:[label]} '''
    valid_ans = pd.read_csv('./csv_output/csv_valid_chk_1012.csv')[['id','label']]
    valid_ans = {k:v for k,v in zip(valid_ans['id'],valid_ans['label'])}
    return mix_score(valid_ans,pred_map,verbose=verbose)