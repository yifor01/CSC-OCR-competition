import os,sys
import json
import yaml
import cv2
import numpy as np
from glob import glob
from img_aug import sharpen, modify_contrast_and_brightness2,dilation
import matplotlib.pyplot as plt

import logging as logger
logger.basicConfig(
        level=logger.INFO,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M')

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../PaddleOCR/')))

import paddle
from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import init_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program
import random


def seed_everything(seed=42):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
        

class Paddlepreditor(object):
    def __init__(self,config_path,pretrain_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.global_config = self.config['Global']
        #self.global_config['character_dict_path'] = self.global_config['character_dict_path'][:]
        self.global_config['character_dict_path'] = '../PaddleOCR/' + self.global_config['character_dict_path'][:]
        self.global_config['pretrained_model']    = pretrain_path
        self.global_config['load_static_weights'] = False
        self.global_config['checkpoints'] = None

        char_file = self.global_config['character_dict_path']
        assert os.path.isfile(char_file) , f'character_dict_path {char_file} not correct!'
        
        self.post_process_class = build_post_process(self.config['PostProcess'],self.global_config)

        if hasattr(self.post_process_class, 'character'):
            char_num = len(getattr(self.post_process_class, 'character'))
            if self.config['Architecture']["algorithm"] in ["Distillation",]:  # distillation model
                for key in self.config['Architecture']["Models"]:
                    self.config['Architecture']["Models"][key]["Head"][
                        'out_channels'] = char_num
            else:  # base rec model
                self.config['Architecture']["Head"]['out_channels'] = char_num

        self.model = build_model(self.config['Architecture'])
        init_model(self.config, self.model)
        
        seed_everything()
        
        transforms = []
        for op in self.config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name in ['RecResizeImg']:
                op[op_name]['infer_mode'] = True
            elif op_name == 'KeepKeys':
                if self.config['Architecture']['algorithm'] == "SRN":
                    op[op_name]['keep_keys'] = [
                        'image', 'encoder_word_pos', 'gsrm_word_pos',
                        'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                    ]
                else:
                    op[op_name]['keep_keys'] = ['image']
            transforms.append(op)
        self.global_config['infer_mode'] = True

        self.ops = create_operators(transforms, self.global_config)
        self.model.eval()
    
    def extra_param(self,batch):
        encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
        gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
        gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
        gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)
        others = [
            paddle.to_tensor(encoder_word_pos_list),
            paddle.to_tensor(gsrm_word_pos_list),
            paddle.to_tensor(gsrm_slf_attn_bias1_list),
            paddle.to_tensor(gsrm_slf_attn_bias2_list)
        ]
        return others
    
    def image_pred(self,img_path,show=False,force180=False):
        ''' Given img path to predict'''
        with open(img_path, 'rb') as f:
            img = f.read()
            data = {'image': img}
        
        batch = transform(data, self.ops)
        
        if force180:
            batch[0][0] = cv2.rotate(batch[0][0],cv2.ROTATE_180)
        
        seed_everything()
        
        images = np.expand_dims(batch[0], axis=0)
        images = paddle.to_tensor(images)
        
        if self.config['Architecture']['algorithm'] == "SRN":
            others = self.extra_param(batch)
            preds = self.model(images, others)
        else:
            preds = self.model(images)
        post_result = self.post_process_class(preds)
        if show:
            print(f"File   : {img_path}")
            print(f"predict: {post_result}")
            plt.imshow(batch[0][0],cmap='gray')
            plt.show()
        return post_result[0]
    
    def super_image_pred(self,img_path,methods=     ['orig','force180','blur','sharpen','bright','contrast','contrast+bright','dilation'],show=False):
        ''' Given img path to predict'''
        with open(img_path, 'rb') as f:
            img = f.read()
            data = {'image': img}        
        
        cand_list = {}
        for method in methods:
            batch = transform(data.copy(), self.ops)
            if method=='orig':
                pass
            elif method=='force180':
                batch[0][0] = cv2.rotate(batch[0][0],cv2.ROTATE_180)
            elif method=='blur':
                batch[0][0] = cv2.blur(batch[0][0],(5,5))
            elif method=='sharpen':
                batch[0][0] = sharpen(batch[0][0],100)
            elif method=='bright':
                batch[0][0] = modify_contrast_and_brightness2(batch[0][0],100,0)
            elif method=='contrast':
                batch[0][0] = modify_contrast_and_brightness2(batch[0][0],0,100)
            elif method=='contrast+bright':
                batch[0][0] = modify_contrast_and_brightness2(batch[0][0],100,100) 
            elif method=='dilation':
                batch[0][0] = dilation(batch[0][0],4,4)    
                
            seed_everything()
                
            images = np.expand_dims(batch[0], axis=0)
            images = paddle.to_tensor(images)

            if self.config['Architecture']['algorithm'] == "SRN":
                others = self.extra_param(batch)
                preds = self.model(images, others)
            else:
                preds = self.model(images)
            post_result = self.post_process_class(preds)
            if show:
                print(post_result)
                plt.imshow(batch[0][0],cmap='gray')
                plt.show()
                
            cand_list[method] = post_result
        return cand_list

    def super_image_pred_1105(self,img_path,methods=['orig','force180','blur','sharpen','bright','contrast','contrast+bright','dilation'],show=False):
        ''' Given img path to predict'''
        with open(img_path, 'rb') as f:
            img = f.read()
            data = {'image': img}        
        
        cand_list = {}
        for method in methods:
            batch = transform(data.copy(), self.ops)
            if method=='orig':
                pass
            elif method=='force180':
                batch[0][0] = cv2.rotate(batch[0][0],cv2.ROTATE_180)
            elif method=='blur':
                batch[0][0] = cv2.blur(batch[0][0],(5,5))
            elif method=='sharpen':
                batch[0][0] =sharpen(batch[0][0],100)
            elif method=='bright':
                batch[0][0] =modify_contrast_and_brightness2(batch[0][0],100,0)
            elif method=='contrast':
                batch[0][0] =modify_contrast_and_brightness2(batch[0][0],0,100)
            elif method=='contrast+bright':
                batch[0][0] =modify_contrast_and_brightness2(batch[0][0],100,100)             
            elif method=='dilation':
                batch[0][0] =dilation(batch[0][0],4,4)    
                
            seed_everything()
                
            images = np.expand_dims(batch[0], axis=0)
            images = paddle.to_tensor(images)

            if self.config['Architecture']['algorithm'] == "SRN":
                others = self.extra_param(batch)
                preds = self.model(images, others)
            else:
                preds = self.model(images)
            post_result = self.post_process_class(preds)
            if show:
                print(post_result)
                plt.imshow(batch[0][0],cmap='gray')
                plt.show()
            #cand_list.extend(post_result)
            cand_list[method] = post_result
        return cand_list

    
def yi_merge(res_map,methods=['orig','force180','blur','contrast']):
    _pred = {}
    for _id,_c in res_map.items():
        tmp = []
        for method in methods:
            tmp.extend(_c[method])
        tmp = sorted(tmp,key=lambda x : x[1],reverse=True)
        _pred[_id] = tmp[0][0]
    return _pred


def mix_score2(true_map,pred_map,verbose=True):
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
    valid_ans = pd.read_csv('../csv_output/csv_valid_chk_1012.csv')[['id','label']]
    valid_ans = {k:v for k,v in zip(valid_ans['id'],valid_ans['label'])}
    return mix_score2(valid_ans,pred_map,verbose=verbose)
