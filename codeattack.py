import os, sys
from xmlrpc.client import Boolean
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(parent_dir)

import argparse
import warnings
import json
import yaml
import pandas as pd
from tqdm import tqdm

import torch

from transformers import RobertaConfig, RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM

from get_models import get_roberta_model, get_codet5_model, get_codebert_model, get_graphcodebert_model
from attack import attack
from get_data import get_summarization_data, get_translation_data

from textfooler.utils_textfooler import get_var, get_sim_predictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES']= '1'
warnings.simplefilter(action='ignore', category=FutureWarning)

# torch.cuda.set_device(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Feature(object):
    def __init__(self, input_code, output, bleu=0):
        self.code = input_code
        self.adverse_code = input_code
        self.gold_out = output
        self.pred_out = output
        self.adv_out = output
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []
        self.pred_bleu = bleu
        self.after_attack_bleu = bleu
        self.imp_words = {}

def dump_features(features, output, idx):
    outputs = []
    for feature in features:
        outputs.append({'success': feature.success,
                        'pred_bleu': feature.pred_bleu,
                        'after_attack_bleu': feature.after_attack_bleu,
                        'change': feature.change,
                        'num_word': len(feature.code.split(' ')),
                        'query': feature.query,
                        'changes': feature.changes,
                        'input': feature.code,
                        'adv': feature.adverse_code,
                        'gold_out': feature.gold_out,
                        'pred_out': feature.pred_out,
                        'adv_out': feature.adv_out,
                        'imp_words': feature.imp_words,
                        })
    output_json = os.path.join(output, str(idx)+ '.json')
    json.dump(outputs, open(output_json, 'w'), indent=2)
    print('Finished dump.')
    print(output_json)


def run_attack():

    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_model", type=str, help='Model that attacks')
    # parser.set_defaults(attack_model='codebert')
    parser.add_argument("--victim_model", type=str, help='Model to attack')
    parser.add_argument("--task", type=str, help='task')
    parser.add_argument("--lang", type=str, help='language [java_cs, cs_java, java_small]')
    parser.add_argument("--use_ast", type=int, help='Use AST Constraint or not')
    parser.add_argument("--use_dfg", type=int, help='Use DFG Constraint or not')
    parser.add_argument("--out_dirname", type=str, help="Output directory")
    parser.add_argument("--input_lang", type=str, help="Input Language [Only for Graphcodebert]")
    parser.add_argument("--use_imp", type=int, help='Whether to randomly select imp words or not')
    parser.add_argument("--theta", type=float, help='How many tokens to attack threshold')
    # parser.add_argument("--allow_unk", type=int, help='Allow [UNK] as substitute during bert-attack')

    args = parser.parse_args()
    
    config_path = os.path.join(parent_dir, 'configs/config_translate.yaml')
    config = yaml.safe_load(open(config_path))

    config['victim_model'] = args.victim_model
    # config['task'] = args.task
    # config['lang'] = args.lang
    # config['input_lang'] = args.input_lang
    config['use_ast_constraint'] = args.use_ast
    # config['use_dfg_constraint'] = args.use_dfg
    config['out_dirname'] = args.out_dirname
    # # config['allow_unk'] = args.allow_unk
    config['attack_model'] = args.attack_model
    # config['use_imp_words'] = args.use_imp
    config['theta'] = args.theta

    print(args)
    print(config)

    config_data_path = os.path.join(parent_dir, 'configs/config_data.yaml')
    config_data = yaml.safe_load(open(config_data_path))

    # Add configurations about the specific task and data to config
    config['data_path_x'] = config_data[config['task']][config['lang']]['data_path_x']
    config['data_path_y'] = config_data[config['task']][config['lang']]['data_path_y']
    config['victim_model_ckpt'] = config_data[config['victim_model']][config['task']][config['lang']]['victim_model_ckpt']
    config['max_source_length'] = config_data[config['victim_model']][config['task']][config['lang']]['max_source_length']
    config['max_target_length'] = config_data[config['victim_model']][config['task']][config['lang']]['max_target_length']
    config['beam_size'] = config_data[config['victim_model']][config['task']][config['lang']]['beam_size']

    # Create output directory
    config['output_dir'] = os.path.join(config['output_path'], config['task'], config['victim_model'], config['attack_model'], config['lang'] + '_'+ config['out_dirname'])
    if not os.path.exists(config['output_dir']):
        # Create a new directory because it does not exist
        os.makedirs(config['output_dir'])
        print("Saving to directory", config['output_dir'])

    config['logfile'] = os.path.join(config['output_dir'], config['task']+'.log')
    config['device'] = device

    print('Start attack')
    
    atk_model = {}

    if config['attack_model'] == 'codebert':
        atk_model['name'] = 'codebert'
        atk_model['tokenizer'] = RobertaTokenizer.from_pretrained(config['mlm_path'])
        atk_config = RobertaConfig.from_pretrained(config['mlm_path'])
        atk_model['mlm'] = RobertaForMaskedLM.from_pretrained(config['mlm_path'], config=atk_config)
        print('CodeBert')
        atk_model['mlm'].to(device)
    elif config['attack_model'] == 'bertattack':
        atk_model['name'] = 'bertattack'
        atk_model['sim_thresh'] = 0.2
        atk_model['tokenizer'] = BertTokenizer.from_pretrained("bert-base-uncased")
        atk_model['mlm']  = BertForMaskedLM.from_pretrained("bert-base-uncased")
        atk_model['sim_predictor'] = get_sim_predictor()
        print("BertAttack")
        atk_model['mlm'].to(device)
    elif config['attack_model'] == 'textfooler':
        atk_model['name'] = 'textfooler'
        atk_model['sim_thresh'] = 0.5
        atk_model['tokenizer'] = ''
        atk_model['mlm']  = ''
        atk_model['idx2word'], atk_model['word2idx'], atk_model['cos_sim'] = get_var()
        atk_model['sim_predictor'] = get_sim_predictor()
        print("Textfooler")
    

    victim_model = {}
    victim_model['name'] = config['victim_model']

    if config['victim_model'] == 'codet5':
        victim_model['model'], victim_model['tokenizer'] = get_codet5_model(config)
    elif config['victim_model'] == 'plbart':
        victim_model['model'], victim_model['tokenizer'] = get_plbart_model(config) 
    elif config['victim_model'] == 'codebert':
        victim_model['model'], victim_model['tokenizer'] = get_codebert_model(config) 
    elif config['victim_model'] == 'graphcodebert':
        victim_model['model'], victim_model['tokenizer'] = get_graphcodebert_model(config) 
    elif config['victim_model'] == 'roberta':
        victim_model['model'], victim_model['tokenizer'] = get_roberta_model(config) 

    victim_model['model'].to(device)
    
    if config['task'] == 'summarize':
        features = get_summarization_data(config['data_path_x'])
    elif config['task'] == 'translation' or config['task'] == 'refinement':
        features = get_translation_data(config['data_path_x'], config['data_path_y'])

    features_output = []

    with torch.no_grad():
        for index, feature in enumerate(tqdm(features[config['start']:config['end']])):
            code, output = feature
            feat = Feature(code.strip(), output.strip())

            print('\n\r Instance {:d} '.format(index) + victim_model['name'], end='')
            feat = attack(feat, config, victim_model, atk_model, max_length=512, threshold_pred_score=0.3)
            print('\n', feat.pred_bleu, feat.after_attack_bleu, feat.changes, feat.change, feat.query, feat.success)

            if feat.success > 2:
                print('success', end='')
            else:
                print('failed', end='')

            print('\n')
            features_output.append(feat)

            if index % 50 == 0:
                dump_features(features_output, config['output_dir'], index)
                # if index > 0:
                #     os.remove(os.path.join(config['output_dir'], str(index-50)+'.json'))
                #     print("Removed previously saved dump.\n")

    dump_features(features_output, config['output_dir'], index)


if __name__ == '__main__':
    run_attack()
