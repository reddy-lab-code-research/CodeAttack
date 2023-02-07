import os, sys
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd


def constrain_type(config, tgt_word, final_words):
    in_file = os.path.join(config['output_dir'], config['victim_model'] + '.txt')
    fp = open(in_file, 'w+')
    fp.write(tgt_word.strip() + '\n')
    for words in final_words:
        # Tokenization cannot handle nested "'"
        if "'" in words:
            words = words.replace("'", "")
        fp.write(words.strip() + '\n')
    fp.close()

    out_file = os.path.join(config['output_dir'], config['victim_model'] + '.json')
    script_file = '/home/akshitajha/adversarial_attack/scripts/tokenizer/./tokenize' 
    command =  script_file + ' ' + in_file + ' -m csv -o ' + out_file
    os.system(command)

    # Saved in the format [line, column, class, token]
    data = pd.read_csv(out_file)

    meta_data = {}
    token_class = ['identifier', 'keyword', 'integer', 'floating', 'string', 'character', 'operator', 'preprocessor', 'sum_classes']
   
    # Insert target word as the first word
    # final_words indexes from 0
    # meta_data indexes from 1
    final_words.insert(0, tgt_word)
    for i in range(1, len(final_words)+1):
        meta_data[i] = {}
        for c in token_class:
            meta_data[i][c] = 0

    for idx, d in data.iterrows():
        meta_data[d['line']][d['class']] =  meta_data[d['line']][d['class']] + 1
        meta_data[d['line']]['sum_classes'] = meta_data[d['line']]['sum_classes'] + 1

    tgt_class = meta_data[1]['sum_classes']
    poss_words = []
    rejected_words = []

    # Stricter Type Constraint
    tgt_tokens = data[data['line']==1]['class'].values.tolist()

    # If DFG constraint only
    if config['use_dfg_constraint'] == 1:
        if list(set(tgt_tokens)) != ['operator']:
            return poss_words

    for key in meta_data:
        # Making sure the same classes are replaced
        if meta_data[key]['sum_classes'] == tgt_class:
            sub_tokens = data[data['line']==key]['class'].values.tolist()
            if tgt_tokens == sub_tokens:
                poss_words.append(final_words[key-1])
            else:
                rejected_words.append(final_words[key-1])

        # Addition of an operator
        elif meta_data[key]['sum_classes'] == tgt_class + 1:
            sub_tokens = data[data['line']==key]['class'].values.tolist()
            sub_len = sub_tokens.count('operator')
            tgt_len = tgt_tokens.count('operator')
            if sub_len - tgt_len == 1:
                # Make sure the token classes are the same except an operator
                # A keyword is only replaced by a keyword, etc.
                if set(sub_tokens) == set(tgt_tokens):
                    poss_words.append(final_words[key-1])
                else:
                    rejected_words.append(final_words[key-1])
            else:
                rejected_words.append(final_words[key-1])

        # Deletion of an operator
        elif meta_data[key]['sum_classes'] == tgt_class - 1:
            sub_tokens = data[data['line']==key]['class'].values.tolist()
            sub_len = sub_tokens.count('operator')
            tgt_len = tgt_tokens.count('operator')
            if tgt_len - sub_len == 1:
                # Make sure the token classes are the same except an operator
                # A keyword is only replaced by a keyword, etc.
                if set(sub_tokens) == set(tgt_tokens):
                    poss_words.append(final_words[key-1])
                else:
                    rejected_words.append(final_words[key-1])
            else:
                rejected_words.append(final_words[key-1])
        else:
            rejected_words.append(final_words[key-1])

    return poss_words[1:]


def get_bpe_substitues(config, tgt_word, substitutes, tokenizer, mlm_model):
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

    # find all possible candidates 

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size
    ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)

    # Add Type Detection as a constraint
    if config['use_ast_constraint'] == 1:
        final_words = constrain_type(config, tgt_word, final_words)

    return final_words


# def get_substitues(config, tgt_word, substitutes, tokenizer, mlm_model, substitutes_score=None, threshold=3.0):
def get_substitues(config, tgt_word, keys, atk_model, word_predictions, word_pred_scores_all, top_index, threshold):

    # Get the sub_words and their scores for a particular maksed word
    # substitues L, k
    # from this matrix to recover a word
    substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
    substitutes_score = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]

    tokenizer = atk_model['tokenizer']
    mlm_model = atk_model['mlm']

    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words
        
    elif sub_len == 1:
        for (i,j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
        
        if config['use_ast_constraint'] == 1:
            words = constrain_type(config, tgt_word, words)

    else:
        if config['use_bpe'] == 1:
            words = get_bpe_substitues(config, tgt_word, substitutes, tokenizer, mlm_model)
        else:
            return words
    return words
