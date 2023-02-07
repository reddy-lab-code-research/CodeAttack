from operator import index
from optparse import IndentedHelpFormatter
import os, sys
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import random

import smooth_bleu
import bleu
from important_words import get_important_scores, get_important_scores_graphcodebert
from substitues import get_substitues

from textfooler.get_substitutes_textfooler import get_substitutes_textfooler
from textfooler.get_substitutes_textfooler import get_similarity_score

from graphcodebert_input import convert_examples_to_features

filter_words = ['[UNK]']

def compute_bleu(config, pred_out, gold_text):
    output_fn = os.path.join(config['output_dir'], 'output_fn')
    gold_fn = os.path.join(config['output_dir'], 'gold_fn')
    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1:
        if config['task'] == 'summarize':
            predictions = []
            gold_list = []
            predictions.append(str(1) + '\t' + pred_out)
            # gold_list.append(str(1) + '\t' + gold_text.strip() + '\n')
            # f.write(str(1) + '\t' + pred_out.strip() + '\n')
            gold_text = gold_text.replace('\n', ' ')
            f1.write(str(1) + '\t' + gold_text.strip() + '\n')
        elif config['task'] == 'translation' or config['task'] == 'refinement':
            f.write(pred_out.strip() + '\n')
            f1.write(gold_text.strip() + '\n')

    # Summarisation: Find the smooth bleu score for the predicted text
    if config['task'] == 'summarize':
        (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
        pred_bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
    # Translation: Find the bleu-4 score for the predicted text
    elif config['task'] == 'translation' or config['task'] == 'refinement':
        pred_bleu = round(bleu._bleu(gold_fn, output_fn), 2)

    return pred_bleu

def _tokenize(seq, tokenizer, config):
    if config['do_lower'] == 1:
        seq = seq.replace('\n', '').lower()
    else:
        seq = seq.replace('\n', '')
    seq = " ".join(seq.split())
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        # Breaking a word into subwords
        sub = tokenizer.tokenize(word)
        sub_words += sub
        # `keys`: Stores the starting and the ending index of each word. This is important
        # because each word is broken into sub-words and we need to keep a track of 
        # each word's starting and ending index in the `sub_words` list.
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys, seq


def attack(feature, config, victim_model, atk_model, max_length=512, threshold_pred_score=0.3):
    
    # Preparing the code for input to the victim model
    words, sub_words, keys, input_code = _tokenize(feature.code, victim_model['tokenizer'], config)

    if config['victim_model'] != 'graphcodebert':
        code_tokens = victim_model['tokenizer'].encode_plus(input_code, None, add_special_tokens=True, max_length=config['max_source_length'])
        code_ids  = torch.tensor(code_tokens["input_ids"]).unsqueeze(0)
        code_mask = code_ids.ne(victim_model['tokenizer'].pad_token_id).to(config['device'])
        attention_mask = code_ids.ne(1)

    # Prediction using the victim model
    if config['victim_model'] == 'codet5':
        pred = victim_model['model'].generate(code_ids.to(config['device']), 
                                            attention_mask=code_mask,
                                            output_scores=True,
                                            return_dict_in_generate=True,
                                            early_stopping=config['task']=='summarize',
                                            max_length=config['max_target_length'])    
        pred_out = victim_model['tokenizer'].batch_decode(pred.sequences[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        scores = pred.scores

    if config['victim_model'] == 'codebert' or config['victim_model'] == 'roberta':
        preds, scores = victim_model['model'](source_ids=code_ids.to(config['device']),
                                      source_mask=code_mask)
        for pred in preds:
            t=pred[0].cpu().numpy()
            t=list(t)
            # Remove 0 padding
            if 0 in t:
                t=t[:t.index(0)]
            pred_out = victim_model['tokenizer'].decode(t, clean_up_tokenization_spaces=False)

    if config['victim_model'] == 'graphcodebert':
        examples = [input_code]
        code_ids, code_mask, position_idx, attention_mask = convert_examples_to_features(examples, 
                                                                                         victim_model['tokenizer'], config)
        code_ids = code_ids.unsqueeze(0)
        code_mask = code_mask.unsqueeze(0)
        position_idx = position_idx.unsqueeze(0)
        preds, scores = victim_model['model'](code_ids.to(config['device']), 
                                              code_mask.to(config['device']),
                                              position_idx.to(config['device']), 
                                              attention_mask.to(config['device']))  
        for pred in preds:
            t=pred[0].cpu().numpy()
            t=list(t)
            if 0 in t:
                t=t[:t.index(0)]
            pred_out = victim_model['tokenizer'].decode(t,clean_up_tokenization_spaces=False)

    special_toks = ['<pad>', '<s>', '</s>']
    for s in special_toks:
        if s in pred_out:
            pred_out.remove(s)
    pred_out = ''.join(pred_out).replace('\t', ' ')
    feature.gold_out = feature.gold_out.replace('\t', ' ')
    if config['task'] == 'summarize':# and (config['lang'] == 'php'or config['lang']=='ruby'):
        # feature.gold_out = feature.gold_out[:30]
        feature.gold_out = feature.gold_out.split('\n')[0]

    # Find predicted sequence score using logits
    scores = torch.stack(list(scores)).squeeze()
    pred_scores, pred_indices = torch.max(scores, dim=1)
    pred_score = torch.sum(torch.tensor(pred_scores))
    
    # Compute Bleu Score
    feature.pred_bleu = compute_bleu(config, pred_out, feature.gold_out)

    # Only attack if the bleu score is above a certain threshold
    if feature.pred_bleu <= config['bleu_theta']:
        feature.success = 3
        feature.adverse_code = input_code
        feature.after_attack_bleu = feature.pred_bleu
        feature.pred_out = pred_out
        feature.adv_out = pred_out
        return feature

    if config['use_imp_words'] == 0:
        index_important_words = []
        rand_idx =  random.sample(range(len(words)), int(len(words) * 0.4))
        for i in rand_idx:
           index_important_words.append((i, pred_score.to(config['device'])))
        
    else:
        # Get important words
        if config['victim_model'] == 'graphcodebert':
            index_important_words = get_important_scores_graphcodebert(words, pred_score, pred_indices, victim_model, config, max_length)
        else:
            index_important_words = get_important_scores(words, pred_score, pred_indices, victim_model, config, max_length)
    
    feature.query = 0
    final_input_words = copy.deepcopy(words)

    # Some values for debugging
    feature.imp_words = {}

    ####### Use CodeBERT/BERT for prediction ##########
    if atk_model['name'] == 'codebert' or atk_model['name'] == 'bertattack':
        # Sub words with [CLS] tokens
        sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
        input_ids_ = torch.tensor([atk_model['tokenizer'].convert_tokens_to_ids(sub_words)])

        # Get the best possible prediction for each position of the I/P code
        word_predictions = atk_model['mlm'](input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) x vocab
        # Top-k Predictions for a masked input index
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, config['k'], -1)  # seq-len x k
        # Ignore the 1st word because it's most probably the same word
        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
            

    # Start attacking the input text one word at a time
    for top_index in index_important_words:
        feature.imp_words[words[top_index[0]]] = float(top_index[1].detach())
        if feature.change > int(config['theta'] * (len(words))):
            feature.success = 1  # exceed
            return feature

        tgt_word = words[top_index[0]]

        if tgt_word in filter_words:
            continue

        # Ignore words after 510 tokens
        if keys[top_index[0]][0] > max_length - 2:
            continue
    
        if atk_model['name'] == 'codebert' or atk_model['name'] == 'bertattack':
            # Get possible substitutes for the target_word (tgt_word)
            substitutes = get_substitues(config, tgt_word, keys, atk_model,
                                        word_predictions, word_pred_scores_all, 
                                        top_index, threshold_pred_score)
        elif atk_model['name'] == 'textfooler':
            word_perturb = tgt_word
            idx = top_index[0]
            substitutes = get_substitutes_textfooler(word_perturb, atk_model)

        most_gap = 0.0
        candidate = None

        for substitute_ in substitutes:
            substitute = substitute_

            # For Baseline BERT_ATTACk
            # If allow_unk == 1; allow unk in substiture; do not skip
            # else skip
            # if '[UNK]' in substitute and atk_model['name'] == 'bertattack' and config['allow_unk'] == 0:
            #     feature.query = feature.query + 1
            #     continue

            if substitute == tgt_word:
                continue  # filter out original word
            if '##' in substitute:
                continue  # filter out sub-word
            if substitute in filter_words:
                continue
            
            adv_code = final_input_words
            adv_code[top_index[0]] = substitute
            adv_code = ' '.join(adv_code)

            if atk_model['name'] == 'textfooler' or atk_model['name'] == 'bertattack':
                feature.query = feature.query + 1
                semantic_sims = atk_model['sim_predictor'].semantic_sim([input_code], [adv_code])[0]
                if semantic_sims < atk_model['sim_thresh']:
                    continue
            
            if config['victim_model'] != 'graphcodebert':
                adv_code_toks = victim_model['tokenizer'].encode_plus(adv_code, None, add_special_tokens=True, max_length=config['max_source_length'])
                adv_code_ids = torch.tensor(adv_code_toks["input_ids"]).unsqueeze(0)
                adv_code_mask = adv_code_ids.ne(victim_model['tokenizer'].pad_token_id).to(config['device'])

            if config['victim_model'] == 'codet5':
                adv_vals = victim_model['model'].generate(adv_code_ids.to(config['device']), 
                                                        attention_mask=adv_code_mask,
                                                        output_scores=True,
                                                        return_dict_in_generate=True,
                                                        max_length=config['max_target_length'])
                adv_score = adv_vals.scores
                adv_text = adv_vals.sequences[0]
            
            if config['victim_model'] == 'codebert' or config['victim_model'] == 'roberta':
                temp_preds, adv_score = victim_model['model'](source_ids=adv_code_ids.to(config['device']),
                                                      source_mask=adv_code_mask)
                for pred in temp_preds:
                    t=pred[0].cpu().numpy()
                    t=list(t)
                    # Remove 0 padding
                    if 0 in t:
                        t=t[:t.index(0)]
                adv_text = t

            if config['victim_model'] == 'graphcodebert':
                adv_examples = [adv_code]
                adv_code_ids, adv_code_mask, adv_position_idx, adv_attention_mask = convert_examples_to_features(adv_examples, victim_model['tokenizer'], config)
                adv_code_ids = adv_code_ids.unsqueeze(0)
                adv_code_mask = adv_code_mask.unsqueeze(0)
                adv_position_idx = adv_position_idx.unsqueeze(0)
                temp_preds, adv_score = victim_model['model'](adv_code_ids.to(config['device']), 
                                                    adv_code_mask.to(config['device']),
                                                    adv_position_idx.to(config['device']), 
                                                    adv_attention_mask.to(config['device']))  
                for pred in temp_preds:
                    t=pred[0].cpu().numpy()
                    t=list(t)
                    if 0 in t:
                        t=t[:t.index(0)]
                adv_text = t

            feature.query += 1
            adv_score = torch.stack(list(adv_score)).squeeze()
            adv_scores, adv_indices = torch.max(adv_score, dim=1)
            adv_score = torch.sum(torch.tensor(adv_scores))

            # Calculate Bleu score of the adversarial sequence
            adv_out = victim_model['tokenizer'].batch_decode(adv_text, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for s in special_toks:
                if s in adv_out:
                    adv_out.remove(s)
            adv_out = ''.join(adv_out).replace('\t', ' ')

            adv_bleu = compute_bleu(config, adv_out, pred_out)

            if adv_bleu < feature.pred_bleu:
                feature.change += 1
                final_input_words[top_index[0]] = substitute
                feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                feature.adverse_code = adv_code
                feature.success = 4
                feature.after_attack_bleu = adv_bleu
                feature.pred_out = pred_out
                feature.adv_out = adv_out

                # # Lines 310-316 are added to geenrate plots: Remove after done
                # if feature.change < int(config['theta'] * (len(words))):
                #     gap = pred_score - adv_score
                #     if gap > most_gap:
                #         most_gap = gap
                #         candidate = substitute
                # else:
                return feature
            else:
                gap = pred_score - adv_score
                if gap > most_gap:
                    most_gap = gap
                    candidate = substitute

        if most_gap > 0:
            feature.change += 1
            feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
            pred_score = pred_score - most_gap
            final_input_words[top_index[0]] = candidate

    feature.adverse_code = ' '.join(final_input_words)
    feature.success = 2
    feature.adv_out = pred_out
    feature.pred_out = pred_out
    feature.after_attack_bleu = feature.pred_bleu
    return feature
