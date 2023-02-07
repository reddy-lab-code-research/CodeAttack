import os, sys
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from graphcodebert_input import convert_examples_to_features

def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    # list of words
    return masked_words


def get_important_scores(words, pred_score, pred_indices, victim_model, config, max_len):
    masked_words = _get_masked(words)
    # List of text of masked words for each function snippet.
    # Mask one word at a time for each function. 
    # There are as many [MASK]ed sentences as the no. of words.
    masked_codes = [' '.join(words) for words in masked_words]
    all_masked_code_ids = []
    all_attention_masks = []
    all_segs = []
    for code in masked_codes:
        inputs = victim_model['tokenizer'].encode_plus(code, None, add_special_tokens=True, max_length=config['max_source_length'])
        input_ids = inputs["input_ids"]
        attention_mask = [1] * len(input_ids)
            
        padding_length = max_len - len(input_ids)
        input_ids = input_ids + (padding_length * [0])
        attention_mask = attention_mask + (padding_length * [0])
            
        all_masked_code_ids.append(torch.tensor(input_ids))
        all_attention_masks.append(torch.tensor(attention_mask))
                    
    all_masked_codes =  torch.stack(all_masked_code_ids).to(config['device'])
    all_attention_masks = torch.stack(all_attention_masks).to(config['device'])
    
    eval_data = TensorDataset(all_masked_codes)
    eval_attention_masks = TensorDataset(all_attention_masks)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=config['batch_size'])

    eval_attention_sampler = SequentialSampler(eval_attention_masks)
    eval_attention_dataloader = DataLoader(eval_attention_masks, sampler=eval_attention_sampler, batch_size=config['batch_size'])

    leave_1_probs = []
    for masked_input, masked_attention in zip(eval_dataloader, eval_attention_dataloader):
        masked_input_batch = masked_input[0]
        masked_attention_batch = masked_attention[0]

        if config['victim_model'] == 'codet5':
            leave_1_prob_batch = victim_model['model'].generate(masked_input_batch,
                                                                output_scores=True, 
                                                                return_dict_in_generate=True,
                                                                early_stopping=config['task']=='summarization',
                                                                max_length=config['max_target_length']) 
            scores = leave_1_prob_batch.scores
            
            # Reshape the scores: B x len(pred_seq) x vocab_size
            batch_scores_stack = torch.stack(list(scores))
            pred_len, batch_size, vocab = batch_scores_stack.shape
            batch_scores_stack = batch_scores_stack.reshape(batch_size, pred_len, vocab)

        elif config['victim_model'] == 'codebert' or config['victim_model'] == 'roberta':
            preds, scores = victim_model['model'](source_ids=masked_input_batch,
                                                  source_mask=masked_attention_batch)  
            
            # Scores: B x len(pred_seq) x vocab_size
            batch_scores_stack = torch.stack(list(scores))

        # Do not use the index of the original predicted sequence
        # Calculate the max scores based on the highest values of the current predicted sequence 
        if config['use_pred_idx'] == 0:
            max_scores_wo_pred_idx, max_scores_pred_idx = torch.max(batch_scores_stack, dim=2)
            max_scores = torch.sum(max_scores_wo_pred_idx, dim=1)
        
        # Use the index of the original predicted sequence
        # Calculate the max scores based on the current values at the index of the original predicted sequence
        # Original Pred Sequence Text: "<s> Two sum </s>"; Orginal pred Indices: [1, 100, 200, 2]
        # Calculate the scores only at these indices
        elif config['use_pred_idx'] == 1:
            max_scores_with_idx = torch.index_select(batch_scores_stack, 2, torch.tensor(pred_indices).to(config['device']))
            max_scores_idx_batch = torch.diagonal(max_scores_with_idx, dim1=-2, dim2=-1)
            max_scores = torch.sum(max_scores_idx_batch, dim=1)

        leave_1_probs.extend(max_scores)
    
    leave_1_probs = torch.stack(leave_1_probs)  # num_words, 1

    import_scores = pred_score - leave_1_probs
    # Index of the most important words and their scores in the I/P code
    index_important_words = sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True)
    return index_important_words


def get_important_scores_graphcodebert(words, pred_score, pred_indices, victim_model, config, max_len):
    masked_words = _get_masked(words)
    # List of text of masked words for each function snippet.
    # Mask one word at a time for each function. 
    # There are as many [MASK]ed sentences as the no. of words.
    masked_codes = [' '.join(words) for words in masked_words]
    all_masked_code_ids = []
    all_source_mask_ids = []
    all_attention_masks = []
    all_position_idx = []
    all_segs = []
    for code in masked_codes:
        examples = [code]
        input_ids, code_mask, position_idx, attention_mask = convert_examples_to_features(examples, victim_model['tokenizer'], config)
        # input_ids = input_ids.unsqueeze(0)
        # code_mask = code_mask.unsqueeze(0)
        # position_idx = position_idx.unsqueeze(0)
        
        all_masked_code_ids.append(input_ids)
        all_source_mask_ids.append(code_mask)
        all_position_idx.append(position_idx)
        all_attention_masks.append(attention_mask)
                    
    all_masked_codes =  torch.stack(all_masked_code_ids).to(config['device'])
    all_source_masks = torch.stack(all_source_mask_ids).to(config['device'])
    all_position_idx = torch.stack(all_position_idx).to(config['device'])
    all_attention_masks = torch.stack(all_attention_masks).to(config['device'])
    
    eval_data = TensorDataset(all_masked_codes)
    eval_source_masks = TensorDataset(all_source_masks)
    eval_position_idx = TensorDataset(all_position_idx)
    eval_attention_masks = TensorDataset(all_attention_masks)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=config['batch_size'])

    eval_sm_sampler = SequentialSampler(eval_source_masks)
    eval_sm_dataloader = DataLoader(eval_source_masks, sampler=eval_sm_sampler, batch_size=config['batch_size'])

    eval_position_sampler = SequentialSampler(eval_position_idx)
    eval_position_dataloader = DataLoader(eval_position_idx, sampler=eval_position_sampler, batch_size=config['batch_size'])

    eval_attention_sampler = SequentialSampler(eval_attention_masks)
    eval_attention_dataloader = DataLoader(eval_attention_masks, sampler=eval_attention_sampler, batch_size=config['batch_size'])

    leave_1_probs = []
    for masked_input, masked_sm, masked_pos, masked_attention in zip(eval_dataloader, eval_sm_dataloader, eval_position_dataloader, eval_attention_dataloader):
        masked_input_batch = masked_input[0]
        masked_sm_batch = masked_sm[0]
        masked_pos_batch = masked_pos[0]
        masked_attention_batch = masked_attention[0]

        preds, scores = victim_model['model'](masked_input_batch, 
                                              masked_sm_batch,
                                              masked_pos_batch, 
                                              masked_attention_batch)  
        
        # Scores: B x len(pred_seq) x vocab_size
        batch_scores_stack = torch.stack(list(scores))

        # Do not use the index of the original predicted sequence
        # Calculate the max scores based on the highest values of the current predicted sequence 
        if config['use_pred_idx'] == 0:
            max_scores_wo_pred_idx, max_scores_pred_idx = torch.max(batch_scores_stack, dim=2)
            max_scores = torch.sum(max_scores_wo_pred_idx, dim=1)
        
        # Use the index of the original predicted sequence
        # Calculate the max scores based on the current values at the index of the original predicted sequence
        # Original Pred Sequence Text: "<s> Two sum </s>"; Orginal pred Indices: [1, 100, 200, 2]
        # Calculate the scores only at these indices
        elif config['use_pred_idx'] == 1:
            max_scores_with_idx = torch.index_select(batch_scores_stack, 2, torch.tensor(pred_indices).to(config['device']))
            max_scores_idx_batch = torch.diagonal(max_scores_with_idx, dim1=-2, dim2=-1)
            max_scores = torch.sum(max_scores_idx_batch, dim=1)

        leave_1_probs.extend(max_scores)
    
    leave_1_probs = torch.stack(leave_1_probs)  # num_words, 1

    import_scores = pred_score - leave_1_probs
    # Index of the most important words and their scores in the I/P code
    index_important_words = sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True)
    return index_important_words
