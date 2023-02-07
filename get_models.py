import torch
import torch.nn as nn

from transformers import BartConfig, BartModel, BartTokenizer, BartForConditionalGeneration, RobertaModel, RobertaConfig
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM
import model_graphcodebert
import model_codebert


def get_plbart_model(config):
    model_ckpt = torch.load(config['victim_model_ckpt']) # fairseq original plbart ckpt

    # Build config
    vocab_size, d_model = model_ckpt['model']['encoder.embed_tokens.weight'].shape
    args = vars(model_ckpt['args'])
    bart_config = BartConfig(vocab_size=vocab_size,
                             d_model=d_model,
                             encoder_layers=args['encoder_layers'],
                             decoder_layers=args['decoder_layers'],
                             encoder_attention_heads=args['encoder_attention_heads'],
                             decoder_attention_heads=args['decoder_attention_heads'],
                             encoder_ffn_dim=args['encoder_ffn_embed_dim'],
                             decoder_ffn_dim=args['decoder_ffn_embed_dim'],
                             activation_function=args['activation_fn'],
                             dropout=args['dropout'],
                             attention_dropout =args['attention_dropout'],
                             activation_dropout=args['activation_dropout'],
                             scale_embedding=not(args['no_scale_embedding']),
                             forced_eos_token_id=2 #<------------------ check
                            )
    model = BartModel(bart_config) # in huggingface
    # model = BartForConditionalGeneration(bart_config)
    weights = model.state_dict()

    # Load weights.
    for k in weights:
        if k in model_ckpt['model']:
            weights[k] = model_ckpt['model'][k]
    # weights['decoder.embed_tokens.weight'] = weights['model.encoder.embed_tokens.weight']
    # weights['shared.weight'] = weights['model.encoder.embed_tokens.weight']

    model.load_state_dict(weights)

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    return model, tokenizer


def get_codet5_model(config):
    model_ckpt = torch.load(config['victim_model_ckpt']) # original codet5 ckpt
    if config['task'] == 'summarize':
        model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base-multi-sum')
    else:
        model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
    weights = model.state_dict()

    # Load weights.
    for k in weights:
        if k in model_ckpt:
            weights[k] = model_ckpt[k]

    model.load_state_dict(weights)

    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    return model, tokenizer


def get_codebert_model(config):
    model_config = RobertaConfig.from_pretrained('microsoft/codebert-base')
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    encoder = RobertaModel.from_pretrained('microsoft/codebert-base', config=model_config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=model_config.hidden_size, nhead=model_config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = model_codebert.Seq2Seq(encoder=encoder, decoder=decoder, config=model_config,
                  beam_size=config['beam_size'],max_length=config['max_target_length'],
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    model.load_state_dict(torch.load(config['victim_model_ckpt']))

    return model, tokenizer


def get_graphcodebert_model(config):
    model_config = RobertaConfig.from_pretrained('microsoft/graphcodebert-base')
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')

    encoder = RobertaModel.from_pretrained('microsoft/graphcodebert-base', config=model_config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=model_config.hidden_size, nhead=model_config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = model_graphcodebert.Seq2Seq(encoder=encoder,decoder=decoder,config=model_config,
                  beam_size=config['beam_size'],max_length=config['max_target_length'],
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    model.load_state_dict(torch.load(config['victim_model_ckpt']))

    return model, tokenizer

def get_roberta_model(config):
    model_config = RobertaConfig.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    encoder = RobertaModel.from_pretrained('roberta-base', config=model_config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=model_config.hidden_size, nhead=model_config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = model_codebert.Seq2Seq(encoder=encoder, decoder=decoder, config=model_config,
                  beam_size=config['beam_size'],max_length=config['max_target_length'],
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    model.load_state_dict(torch.load(config['victim_model_ckpt']))

    return model, tokenizer
