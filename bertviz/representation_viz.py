import torch
import json
import os
import IPython.display as display
import string
import nltk
from nltk.tokenize import word_tokenize

class RepresentationData:
    """Represents data needed for attention map visualization"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def get_data(self, sentence):
        tokens_tensor, token_type_tensor, tokens = self._get_inputs(sentence)
        output, _, _ = self.model(tokens_tensor, token_type_ids=token_type_tensor)
        output = torch.stack(output)
        return tokens, output.squeeze().detach().numpy()

    def _get_inputs(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens_delim = ['[CLS]'] + tokens + ['[SEP]']
        
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens_delim)
        tokens_tensor = torch.tensor([token_ids])
        token_type_tensor = torch.LongTensor([[0] * len(tokens_delim)])
        return tokens_tensor, token_type_tensor, tokens_delim
    
class VerbVisualizationData:
    """Represents data needed for attention map visualization"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def get_data(self, sentence, bounds):
        tokens_tensor, token_type_tensor, tokens, pos_tags, bert_to_orig, orig_to_bert = self._get_inputs(sentence)
        output, _, _ = (0,0,0)#self.model(tokens_tensor, token_type_ids=token_type_tensor)
        #output = torch.stack(output)
        
        #Get location of start of verbs
        verb_group_idxs = [0] * len(tokens)
        verb_idxs = []
        bert_verb_idxs = []
        for i,tag in enumerate(pos_tags):
            if 'VB' in tag and all([i > b[0] and i < b[1] for b in bounds]):
                verb_group_idxs[orig_to_bert[i]] = 1
                verb_idxs.append(i)
                bert_verb_idxs.append(orig_to_bert[i])
                
        return tokens, output, verb_group_idxs, verb_idxs, bert_verb_idxs

    def _get_inputs(self, sentence):
        #1) tokenize with nltk and get pos tags
        nltk_tokens = word_tokenize(sentence)
        nltk_out = []
        #2) POS tag
        pos_tags = [t[1] for t in nltk.pos_tag(nltk_tokens)]
        pos_out = []
        #3) tokenize the individual tokens and map to original
        bert_to_orig = [-1]
        orig_to_bert = []
        tokens = []
        for i,token in enumerate(nltk_tokens):
            tokens_curr = self.tokenizer.tokenize(token)
            nltk_out.append(token)
            pos_out.append(pos_tags[i])
            
            orig_to_bert.append(len(tokens) + 1) # Add 1 for CLS
                                
            tokens.extend(tokens_curr)
            bert_to_orig.extend([i for tok in tokens_curr])
            if len(tokens) > 510:
                tokens = tokens[:510]
                bert_to_orig = bert_to_orig[:511]
                break
        bert_to_orig.append(-1) # For SEP
        
        tokens_delim = ['[CLS]'] + tokens + ['[SEP]']
        
        
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens_delim)
        tokens_tensor = torch.tensor([token_ids])
        token_type_tensor = torch.LongTensor([[0] * len(tokens_delim)])
        return tokens_tensor, token_type_tensor, tokens_delim, pos_out, bert_to_orig, orig_to_bert