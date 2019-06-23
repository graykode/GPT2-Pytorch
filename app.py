'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder


from flask import Flask
from flask import request
import traceback
import json
import re

app = Flask(__name__)


def text_generator(params):
    if params['batch_size'] == -1:
        params['batch_size'] = 1
    assert params['nsamples'] % params['batch_size'] == 0

    if params['length'] == -1:
        params['length'] = config.n_ctx // 2
    elif params['length'] > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    context_tokens = my_encoder.encode(params['text'])

    generated = 0
    output = []
    output_text = []
    for _ in range(params['nsamples'] // params['batch_size']):
        out = sample_sequence(
            model=model,
            length=params['length'],
            context=context_tokens if not params['unconditional'] else None,
            start_token=my_encoder.encoder['<|endoftext|>'] if params['unconditional'] else None,
            batch_size=params['batch_size'],
            temperature=params['temperature'], 
            top_k=params['top_k'], 
            device=device
        )
    
        out = out[:, len(context_tokens):].tolist()
        output.append(out)

        for i in range(params['batch_size']):
            generated += 1
            text = my_encoder.decode(out[i])
            output_text.append(text.split('<|endoftext|>')[0])
    
        return output_text, output


def validate_input(passed_json, default):
    out = dict()
    for key, value in default.items():
        try:
            out[key] = set_variable(passed_json[key], value['type'])
        except:
            out[key] = set_variable(value['default'], value['type'])
    return out


def set_variable(variable, variable_type):
    if variable_type == 'boolean':
        return to_boolean(variable)
    elif variable_type == 'integer':
        return int(variable)
    elif variable_type == 'float':
        return float(variable)
    else:
        return str(variable)


def to_boolean(variable):
    if type(variable)==bool:
        return variable
    else:
        return variable.lower() in ['true', '1', 't', 'y', 'yes']


@app.route("/generate", methods=["POST"])
def generate():

    default = {
        'nsamples': {'default': 1, 'type': 'integer', 'description': 'number of sample sampled in batch when multinomial function use'},
        'unconditional': {'default': True, 'type': 'boolean', 'description': 'If true, unconditional generation'},
        'batch_size': {'default': -1, 'type': 'integer', 'description': 'number of batch size'},
        'temperature': {'default': 0.7, 'type': 'float', 'description': 'the thermodynamic temperature in distribution'},
        'top_k': {'default': 40, 'type': 'integer', 'description': 'Returns the top k largest elements of the given input tensor along a given dimension'},
        'length': {'default': -1, 'type': 'integer', 'description': 'sentence length (< number of context)'},
    }
   
    try:
        results = []
        
        params = validate_input(request.json, default) 
        params['text'] = request.json["text"]
        text, tokens = text_generator(params)

        results.append({'text': text, 'tokens': tokens})

        return json.dumps(results), 200
        
    except:
        traceback.print_exc()
        return {'message': 'input error'}, 400


    


def get_model_bin():
    cmd = 'curl --output gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin'
    os.system(cmd)


if __name__ == '__main__':
    global state_dict, device
    global my_encoder, config, model

    if not os.path.exists('gpt2-pytorch_model.bin'):
        get_model_bin()

    state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    my_encoder = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    port = 5000
    host = '0.0.0.0'

    app.run(host=host, port=port, threaded=True)




