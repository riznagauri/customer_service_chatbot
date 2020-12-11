#!/usr/bin/env python3

import torch
import torch.nn as nn
import os
import argparse
from model import predict_model_factory
from dataset import field_factory, metadata_factory
from serialization import load_object
from constants import MODEL_START_FORMAT
global model
from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'



class ModelDecorator(nn.Module):
      def __init__(self, model):
        super(ModelDecorator, self).__init__()
        self.model = model

      def forward(self, question, sampling_strategy, max_seq_len):
        return self.model([question], sampling_strategy, max_seq_len)[0]


customer_service_models = {
    'amazon': ('trained-model/amazon', 10),
}

model_path = 'trained-model/amazon'
epoch = 10
def parse_args():
    parser = argparse.ArgumentParser(description='Script for "talking" with pre-trained chatbot.')
    parser.add_argument('-cs', '--customer-service', choices=['amazon'])
    parser.add_argument('-p', '--model-path',
                        help='Path to directory with model args, vocabulary and pre-trained pytorch models.')
    parser.add_argument('-e', '--epoch', type=int, help='Model from this epoch will be loaded.')
    parser.add_argument('--sampling-strategy', choices=['greedy', 'random', 'beam_search'], default='greedy',
                        help='Strategy for sampling output sequence.')
    parser.add_argument('--max-seq-len', type=int, default=50, help='Maximum length for output sequence.')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use cuda if available.')

    args = parser.parse_args()
 
    #if args.customer_service:
     #   cs = customer_service_models[args.customer_service]
      #  args.model_path = cs[0]
      #  args.model_path='amazon'
       # args.model_path = 'pretrained-models/amazon'
      #  print(cs[0])
       # print(cs[1])
       # args.epoch = 10

    return args


def get_model_path(dir_path, epoch):
    name_start = MODEL_START_FORMAT % epoch
    for path in os.listdir(dir_path):
        if path.startswith(name_start):
            return dir_path + path
    raise ValueError("Model from epoch %d doesn't exist in %s" % (epoch, dir_path))


def main():
    global model
    torch.set_grad_enabled(False)
    args = parse_args()
   # print('Args loaded')
 #   model_args = load_object(args.model_path + os.path.sep + 'args')
    model_args = load_object(model_path + os.path.sep + 'args')
   # print('Model args loaded.')
    vocab = load_object(model_path + os.path.sep + 'vocab')
   # print('Vocab loaded.')

    cuda = torch.cuda.is_available() and args.cuda
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
   # print("Using %s for inference" % ('GPU' if cuda else 'CPU'))

    field = field_factory(model_args)
    field.vocab = vocab
    metadata = metadata_factory(model_args, vocab)

    model = ModelDecorator(
        predict_model_factory(model_args, metadata, get_model_path(model_path + os.path.sep, epoch), field))
   # print('model loaded')
    model.eval()
    #response = model(userText, sampling_strategy=args.sampling_strategy, max_seq_len=args.max_seq_len)

    
   # print('\n\nBot: Hello, how can i help you ?', flush=True)
    #while question != 'bye':
     #   while True:
      #      print('Me: ', end='')
       #     question = input()
       #     if question:
        #        break

      
@app.route("/")
def home():
  return render_template("index.html")

@app.route("/get")
def get_bot_response():

  userText = request.args.get('msg')
  response = model(userText, sampling_strategy=args.sampling_strategy, max_seq_len=args.max_seq_len)
  return str(response)      
    


if __name__ == '__main__':
     main()
     app.run()
     
     
