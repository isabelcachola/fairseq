import torch
from fairseq.models.bart import BARTModel
import argparse
from tqdm import tqdm
import os
from os.path import join
import numpy as np
import json 
from time import time

def evaluate(bart, decoder_params, inp_text):
    if torch.cuda.is_available():
        bart.cuda()
        bart.half()
    bart.eval()
    sline = inp_text.strip()
    slines = [sline]
    with torch.no_grad():
        hypotheses_batch = bart.sample(slines, beam=decoder_params['beam'], 
                                        lenpen=decoder_params['lenpen'], 
                                        max_len_b=decoder_params['max_len_b'],
                                        min_len=decoder_params['min_len'],
                                        no_repeat_ngram_size=decoder_params['no_repeat_ngram_size'])
    return hypotheses_batch[0]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', default='')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt')
    parser.add_argument('--datadir', default='tldr_data/')
    parser.add_argument('--count', default=1, type=int)
    parser.add_argument('--batch_size', '--bsz', default=32, type=int, dest='bsz')
    parser.add_argument('--beam', default=6, type=int)
    parser.add_argument('--lenpen', default=1.0, type=float)
    parser.add_argument('--max_len_b', default=30, type=int)
    parser.add_argument('--min_len', default=5, type=int)
    parser.add_argument('--no_repeat_ngram_size', default=3, type=int)
    args = parser.parse_args()

    if (not os.path.exists(join(args.checkpoint_dir, args.checkpoint_file))):
        print(f'{join(args.checkpoint_dir, args.checkpoint_file)} does not exist')
        exit(0)

    # Need to put dict.source and dict.target in checkpoint dir later

    bart = BARTModel.from_pretrained(
        args.checkpoint_dir,
        checkpoint_file=args.checkpoint_file,
        data_name_or_path=args.datadir + '-bin',
        task='translation'
    )

    decoder_params ={
        'beam': args.beam,
        'lenpen': args.lenpen,
        'max_len_b': args.max_len_b,
        'min_len': args.min_len, 
        'no_repeat_ngram_size': args.no_repeat_ngram_size
    }

    while True:
        try:
            inp_text = input('Enter abstract: ')
            hyp = evaluate(bart, decoder_params, inp_text)
            start = time()
            print(f'Prediction: {hyp}')
            end = time()
            print(f'Time to generate: {end-start} seconds')
        except KeyboardInterrupt:
            print('\nExiting...')
            exit(0)
