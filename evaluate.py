import torch
from fairseq.models.bart import BARTModel
import argparse
import pyrouge
from pprint import pprint
from tqdm import tqdm
import os
import shutil
import time
import re
import logging

def filter_rouge(r):
    _r = {}
    for key, value in r.items():
        if re.match('rouge_[1l2]_\w+', key):
            _r[key] = value
    return _r

def test_rouge(cand, ref, temp_dir='./tmp'):
    candidates = [line.strip() for line in open(cand, encoding='utf-8')]
    references = [line.strip() for line in open(ref, encoding='utf-8')]
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(tmp_dir + "/candidate", exist_ok=True)
    os.makedirs(tmp_dir + "/reference", exist_ok=True)
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        # print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict

def evaluate(bart, bsz, count, datadir, outdir, decoder_params,
            visible_device=-1, test_fname='test.hypo'):
    device = f'cuda:{visible_device}' if visible_device != -1 and torch.cuda.is_available() else 'cpu'
    bart.to(torch.device(device))
    bart.eval()
    bart.half()
    source_fname = os.path.join(datadir, 'test.source')
    pred_fname = os.path.join(outdir, test_fname)
    with open(source_fname, encoding="utf-8") as source, open(pred_fname, 'w', encoding="utf-8") as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in tqdm(source):
            if count % bsz == 0:
                with torch.no_grad():
                    # import ipdb; ipdb.set_trace()
                    hypotheses_batch = bart.sample(slines, beam=decoder_params['beam'], 
                                                    lenpen=decoder_params['lenpen'], 
                                                    max_len_b=decoder_params['max_len_b'],
                                                    min_len=decoder_params['min_len'],
                                                    no_repeat_ngram_size=decoder_params['no_repeat_ngram_size'])
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
        if slines != []:
            # import ipdb; ipdb.set_trace()
            hypotheses_batch = bart.sample(slines, beam=decoder_params['beam'], 
                                                    lenpen=decoder_params['lenpen'], 
                                                    max_len_b=decoder_params['max_len_b'],
                                                    min_len=decoder_params['min_len'],
                                                    no_repeat_ngram_size=decoder_params['no_repeat_ngram_size'])
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
    r = test_rouge(os.path.join(datadir, 'test.target'), pred_fname)
    r = filter_rouge(r)
    pprint(r)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name_or_path', default='tldr_data_ao-bin')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt')
    parser.add_argument('--checkpoint_dir', default='checkpoints/')
    # parser.add_argument('--datadir', default='tldr_data/')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--count', default=1, type=int)
    parser.add_argument('--batch_size', '--bsz', default=32, type=int, dest='bsz')
    parser.add_argument('--visible_device', default=0, type=int)
    parser.add_argument('--test_fname', default='test.hypo')
    parser.add_argument('--beam', default=4, type=int)
    parser.add_argument('--lenpen', default=2.0, type=float)
    parser.add_argument('--max_len_b', default=140, type=int)
    parser.add_argument('--min_len', default=55, type=int)
    parser.add_argument('--no_repeat_ngram_size', default=3, type=int)
    args = parser.parse_args()

    # evaluator = Rouge()
    # if args.checkpoint_dir_or_name == 'bart.large':
    #     bart = torch.hub.load('pytorch/fairseq', 'bart.large')
    # elif args.checkpoint_dir_or_name == 'bart.large.xsum':
    #     bart = torch.hub.load('pytorch/fairseq', 'bart.large.xsum')
    # else:
    bart = BARTModel.from_pretrained(
        args.checkpoint_dir,
        checkpoint_file=args.checkpoint_file,
        data_name_or_path=args.data_name_or_path,
        task='translation'
    )

    print(bart)

    args.datadir = args.data_name_or_path.split('-')[0]

    decoder_params ={
        'beam': args.beam,
        'lenpen': args.lenpen,
        'max_len_b': args.max_len_b,
        'min_len': args.min_len, 
        'no_repeat_ngram_size': args.no_repeat_ngram_size
    }

    if not args.outdir:
        args.outdir = args.checkpoint_dir
    evaluate(bart, args.bsz, args.count, 
            args.datadir, args.outdir, 
            decoder_params, 
            visible_device=args.visible_device,
            test_fname=args.test_fname)
