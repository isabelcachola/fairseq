import torch
from fairseq.models.bart import BARTModel
import argparse
from pprint import pprint
from tqdm import tqdm
import os
from os.path import join
import shutil
import time
import re
import logging
import numpy as np
import json 
import random
import string
import shutil
import files2rouge
import time

def test_rouge(cand, ref, outpath=None, tmp_dir='/tmp/', multitarget=False, quick=False):
    print(cand)
    print(ref)
    def random_string(stringLength=8):
        """Generate a random string of fixed length """
        letters= string.ascii_lowercase
        return ''.join(random.sample(letters,stringLength))
    tmp_path = join(tmp_dir, 'tmp'+random_string())
    os.makedirs(tmp_path)
    # print(tmp_path)
    hyp_path = join(tmp_path, 'hyp.txt')
    ref_path = join(tmp_path, 'ref.txt')

    candidates = [line.strip().lower() for line in open(cand, encoding='utf-8')]
    if multitarget or not quick:
        references = [json.loads(line.strip())['target'] for line in open(ref, encoding='utf-8')]
    else:
        references = [line.lower().strip() for line in open(ref, encoding='utf-8')]
    assert len(candidates) == len(references), f'{tmp_dir}: len cand {len(candidates)} len ref {len(references)}'

    if quick and not multitarget:
        hyp = open(hyp_path, 'w')
        hyp.write('\n'.join([c.replace('\n', '') for c in candidates]))
        hyp.close()
        ref = open(ref_path, 'w')
        ref.write('\n'.join([r.lower().replace('\n', '') for r in references]))
        ref.close()
        _r = files2rouge.run(ref_path, hyp_path, to_json=True)
        return _r

    paper_ids = [json.loads(line.strip())['paper_id'] for line in open(ref, encoding='utf-8')]
    all_scores = []
    save_scores = []

    # For each prediction
    for cand_idx, cand in enumerate(candidates):
        curr_targets = references[cand_idx]
        curr_scores = []
        hyp = open(hyp_path, 'w')
        hyp.write(cand)
        hyp.close()
        # import ipdb; ipdb.set_trace()
        for tgt in curr_targets:
            tgt = tgt.lower().strip('\n')
            ref = open(ref_path, 'w')
            ref.write(tgt)
            ref.close()
            try:
                _r = files2rouge.run(ref_path, hyp_path, to_json=True)
            except Exception as e:
                print(e)
                exit(0)
            curr_scores.append(_r)
        # Take the max of curr scores
        r1 = [r['rouge-1']['f']['value'] for r in curr_scores]
        max_idx = r1.index(max(r1))

        save_scores.append({
                        'paper_id': paper_ids[cand_idx],
                        'all_scores': curr_scores,
                        'max_idx': max_idx,
                        'prediction': cand,
                        'target': curr_targets
                            })
        all_scores.append(curr_scores[max_idx])

    hyp = open(hyp_path, 'w')
    ref = open(ref_path, 'w')
    for score in save_scores:
        hyp.write(score['prediction'].lower() + '\n')
        ref.write(score['target'][score['max_idx']].lower().strip('\n') + '\n')
    hyp.close()
    ref.close()
    print(hyp_path, ref_path)

    final_rouge = files2rouge.run(hyp_path, ref_path, ignore_empty=True, to_json=True)

    if outpath:
        with open(outpath, 'w') as fout:
            for s in save_scores:
                fout.write(json.dumps(s) + '\n')

    shutil.rmtree(tmp_path)
    return final_rouge

def evaluate(bart, bsz, count, datadir, outdir, decoder_params,
            test_fname='test.hypo', multitarget=False, quick=False,
            gen_only=False, rouge_only=False):
    source_fname = os.path.join(datadir, 'test.source')
    pred_fname = os.path.join(outdir, test_fname)
    if not rouge_only:
        bart.cuda()
        bart.eval()
        bart.half()
        with open(source_fname, encoding="utf-8") as source, open(pred_fname, 'w', encoding="utf-8") as fout:
            sline = source.readline().strip()
            slines = [sline]
            for sline in tqdm(source):
                if count % bsz == 0:
                    with torch.no_grad():
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
                hypotheses_batch = bart.sample(slines, beam=decoder_params['beam'], 
                                                        lenpen=decoder_params['lenpen'], 
                                                        max_len_b=decoder_params['max_len_b'],
                                                        min_len=decoder_params['min_len'],
                                                        no_repeat_ngram_size=decoder_params['no_repeat_ngram_size'])
                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis.replace('\n', '') + '\n')
                    fout.flush()
    if not gen_only:
        ref_fname = 'test-multitarget.jsonl' if (multitarget or not quick) else 'test.target'
        ref_fname = os.path.join(datadir, ref_fname)
        r = test_rouge(pred_fname, 
                        ref_fname, 
                        outpath=os.path.join(outdir, test_fname + '.rouge'),
                        multitarget=multitarget, quick=quick)

        return r

def tune_decoder_params(bart, bsz, count, datadir, 
                        max_len_b, min_len, no_repeat_ngram_size,
                        outdir, test_fname='test.hypo',
                        multitarget=False, quick=False):
    print('Tuning decoder params...')
    
    beams = list(range(2,9))
    lenpens = list(np.arange(0.2, 1.2, 0.2))
    lenpens = [round(l, 2) for l in lenpens]
    
    n = len(beams) * len(lenpens)
    pbar = tqdm(total=n)
    
    decoder_params ={
        'max_len_b': args.max_len_b,
        'min_len': args.min_len, 
        'no_repeat_ngram_size': args.no_repeat_ngram_size
    }

    best_r1 = 0.
    best_r = None
    best_beam = None
    best_lenpen = None

    for  b in beams:
        for l in lenpens:
            decoder_params["beam"] = b
            decoder_params["lenpen"] = l 

            r = evaluate(bart, bsz, count, datadir, outdir, decoder_params,
                        test_fname=f'tune-beam{b}-lenpen{l}-{test_fname}',
                        multitarget=multitarget, quick=quick)
            with open(join(outdir, f'tune-beam{b}-lenpen{l}-{test_fname}.score'), 'w') as fscore:
                json.dump(r, fscore, indent=4)
            # print(float(r['rouge-1']['f']['value']), best_r1,float(r['rouge-1']['f']['value']) > best_r1)
            if min:
                if float(r['rouge-1']['f']['value']) < best_r1:
                    best_r1 = r['rouge-1']['f']['value']
                    best_r = r
                    best_beam = b
                    best_lenpen = l
            else:
                if float(r['rouge-1']['f']['value']) > best_r1:
                    best_r1 = r['rouge-1']['f']['value']
                    best_r = r
                    best_beam = b
                    best_lenpen = l
            pbar.update(1)
    pbar.close()
    print(f'Best beam: {best_beam} \t Best lenpen: {best_lenpen}')
    best_r['beam'] = best_beam
    best_r['lenpen'] = best_lenpen
    return best_r

def maybe_percentages(r, percentages):
    if percentages:
        for r_type in ['rouge-1', 'rouge-2', 'rouge-l']:
            for m_type in ['f', 'p', 'r']:
                for x in r[r_type][m_type]:
                    r[r_type][m_type][x] = r[r_type][m_type][x] * 100

    return r

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir', default='tldr_data_ao')
    # parser.add_argument('--datafile', help='path to datafile, overrides datadir')
    parser.add_argument('--checkpoint_file', default='')
    parser.add_argument('--checkpoint_dir', default='')
    parser.add_argument('--dict_dir', default='')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--count', default=1, type=int)
    parser.add_argument('--batch_size', '--bsz', default=32, type=int, dest='bsz')
    parser.add_argument('--test_fname', default='test.hypo')
    parser.add_argument('--beam', default=6, type=int)
    parser.add_argument('--lenpen', default=1.0, type=float)
    parser.add_argument('--max_len_b', default=30, type=int)
    parser.add_argument('--min_len', default=5, type=int)
    parser.add_argument('--no_repeat_ngram_size', default=3, type=int)
    parser.add_argument('--tune', action='store_true', default=False)
    parser.add_argument('--multitarget', action='store_true', default=False)
    parser.add_argument('--quick', action='store_true', default=False)
    parser.add_argument('--rouge_only', action='store_true', default=False, help='flag if you don\'t want to run predictions')
    parser.add_argument('--percentages', action='store_true', default=False, help='flag if you want to print as percentages')
    args = parser.parse_args()

    start = time.time()
    #### Path checks
    if not os.path.exists(args.datadir):
        print(f'{args.datadir} does not exist')
        exit(0)
    if not os.path.exists(join(args.datadir, 'test.source')):
        print(f'{join(args.datadir, "test.source")} does not exist')
        exit(0)
    # if not os.path.exists(join(args.datadir, 'test.target')):
    #     print(f'{join(args.datadir, "test.target")} does not exist')
    #     exit(0)
    if (not os.path.exists(join(args.checkpoint_dir, args.checkpoint_file))) and (not args.rouge_only):
        print(f'{join(args.checkpoint_dir, args.checkpoint_file)} does not exist')
        exit(0)

    if not args.outdir:
            args.outdir = args.checkpoint_dir
    if args.tune:
        args.outdir = os.path.join(args.outdir, 'tuning')
    os.makedirs(args.outdir, exist_ok=True)

    # if args.rouge_only:
    #     ref_fname = 'test-multitarget.jsonl' if args.multitarget else 'test.jsonl'
    #     r = test_rouge(join(args.outdir, args.test_fname), os.path.join(args.datadir, ref_fname), 
    #                 outpath=os.path.join(args.outdir, args.test_fname + '.rouge'),
    #                 multitarget=args.multitarget, quick=args.quick)
    #     # r['beam'] = args.beam
    #     # r['lenpen'] = args.lenpen
    #     pprint(maybe_percentages(r, args.percentages))

    # else:
    if args.datadir.endswith('/'):
        args.datadir = args.datadir[:-1]
    
    if not args.dict_dir:
        args.dict_dir = args.datadir + '-bin'

    if args.rouge_only:
        bart = None
    else:
        bart = BARTModel.from_pretrained(
            args.checkpoint_dir,
            checkpoint_file=args.checkpoint_file,
            data_name_or_path=args.dict_dir,
            task='translation'
        )

    decoder_params ={
        'beam': args.beam,
        'lenpen': args.lenpen,
        'max_len_b': args.max_len_b,
        'min_len': args.min_len, 
        'no_repeat_ngram_size': args.no_repeat_ngram_size
    }

    if args.tune:
        r = tune_decoder_params(bart, args.bsz, args.count, 
                args.datadir,
                args.max_len_b, args.min_len, args.no_repeat_ngram_size, 
                args.outdir, 
                test_fname=args.test_fname,
                multitarget=args.multitarget,
                quick=args.quick)
        pprint(maybe_percentages(r, args.percentages))
    else:
        r = evaluate(bart, args.bsz, args.count, 
                args.datadir, args.outdir, 
                decoder_params, 
                test_fname=args.test_fname,
                multitarget=args.multitarget,
                quick=args.quick, rouge_only=args.rouge_only)
        r['beam'] = args.beam
        r['lenpen'] = args.lenpen
        pprint(maybe_percentages(r, args.percentages))
    
    with open(join(args.outdir, args.test_fname + '.score'), 'w') as fout:
        fout.write(json.dumps(r, indent=4))

    end = time.time()
    print(f'Time to run script: {(end-start)} sec')

if __name__=='__main__':
    import sys
    main(sys.argv[1:])