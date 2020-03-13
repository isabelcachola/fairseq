import torch
from fairseq.models.bart import BARTModel
import argparse
import pyrouge
from pprint import pprint
from tqdm import tqdm
import os
import shutil
import time

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

def evaluate(bart, bsz, count, datadir, outdir, visible_device=-1):
    device = f'cuda:{visible_device}' if visible_device != -1 and torch.cuda.is_available() else 'cpu'
    bart.to(torch.device(device))
    bart.eval()
    bart.half()
    source_fname = os.path.join(datadir, 'test.source')
    pred_fname = os.path.join(outdir, 'test.hypo')
    with open(source_fname) as source, open(pred_fname, 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in tqdm(source):
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=6, lenpen=1.0, max_len_b=60, min_len=10, no_repeat_ngram_size=3)

                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=6, lenpen=1.0, max_len_b=60, min_len=10, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
    r = test_rouge(os.path.join(datadir, 'test.target'), pred_fname)
    pprint(r)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name_or_path', default='tldr_data-bin')
    parser.add_argument('--checkpoint_file', default='checkpoint_best.pt')
    parser.add_argument('--checkpoint_dir_or_name', default='checkpoints/')
    parser.add_argument('--datadir', default='tldr_data/')
    parser.add_argument('--outdir', default='')
    parser.add_argument('--count', default=1, type=int)
    parser.add_argument('--batch_size', default=32, type=int, dest='bsz')
    parser.add_argument('--visible_device', default=0, type=int)
    args = parser.parse_args()

    # evaluator = Rouge()
    # if args.checkpoint_dir_or_name == 'bart.large':
    #     bart = torch.hub.load('pytorch/fairseq', 'bart.large')
    # elif args.checkpoint_dir_or_name == 'bart.large.xsum':
    #     bart = torch.hub.load('pytorch/fairseq', 'bart.large.xsum')
    # else:
    bart = BARTModel.from_pretrained(
        args.checkpoint_dir_or_name,
        checkpoint_file=args.checkpoint_file,
        data_name_or_path=args.data_name_or_path,
        task='translation'
    )
    if not args.outdir:
        args.outdir = args.checkpoint_dir_or_name
    evaluate(bart, args.bsz, args.count, args.datadir, args.outdir, visible_device=args.visible_device)
