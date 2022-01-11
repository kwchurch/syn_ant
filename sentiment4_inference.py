#!/usr/bin/env python

import argparse

import sys,json
import torch

from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def f2s(f):
    return str(f)

def floats2str(fs):
    return '\t'.join(map(str, fs))

def collate_fn(tokenizer, examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

def get_config(fn):
    with open(fn + '/config.json', 'r') as fd:
        return json.loads(fd.read())
    

def main():
    parser = argparse.ArgumentParser(description="Simple example of inference script.")
    parser.add_argument("--model", type=str, help="base model | checkpoint", required=True)
    parser.add_argument("--delimiter", type=str, help="defaults to tab", default='\t')
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(get_config(args.model)["_name_or_path"])

    for line in sys.stdin:
        fields = line.rstrip().split(args.delimiter)
        if len(fields) >= 2:
            w1,w2=fields[0:2]
            parsed = collate_fn(tokenizer, [tokenizer(w1, w2, truncation=True, max_length=None)])
            outputs = model(**parsed)
            print(line.rstrip() + args.delimiter +  floats2str(outputs['logits'].detach().numpy()[0]))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
