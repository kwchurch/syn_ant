#!/usr/bin/env python

# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
# modified by author of paper under review

import argparse

import sys,os,shutil
import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedType

from datasets import load_dataset, load_metric
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)


########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu)

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    correct_bias = config["correct_bias"]
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    # datasets = load_dataset("glue", "mrpc")

    dir='datasets/datasets_syn_ant/dataset_csv/'
    f='tag-' + args.pos + '-pairs'
    df = {}
    for split in ['train', 'val', 'test']:
        df[split] = dir + f + '.' + split + '.csv'
    datasets = load_dataset('csv', data_files=df)
    metric = load_metric("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        # outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)

        w1 = examples['word1']
        w2 = examples['word2']
        if w1 is None: w1 = 'null'
        if w2 is None: w2 = 'null'

        outputs = tokenizer(w1, w2, truncation=True, max_length=None)
        outputs['labels'] = examples['gold']
        return outputs

    tokenized_datasets = { 'train' : [tokenize_function(e) for e in datasets['train']],
                           'validation' : [tokenize_function(e) for e in datasets['val']],
                           'test' : [tokenize_function(e) for e in datasets['test']]}

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    def collate_fn(examples):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        if accelerator.distributed_type == DistributedType.TPU:
            # return tokenizer.pad(examples, padding="max_length", max_length=128, return_tensors="pt")
            return tokenizer.pad(examples, padding="max_length", max_length=12, return_tensors="pt")
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=EVAL_BATCH_SIZE
    )

    set_seed(seed)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = AutoModelForSequenceClassification.from_pretrained(args.pretrained, return_dict=True)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr, correct_bias=correct_bias)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Instantiate learning rate scheduler after preparing the training dataloader as the prepare method
    # may change its length.
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * num_epochs,
    )


    best=-1
    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            pdb.set_trace()
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if step % args.eval_steps == 0:
                accelerator.print("step %d of %d"% (step, len(train_dataloader)))
                sys.stdout.flush()
            if step > 0 and step % args.checkpoint_steps == 0:
                model.save_pretrained('%s/%s.step.%d' % (args.checkpoint_dir, args.pos, step))

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)
        fn = '%s/%s.epoch.%d' % (args.checkpoint_dir, args.pos, epoch)
        if eval_metric['accuracy'] > best:
            best = eval_metric['accuracy']
            fn = '%s/%s.best' % (args.checkpoint_dir, args.pos)
        model.save_pretrained(fn)        
        prev = '%s/%s.epoch.%d' % (args.checkpoint_dir, args.pos, epoch-1)
        if os.path.exists(prev):
            shutil.rmtree(prev)


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument("--checkpoint_dir", type=str, help="directory name", default="checkpoints4")
    parser.add_argument("--pretrained", type=str, help="base model or fine-tuned model", default="bert-base-cased")
    parser.add_argument("--eval_steps", type=int, help="evaluate every n steps", default=100)
    parser.add_argument("--checkpoint_steps", type=int, help="save every n steps", default=5000)
    parser.add_argument("--epochs", type=int, help="number of epochs", default=10)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=16)
    parser.add_argument("--seed", type=int, help="seed", default=42)
    parser.add_argument("--pos", type=str, help="adjective|fallows", default='adjective')
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": args.epochs, "correct_bias": True, "seed": args.seed, "batch_size": args.batch_size}
    training_function(config, args)


if __name__ == "__main__":
    main()
