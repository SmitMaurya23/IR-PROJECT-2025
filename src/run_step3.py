import argparse
import yaml
import random
import numpy as np
import pandas as pd
import os

from dotted_dict import DottedDict
from datasets import load_dataset
from accelerate import Accelerator

import torch

from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments,
    set_seed
)

from peft import (
    PeftModel, get_peft_model,
    LoraConfig, TaskType,
    prepare_model_for_kbit_training
)

from data.data_preprocessing import LegalDataPreprocessor
from data.data_collator import LegalDataCollatorWithPadding

from evaluation import task3_metrics


import torch._dynamo
torch._dynamo.config.suppress_errors = True

torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-C", "--config", help="config YAML filename")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        args = yaml.safe_load(f)

    args = DottedDict(args)

    return {**args}


def determine_compute_dtype(training_args):
    """ Determines the compute dtype based on the arguments.

        if using QLoRA, the compute dtype will be the dtype that QLoRA should
        dequantize the weights to.

    Args:
        training_args: TrainingArguments.
    Returns:
        compute_dtype (torch.dtype): The compute dtype for the model.
    """

    if training_args.fp16 or training_args.fp16_full_eval: compute_dtype = torch.float16
    elif training_args.bf16 or training_args.bf16_full_eval: compute_dtype = torch.bfloat16
    else: compute_dtype = torch.float32

    return compute_dtype


def get_model_and_tokenizer(model_args, training_args):
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.pretrained_model_name_or_path,
        trust_remote_code=model_args.get("trust_remote_code", False),
    )

    # Model
    force_resize_token_embeddings = model_args.pop("force_resize_token_embeddings", True)
    pad_token = model_args.pop("pad_token", None)
    padding_side = model_args.pop("padding_side", None)

    if "torch_dtype" in model_args:
        if model_args.torch_dtype == "float16": model_args.torch_dtype = torch.float16
        elif model_args.torch_dtype == "bfloat16": model_args.torch_dtype = torch.bfloat16
        else: model_args.torch_dtype = torch.float32

    if "quantization_config" in model_args and "load_in_4bit" in model_args.quantization_config:
        model_args.quantization_config.update({
            "bnb_4bit_compute_dtype": determine_compute_dtype(training_args)
        })

    # https://github.com/huggingface/accelerate/issues/1840#issuecomment-1683105994
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    model = AutoModelForSequenceClassification.from_pretrained(
        **model_args,
        num_labels=2,
        device_map=device_map
    )

    if force_resize_token_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    else:
        # resize the embeddings only when necessary to avoid index errors.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

    if pad_token is not None or tokenizer.pad_token is None:
        tokenizer.pad_token = pad_token
        model.config.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    if padding_side is not None:
        tokenizer.padding_side = padding_side

    # postprocess model
    if "bert" not in model_args.pretrained_model_name_or_path: model.enable_input_require_grads()
    if "quantization_config" in model_args: model = prepare_model_for_kbit_training(model)

    # # TODO: integrate with DDP
    # model = torch.compile(model)

    return model, tokenizer


def get_datasets(data_args):
    data_files = dict()
    if "train_file" in data_args: data_files["train_dataset"] = data_args.train_file
    if "eval_file" in data_args: data_files["eval_dataset"] = data_args.eval_file

    file_extension = "json"

    return load_dataset(file_extension, data_files=data_files)


def preprocess_logits_for_metrics(logits, labels=None):
    if isinstance(logits, tuple): logits = logits[0]
    return logits


def compute_metrics(eval_dataset):
    def call(eval_preds):
        preds, labels = eval_preds
        preds = np.argmax(preds, axis=1) if len(preds.shape) > 1 else preds
        labels = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels

        # prediction
        pred_df = eval_dataset.to_pandas() if type(eval_dataset) != pd.core.frame.DataFrame else eval_dataset
        pred_df["preds"] = preds
        pred_df = pred_df[pred_df["preds"] == 1]
        pred_df = pred_df.groupby("query_id")["article_id"].apply(list).reset_index()

        # groundtruth
        groundtruth_df = eval_dataset.to_pandas() if type(eval_dataset) != pd.core.frame.DataFrame else eval_dataset
        groundtruth_df["label"] = labels
        groundtruth_df = groundtruth_df[groundtruth_df["label"] == 1]
        groundtruth_df = groundtruth_df.groupby("query_id")["article_id"].apply(list).reset_index()

        # merge prediction to groundtruth
        groundtruth_df = groundtruth_df.merge(pred_df, on="query_id", how="left", suffixes=("_gt", "_pred"))
        groundtruth_df["article_id_pred"] = groundtruth_df["article_id_pred"].apply(lambda d: d if isinstance(d, list) else [])

        return task3_metrics(groundtruth_df["article_id_gt"], groundtruth_df["article_id_pred"])

    return call


def main():
    # 1. Parse arguments
    args = parse_args()
    model_args = args["model"]
    lora_args = model_args.pop("lora_config", None)
    data_args = args["data"]
    training_args = TrainingArguments(**args["training"])

    model_name = model_args.pretrained_model_name_or_path\
                           .split("/")[-1]\
                           .replace("-", "_")\
                           .lower()
    training_args.run_name = model_name
    training_args.output_dir = os.path.join(training_args.output_dir, model_name)

    # 2. Fix seed
    set_seed(training_args.seed)
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)

    # 3. Model & Tokenizer
    adapter_model_name = model_args.pop("adapter_model_name", None)
    model, tokenizer = get_model_and_tokenizer(model_args, training_args)

    if lora_args is not None:
        # for training
        peft_config = LoraConfig(**lora_args, task_type=TaskType.SEQ_CLS)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif adapter_model_name is not None:
        # for inference
        model = PeftModel.from_pretrained(model, adapter_model_name)
        if "quantization_config" not in model_args: model = model.merge_and_unload()

    # 4. Datasets
    datasets = get_datasets(data_args)
    data_preprocessor = LegalDataPreprocessor(tokenizer, data_args.max_len)
    datasets = data_preprocessor(datasets)

    # 5. Train or Predict
    training_args.max_length = data_args.max_len

    trainer_class = Trainer
    trainer = trainer_class(
        model=model,
        **datasets,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=LegalDataCollatorWithPadding(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_train else None,
        compute_metrics=compute_metrics(datasets["eval_dataset"]) if training_args.do_train else None
    )

    if training_args.do_train:
        model.config.use_cache = False
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if training_args.do_predict:
        logits = trainer.predict(datasets["eval_dataset"]).predictions
        logits = preprocess_logits_for_metrics(logits)

        split = data_args.eval_file.split("/")[-1].split(".")[0]
        np.save(
            os.path.join(training_args.output_dir, f"{split}_logits.npy"),
            logits
        )


if __name__ == "__main__":
    main()
