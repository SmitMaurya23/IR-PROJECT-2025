from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from transformers.data.data_collator import pad_without_fast_tokenizer_warning


@dataclass
class LegalDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        need_to_pad = []
        labels = []
        
        for feature in features:
            need_to_pad.append({
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            })

            if "label" in feature:
                labels.append(feature["label"])

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            need_to_pad,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if len(labels) > 0:
            batch["labels"] = torch.tensor(labels, dtype=torch.float16)

        return batch

