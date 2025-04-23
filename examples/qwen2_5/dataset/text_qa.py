import logging
import random
from typing import Literal, Optional, Tuple, Union

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class TextQADataset:
    def __init__(
        self,
        dataset_name: str,
        max_token_length: int = 1024,
        ignore_index: int = -100,
        tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        split: Literal["train", "test"] = "train",
        test_size: float = 0.1,
    ) -> None:
        if dataset_name.lower().endswith("pubmed_qa"):
            self.dataset = load_dataset(dataset_name, "pqa_labeled", split="train")
            self.dataset = self.dataset.train_test_split(test_size=test_size, shuffle=False)[split]
            self._dataset_name = "pubmed_qa"
        elif dataset_name.lower().endswith("openorca"):
            self.dataset = load_dataset(dataset_name, split="train")
            self.dataset = self.dataset.train_test_split(test_size=test_size, shuffle=False)[split]
            self._dataset_name = "openorca"
        else:
            raise NotImplementedError

        self.length = len(self.dataset)
        self.max_token_length = max_token_length
        self.ignore_index = ignore_index
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: Union[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        record = self.dataset[int(idx)]

        if self._dataset_name == "pubmed_qa":
            question = record["question"]
            answer = record["long_answer"]
            system_prompt = None
        else:
            question = record["question"]
            answer = record["response"]
            system_prompt = record["system_prompt"]

        try:
            input_ids, labels, attention_mask = preprocess(
                question,
                answer,
                self.tokenizer,
                system_prompt=system_prompt,
                max_length=self.max_token_length,
                ignore_index=self.ignore_index,
            )
        except IndexError as e:
            logger.debug(repr(e))
            return self[random.randint(0, len(self))]

        return input_ids, labels, attention_mask


def preprocess(
    question: str,
    answer: str,
    tokenizer: PreTrainedTokenizer,
    system_prompt: Optional[str] = None,
    max_length: int = 512,
    ignore_index: int = -100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not system_prompt:
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    # set generation prompt to be false, following suggestion from https://huggingface.co/docs/transformers/main/en/chat_templating#model-training
    text = tokenizer.apply_chat_template(conversation=conversation, tokenize=False, add_generation_prompt=False)

    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
        add_special_tokens=True,
    )

    # prepare inputs and the corresponding padding mask
    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0].astype(np.bool_)

    # prepare labels, only the text in the answer content is a valid label
    labels = input_ids.copy()

    content_begin_token = tokenizer.vocab.get("<|im_start|>")
    # right shift 3 tokens as the answer start
    answer_begin_idx = np.where(input_ids == content_begin_token)[0][2].item() + 3
    labels[:answer_begin_idx] = ignore_index

    padding_token = tokenizer.vocab.get(tokenizer.special_tokens_map["pad_token"])
    # left shift 1 tokens as the answer end
    padding_begin_idx = np.where(input_ids == padding_token)[0][0].item() - 1
    labels[padding_begin_idx:] = ignore_index

    return input_ids, labels, attention_mask
