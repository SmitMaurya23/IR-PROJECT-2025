import numpy as np


SYSTEM_PROMPT = """You will be given a user_query and a legal_article couple.
Your task is to provide a 'relevance' scoring how well the user_query the legal_article related to each other. A higher score means that the legal_article is more relevant to the user_query.
Give your answer as a float on a scale of 0 to 1.0, where 0 means that they are not related at all, and 1.0 means that the legal_article completely and helpfully addresses the user_query even if they are contradicted to each other.

Please think step by step carefully and provide your answer as follows:

Score: (your rating, as a float between 0 and 1.0)"""


USER_PROMPT = """Now here are the user_query and legal_article.

User Query: {query}
Legal Article: {article}

Score:"""


class LegalDataPreprocessor:
    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len


    def one_hot_encode(self, label):
        alpha = 0.0  # Label smoothing factor

        num_classes = 2

        encoded_arr = np.zeros(num_classes, dtype=int) + alpha/(num_classes-1)
        encoded_arr[label] = 1 - alpha

        return encoded_arr.tolist()


    def _encode_sample(self, query, article, label=None):
        model_name = self.tokenizer.name_or_path

        if "t5" in model_name or "bert" in model_name:
            t5_prompt_format = "Query: {query} Document: {article} Relevant:"
            prompt = t5_prompt_format.format(query=query, article=article)
        elif "Embed" in model_name or "bge" in model_name or "e5" in model_name:
            prompt = SYSTEM_PROMPT + "\n" + USER_PROMPT.format(query=query, article=article)
        elif "gemma" in model_name:
            messages = [{
                "role": "user", 
                "content": SYSTEM_PROMPT + "\n" + USER_PROMPT.format(query=query, article=article)
            }]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(query=query, article=article)}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        encoding = self.tokenizer(prompt, max_length=self.max_len, truncation=True)

        if label is not None:
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "label": self.one_hot_encode(int(label))
            }
        
        return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
            }


    def __call__(self, datasets):
        def process(batch):
            if "label" in batch:
                batch_tuple = (batch["query_content"], batch["article_content"], batch["label"])
            else:
                batch_tuple = (batch["query_content"], batch["article_content"])

            res = [self._encode_sample(*sample) for sample in zip(*batch_tuple)]  # list of dicts
            res = {k: [d[k] for d in res] for k in res[0]}  # convert to dict of list

            return res

        
        return datasets.map(process, batched=True)
