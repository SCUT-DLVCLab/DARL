"""
Dataset Module for DARL Training
Handles multi-source document dataset loading and preprocessing.
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from pprint import pprint


class MotherDataset(Dataset):
    """
    Multi-source dataset manager with per-dataset amplification.
    
    Combines multiple document datasets with configurable sampling rates
    to balance training distribution across different document types.
    """
    
    def __init__(self, all_data_jsons, amplification, tokenizer, processor):
        """
        Args:
            all_data_jsons: List of paths to dataset JSON files
            amplification: List of amplification factors (>1 upsample, <1 downsample)
            tokenizer: Tokenizer for text processing
            processor: Processor for multimodal inputs
        """
        self.all_dataset_instances = []
        self.org_lens = []
        self.global_idx = []
        self.amplification = amplification
        self.tokenizer = tokenizer
        self.processor = processor
        random.seed(45)

        assert len(amplification) == len(all_data_jsons), \
            'Number of datasets and amplification factors must match'

        # Load all subdatasets
        for djson in all_data_jsons:
            this_dataset = SubDataset(djson, tokenizer, processor)
            self.all_dataset_instances.append(this_dataset)
            self.org_lens.append(len(this_dataset))
        
        if os.environ.get("LOCAL_RANK") in ['0', None]:
            print('Initializing all datasets:')
            pprint([(j, a, o) for j, a, o in zip(all_data_jsons, amplification, self.org_lens)])
        
        # Build global index with amplification
        for set_idx, nsample in enumerate(self.org_lens):
            this_idx = list(range(nsample))
            this_set_idx = self._amplify(this_idx, amplification[set_idx])
            self.global_idx += [(set_idx, iidx) for iidx in this_set_idx]
        
        random.shuffle(self.global_idx)
        print(f'Total samples after amplification: {len(self.global_idx)}')

    def _amplify(self, lst, r):
        """
        Amplify or downsample a list by factor r.
        
        Args:
            lst: Original list
            r: Amplification factor (>1 upsample with replacement, <1 downsample)
        """
        original_len = len(lst)
        target_len = int(original_len * r)
        
        if r > 1:
            # Upsample with replacement
            extra_items = random.choices(lst, k=target_len - original_len)
            return lst + extra_items
        elif r < 1:
            # Downsample without replacement
            return random.sample(lst, target_len)
        return lst

    def __getitem__(self, idx):
        """Get item from the appropriate subdataset"""
        didx, iidx = self.global_idx[idx]
        return self.all_dataset_instances[didx][iidx]
    
    def __len__(self):
        return len(self.global_idx)


class SubDataset(Dataset):
    """
    Single document dataset loader.
    
    Handles loading and preprocessing of individual document datasets
    in JSON format with associated images.
    """
    
    def __init__(self, json_path, tokenizer, processor):
        """
        Args:
            json_path: Path to dataset JSON file
            tokenizer: Tokenizer for text processing
            processor: Processor for multimodal inputs
        """
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.processor = processor

        with open(self.json_path, 'r', encoding='utf8') as f:
            self.data = json.load(f)

        self.prefix = os.path.dirname(self.json_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return self.process_func(example)

    def process_func(self, meta):
        """
        Preprocess a single example with image and text.
        
        Args:
            meta: Dictionary containing 'prompt', 'text', and 'img' keys
            
        Returns:
            Dictionary with processed inputs ready for model training
        """
        MAX_LENGTH = 5000
        prompt = meta['prompt']
        output_content = meta['text']
        img = meta['img']

        file_path = f'{self.prefix}/{img}'

        # Construct multimodal conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"{file_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision inputs
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = {key: value.tolist() for key, value in inputs.items()}
        
        # Construct target output with labels
        response = self.tokenizer(f"{output_content}", add_special_tokens=False)
        input_ids = inputs["input_ids"][0] + response["input_ids"] + [self.tokenizer.eos_token_id]
        attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
        labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [self.tokenizer.eos_token_id]

        # Handle sequences exceeding max length
        if len(input_ids) > MAX_LENGTH:
            print(f'Warning: Sequence length {len(input_ids)} exceeds MAX_LENGTH={MAX_LENGTH}')
            # Note: Truncation is handled in trajectory generation loop

        return {
            "input_ids": torch.tensor(input_ids).clone(),
            "attention_mask": torch.tensor(attention_mask).clone(),
            "labels": torch.tensor(labels).clone(),
            "pixel_values": torch.tensor(inputs["pixel_values"]).clone(),
            "image_grid_thw": torch.tensor(inputs["image_grid_thw"]).squeeze(0).clone()
        }
