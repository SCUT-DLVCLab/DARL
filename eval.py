"""
Evaluation Script for DARL
Runs inference on document parsing benchmarks and saves results.
"""

import os
import json
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


class DARLInference:
    """
    DARL Inference Engine with speculative decoding.
    """
    
    def __init__(self, model_path, device='cuda:0', max_pixels=2500*28*28):
        """
        Initialize DARL inference model.
        
        Args:
            model_path: Path to trained DARL model
            device: Device to run inference on
            max_pixels: Maximum image resolution
        """
        print(f"Loading model from {model_path}...")
        self.device = device
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            max_pixels=max_pixels,
            trust_remote_code=True
        )
        
        self.tokenizer = self.processor.tokenizer
        print("Model loaded successfully!")
    
    def inference(self, image_path, prompt, max_new_tokens=2048):
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for the task
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (generated_text, inference_time)
        """
        # Construct messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate with DARL
        start_time = time.perf_counter()
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text, inference_time


def evaluate_dataset(model_path, dataset_json, output_dir, 
                     device='cuda:0', max_new_tokens=2048):
    """
    Evaluate DARL on a dataset.
    
    Args:
        model_path: Path to trained model
        dataset_json: Path to dataset JSON file
        output_dir: Directory to save results
        device: Device for inference
        max_new_tokens: Maximum tokens to generate per sample
    """
    # Load dataset
    with open(dataset_json, 'r', encoding='utf8') as f:
        dataset = json.load(f)
    
    # Initialize model
    inferencer = DARLInference(model_path, device=device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get dataset directory
    dataset_dir = os.path.dirname(dataset_json)
    
    # Run inference
    total_time = 0
    results = []
    
    for item in tqdm(dataset, desc="Evaluating"):
        img_path = os.path.join(dataset_dir, item['img'])
        prompt = item.get('prompt', 'Extract all text from this document.')
        img_name = os.path.basename(item['img'])
        
        # Run inference
        try:
            output_text, inference_time = inferencer.inference(
                img_path, prompt, max_new_tokens
            )
            
            # Save result
            result = {
                'image': img_name,
                'output': output_text,
                'time': inference_time,
                'ground_truth': item.get('text', '')
            }
            
            # Save individual result
            with open(f'{output_dir}/{img_name}.json', 'w', encoding='utf8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            with open(f'{output_dir}/{img_name}.md', 'w', encoding='utf8') as f:
                f.write(output_text)
            
            results.append(result)
            total_time += inference_time
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    # Save summary
    summary = {
        'total_samples': len(results),
        'total_time': total_time,
        'average_time': total_time / len(results) if results else 0,
        'model_path': model_path,
    }
    
    with open(f'{output_dir}/summary.json', 'w', encoding='utf8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nEvaluation completed!")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Average time: {summary['average_time']:.2f}s")
    print(f"Results saved to: {output_dir}")


def main():
    """Main evaluation function"""
    
    # Configuration
    model_path = "checkpoints/your_trained_model"  # Update with your model path
    
    # Evaluate on OmniDocBench-1.5
    print("=" * 50)
    print("Evaluating on OmniDocBench-1.5")
    print("=" * 50)
    evaluate_dataset(
        model_path=model_path,
        dataset_json="data/OmniDocBench1_5/test.json",
        output_dir="results/omnidocbench_1_5",
        device='cuda:0',
        max_new_tokens=2048
    )
    
    # Evaluate on olmOCR-Bench
    print("\n" + "=" * 50)
    print("Evaluating on olmOCR-Bench")
    print("=" * 50)
    evaluate_dataset(
        model_path=model_path,
        dataset_json="data/olmOCR/test.json",
        output_dir="results/olmocr_bench",
        device='cuda:0',
        max_new_tokens=2048
    )


if __name__ == '__main__':
    main()
