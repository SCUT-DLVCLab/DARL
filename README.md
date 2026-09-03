# [ECCV 2026] DARL: Efficient Document-to-Markup Generation via Look-Ahead Diffusion Trajectory Sampling

Official PyTorch implementation of **DARL** from the ECCV 2026 paper.

**DARL** is a hybrid diffusion-autoregressive framework that achieves **2.3×** inference speedup on document parsing tasks while maintaining state-of-the-art accuracy. The method introduces:

- **Online Monte Carlo Trajectory Generation (OMTG)**: Real-time trajectory sampling with sliding window mechanism
- **Diffusion Trajectory Preference Optimization (DTPO)**: Reinforcement learning with longest common prefix (LCP) rewards for improved convergence

## 🚀 Key Features

- **2.3× Speedup**: Accelerates document parsing inference through parallel token generation
- **Plug-and-Play**: Compatible with various VLLM backbones (Qwen, InternVL, dots.ocr, etc.)
- **High Accuracy**: Maintains comparable or better accuracy than autoregressive baselines
- **Scalable**: Better performance with larger models (up to 2.78× speedup on 14B models)

## 📋 Requirements

### Environment Setup

First, install the base model [dots.ocr](https://huggingface.co/dots-studio/dots.ocr).

Then install the required dependencies:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### System Requirements

- Python 3.8+
- CUDA 12.1+
- Multi-GPU setup recommended (tested on 8× A100)
- Flash Attention 2.x

## 🔧 Installation

```bash
git clone https://github.com/your-repo/DARL.git
cd DARL
pip install -r requirements.txt
```

## 📊 Dataset Preparation

Organize your document datasets in the following structure:

```
data/
├── magazine/
│   ├── magazine.json
│   └── images/
├── newspaper/
│   ├── newspaper.json
│   └── images/
└── ...
```

Each JSON file should contain entries like:

```json
{
  "prompt": "Recognize image as Markdown format",
  "text": "Ground truth transcription...",
  "img": "images/sample_001.jpg"
}
```

## 🏃 Training

### Single-Node Multi-GPU Training

Launch training with DeepSpeed ZeRO-2:

```bash
accelerate launch --config_file configs/deep_config.yaml train.py
```

### Configuration

Modify training parameters in `train.py`:

- `max_new_tokens`: Sliding window size
- `learning_rate`: Learning rate
- `weight_ar`: DAR loss weight
- `weight_rl`: DTPO loss weight

## 📈 Evaluation

Run inference and evaluation on document parsing benchmarks:

```bash
python eval.py
```

The evaluation script supports:

- OmniDocBench-1.5 benchmark
- olmOCR-Bench benchmark
- Automatic result saving and time tracking
- Batch processing with GPU acceleration

Results will be saved to `results/` directory with individual outputs and summary statistics.

## 🔬 Model Architecture

DARL combines:

1. **Diffusion-based Parallel Generation**: Generates multiple tokens simultaneously using special initialization tokens
2. **Sliding Window Verification**: Verifies generated tokens incrementally from left to right
3. **Hybrid AR-NAR Training**: Combines autoregressive and non-autoregressive objectives

### Key Components

- **Trajectory Generator** (`src/trajectory_generator.py`): Implements OMTG for diverse trajectory sampling
- **Reward Calculator** (`src/reward_calculator.py`): Computes LCP and verification rewards for DTPO
- **DARL Trainer** (`src/dar_trainer.py`): Main training loop combining supervised and RL losses

## 📦 Pre-trained Weights

Pre-trained DARL models are available on Hugging Face:

🤗 **Model Hub**: [https://huggingface.co/Geong/DARL](https://huggingface.co/Geong/DARL)

Download and use:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "your-org/DARL-weights",
    trust_remote_code=True
)
```

## 📖 Citation

If you find DARL useful in your research, please cite:

```bibtex
@inproceedings{darl2026,
  title={DARL: Efficient Document-to-Markup Generation via Look-Ahead Diffusion Trajectory Sampling},
  author={Yang, Wentao and Shi, Yongxin and Tang, Rui and Zhang, Peirong and Wu, Shihang and He, Huiguo and Huang, Zheng and Peng, Dezhi and Liao, Minghui and Jin, Lianwen},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## 🙏 Acknowledgements

This research is built on top of:

- [dots.ocr](https://huggingface.co/dots-studio/dots.ocr): Base document parsing model
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL): Vision-language foundation
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): Distributed training framework

## 📄 License

This project is licensed under the Apache 2.0 License.

## 📧 Contact

For questions and feedback, please contact: wente_young@foxmail.com
