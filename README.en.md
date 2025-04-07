# STT Integration LLM

**STT Integration LLM** is a project that integrates Speech-to-Text (STT) models with Large Language Models (LLMs) to enhance the processing and transcription of Vietnamese speech into text. The project leverages state-of-the-art models such as **PhoWhisper-large** from VinAI and **Viet-Mistral-7B-Chat** to achieve high performance in speech recognition and transcription.

## Training Results
Training results for the project can be tracked at:  
[**W&B Run**](https://wandb.ai/huutri231103/slam-llm/runs/9iic653m?nw=nwuserhuutri231103)

## Acknowledgments
This project is built upon the foundation of the following excellent open-source works:  
- [X-LANCE/SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM)  
- [Viet-Mistral/Vistral-7B-Chat](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat)  
- [VinAI/PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large)  

We express our sincere gratitude to the authors and the community for their contributions to these projects.

## License
The project is released under the **MIT License**. See details in the [LICENSE](LICENSE) file.

## Directory Structure
Below is the directory structure of the project:

```
fuutoru-stt-integration-llm/
├── README.md                   # Main documentation
├── LICENSE                     # MIT License
├── scripts/                    # Main scripts
│   ├── config.py               # Configuration for models, training, and data
│   ├── finetune.py             # Script for model fine-tuning
│   ├── inference.py            # Script for inference
│   ├── finetune/               # Shell scripts for fine-tuning
│   │   └── finetune_phowhisper_large_vietmistral_7b.sh
│   └── inference/              # Shell scripts for inference
│       └── inference_phowhisper_large_vietmistral_7b.sh
└── src/                        # Source code
    ├── data/                   # Data processing
    │   ├── concatenator.py     # Data concatenation
    │   └── sampler.py          # Data sampling
    ├── dataset/                # Dataset handling
    │   └── speech_dataset.py   # Speech dataset processing
    ├── models/                 # Models
    │   ├── encoder.py          # Audio encoder (Whisper)
    │   ├── projector.py        # Projector from encoder to LLM
    │   └── stt_model.py        # Integrated STT model
    ├── pipeline/               # Training and inference pipelines
    │   ├── finetune.py         # Fine-tuning pipeline
    │   └── inference_batch.py  # Batch inference pipeline
    ├── policies/               # Optimization policies
    │   ├── activation_checkpointing_functions.py
    │   ├── anyprecision_optimizer.py
    │   ├── mixed_precision.py
    │   └── wrapping.py
    └── utils/                  # Utility functions
        ├── checkpoint_handler.py
        ├── dataset_utils.py
        ├── memory_utils.py
        ├── metric.py
        ├── model_utils.py
        └── train_utils.py
```

## Installation Requirements
To run the project, the following libraries are required:
- Python 3.8+
- PyTorch (CUDA-supported version if using GPU)
- Transformers (Hugging Face)
- Hydra (configuration management)
- Wandb (training monitoring, optional)
- Additional dependencies listed in `requirements.txt` (if provided).

Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage Instructions

### 1. Data Preparation
Input data should be in `.jsonl` format, with each line containing information about the audio file path (`source`) and target text (`target`). Example:
```json
{"source": "/path/to/audio.wav", "target": "Hello, this is an example."}
```

### 2. Model Fine-Tuning
Use the provided shell script to fine-tune the model:
```bash
cd scripts/finetune
bash finetune_phowhisper_large_vietmistral_7b.sh
```
- Update the paths in the script (`speech_encoder_path`, `llm_path`, `train_data_path`, `val_data_path`, etc.) to match your system.
- Adjust parameters in `hydra_args` as needed.

### 3. Inference
Use the shell script for inference:
```bash
cd scripts/inference
bash inference_phowhisper_large_vietmistral_7b.sh
```
- Update the paths (`speech_encoder_path`, `llm_path`, `val_data_path`, `ckpt_path`) and other parameters as required.

### 4. Training Monitoring
If `use_wandb` is enabled in `log_config`, you can monitor the training process on Weights & Biases. Ensure the `WANDB_API_KEY` is provided in the shell script.

## Key Configurations
The `config.py` file contains critical configurations:
- `ModelConfig`: Model settings (encoder, LLM, projector).
- `TrainConfig`: Training parameters (epochs, batch size, learning rate, etc.).
- `DataConfig`: Data settings (paths, input format).
- `FSDPConfig`: Fully Sharded Data Parallel settings (if using FSDP).
- `LogConfig`: Logging and monitoring settings.

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the project.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add XYZ feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Create a Pull Request.

## Contact
For questions or support, please reach out via email: [huutri231103@gmail.com](mailto:huutri231103@gmail.com).
