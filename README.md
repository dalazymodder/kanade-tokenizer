# Kanade: Compact Linguistically Rich Speech Tokens for Spoken Language Models

Kanade is a speech tokenizer that encodes speech into compact content tokens and global embeddings and decodes them back to mel spectrograms.

## Project Structure

- `kanade_tokenizer/`: The main module.
  - `model.py`: Defines Kanade's model architecture and data flow. This is the core model code and should be enough for inference.
  - `pipeline.py`: A wrapper for training related logic, written as `LightningModule`. This is expected to be used together with `cli.py`.
  - `data/`: Datasets and data loading logic, written as `LightningDataModule`.
  - `module/`: All neural network components.
- `config/`: Model config files.
  - `model/`: Minimal config files for inference.
  - `train/`: Config files for training.
- `script/`: Useful scripts.
  - `dump_dataset.py`: Dump the metadata of a directory of audio files into CSV format for data loading during training.
  - `export_safetensors.py`: Export a more lightweight Safetensors checkpoint from a Lightning checkpoint.
- `demo.ipynb`: A demo notebook showing how to use the model for inference.
- `cli.py`: Main entrypoint for the Lightning framework, used for training, validation, testing, and prediction.

## Models

| Model  | Token Rate | Vocab Size | Bit Rate | Dataset  | SSL Encoder | Vocoder     | Parameters |
| ------ | ---------- | ---------- | -------- | -------- | ----------- | ----------- | ---------- |
| `12hz` | 12.5 Hz    | 12800      | 171 bps  | LibriTTS | WavLM-base+ | Vocos 24kHz | 120M       |
| `25hz` | 25 Hz      | 12800      | 341 bps  | LibriTTS | WavLM-base+ | Vocos 24kHz | 118M       |

Weights files will be made available soon.

## Installation

For simple inference, you can install Kanade tokenizer from this repository directory as an editable package to your virtual environment:

```bash
# In your own project's virtual environment
uv add --editable /path/to/kanade-tokenizer
# or using pip
pip install -e /path/to/kanade-tokenizer
```

If you want to train the model, you can directly work in the repository directory after installing the dependencies (with extra `train` dependencies):
```bash
cd /path/to/kanade-tokenizer
# In the repository directory
uv sync --extra train
# or using pip
pip install -e ".[train]"
```

> [!IMPORTANT]
We use [FlashAttention](https://github.com/Dao-AILab/flash-attention) for efficient local window attention in our training. We recommend installing it following the instructions in their repository to get the best performance and the closest match to our setup. The model will fall back to regular PyTorch SDPA implementation if FlashAttention is not available. In this case, we cannot guarantee the same quality as reported in the paper.  
If using uv, you can install FlashAttention like: `uv pip install flash-attn --no-build-isolation`.

## Usage

Example code to load the model and do inference:

```python
from kanade_tokenizer import KanadeModel, load_audio, load_vocoder, vocode

# Load Kanade model
config_path = "config/model/12hz.yaml"
weights_path = "weights/12hz.safetensors"
model = KanadeModel.from_pretrained(config_path, weights_path=weights_path)
model = model.eval().cuda()

# Load vocoder
vocoder = load_vocoder().cuda()

# Load audio (samples,)
audio = load_audio("path/to/audio.wav", sample_rate=model.config.sample_rate).cuda()

# Extract features
features = model.encode(waveform)

# Synthesize audio from extracted features
mel_spectrogram = model.decode(
    content_token_indices=features.content_token_indices, # (seq_len,)
    global_embedding=features.global_embedding, # (dim,)
) # (n_mels, T)

# Resynthesize waveform using vocoder
resynthesized_waveform = vocode(vocoder, mel_spectrogram.unsqueeze(0)) # (1, samples)
```

For more routines such as voice conversion, please refer to the `demo.ipynb` notebook.

## Training

The training relies on the Lightning framework. The general form of Lightning commands is `python cli.py {fit|test|validate|predict} --config config.yaml`. A config file contains the information the framework needs: model architecture, training configuration, data loading. The fields can be overridden by extra command flags.

First, prepare the dataset. You can use the `script/dump_dataset.py` script to dump a CSV file containing the metadata of audio files for data loading:

```bash
# If not using uv, replace `uv run` with `python` in the following commands
uv run script/dump_dataset.py /path/to/LibriTTS --pattern "train-*/**/*.wav" -o data/libritts_train.csv
uv run script/dump_dataset.py /path/to/LibriTTS --pattern "dev-*/**/*.wav" -o data/libritts_dev.csv
uv run script/dump_dataset.py /path/to/LibriTTS --pattern "test-clean/**/*.wav" -o data/libritts_test.csv
```

Then, modify the config file in `config/train/` for dataset paths and other training settings as needed. Apart from common hyperparameters, there are some paths you may want to change:

- `trainer.default_root_dir`: Directory to save checkpoints and logs.
- `data.{train|val|test}_config.csv_path`: Path to the CSV file containing dataset metadata, generated in the previous step.
- `data.{train|val|test}_config.audio_root`: Root directory of the audio files. This should be consistent with the directory used when dumping the CSV files. (You can check the CSV files to see the relative paths.)
- `model.pipeline_config.ckpt_path`: Optional path to a checkpoint to load weights and optimizer states from. Used for post-training in our case.

Finally, run the training command:

```bash
uv run cli.py fit --config config/train/12hz_pretrain.yaml
```

After the pretraining, you can continue to run the post-training by changing the config file and the checkpoint path accordingly:

```bash
uv run cli.py fit --config config/train/12hz_gan.yaml
```

If you want to use WandB for logging, you can set the WandB logger object provided by Lightning to `trainer.logger` in the config file:
```yaml
trainer:
  logger:
  class_path: lightning.pytorch.loggers.WandbLogger
  init_args:
    name: run_name
    project: project_name
    save_dir: /path/to/log/dir
```

### Fine-tuning

To fine-tune on your own dataset, you can create a new config file based on the existing ones and change the dataset paths. In this case, specify the `model.pipeline_config.ckpt_path` field to load the pretrained weights. The checkpoint can be either the Lightning checkpoint saved during training or a Safetensors checkpoint.
