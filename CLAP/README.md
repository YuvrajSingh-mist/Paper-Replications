

# Whisper model in Pytorch from scratch implementation

Implementation of CLAP model coded from scratch in Pytorch 



[CLAP : LEARNING AUDIO CONCEPTS FROM NATURAL LANGUAGE SUPERVISION](https://arxiv.org/pdf/2206.04769)

## ModelArgs Hyperparameters

### Hyperparameters

| Parameter             | Value       | Description                                                                 |
|-----------------------|-------------|-----------------------------------------------------------------------------|
| `epochs`              | 30          | Number of training epochs.                                                  |
| `text_embeddings`     | 768         | Dimensionality of text embeddings.                                          |
| `audio_embeds`        | 2048        | Dimensionality of audio embeddings.                                         |
| `block_size`          | 100         | Size of input blocks (e.g., sequence length).                               |
| `batch_size`          | 32          | Number of samples per batch.                                                |
| `lr`                  | 4e-4        | Learning rate for the main model.                                           |
| `device`              | `'cuda:0'`  | Device to run the model on (e.g., GPU).                                     |
| `SAMPLING_RATE`       | 44100       | Sampling rate of the audio (in Hz).                                         |
| `N_MELS`              | 64          | Number of mel-spectrogram bins.                                             |
| `max_t`               | 500         | Maximum time steps for sequences.                                           |
| `n_channels`          | `N_MELS`    | Number of channels in the input (same as `N_MELS`).                         |
| `window_size`         | 1024        | Window size for STFT (Short-Time Fourier Transform).                        |
| `hop_size`            | 320         | Hop size for STFT.                                                         |
| `mel_bins`            | `N_MELS`    | Number of mel bins (same as `N_MELS`).                                      |
| `fmin`                | 50          | Minimum frequency for mel-spectrogram computation.                          |
| `fmax`                | 8000        | Maximum frequency for mel-spectrogram computation.                          |
| `output_embeddings`   | 1024        | Dimensionality of output embeddings.                                        |
| `head_lr`             | 1e-3        | Learning rate for the task-specific head.                                   |
| `audio_encoder_lr`    | 1e-4        | Learning rate for the audio encoder.                                        |
| `text_encoder_lr`     | 1e-5        | Learning rate for the text encoder.                                         |

### Dataset

[Gigaspeech](https://huggingface.co/datasets/speechcolab/gigaspeech)

Used the 'xs' snapshot.

### Frameworks:
**Pytorch**


### NOTE
The loss was stagged at 2.079 -loge(1/8), that is, the logits tend to be too small for softmax to outputs anythign except uniform probs. Pls let me know where am I making a mistake.



