

# Whisper model in Pytorch from scratch implementation

Trained a small whisper model coded and trained from scratch in Pytorch 



[Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)

## ModelArgs Hyperparameters

| Parameter               | Value                  | Description                                                                 |
|-------------------------|------------------------|-----------------------------------------------------------------------------|
| `batch_size`            | 64                     | The number of samples processed before the model is updated.                |
| `max_lr`                | 2e-4                   | Maximum learning rate.                                                      |
| `dropout`               | 0.1                    | Dropout rate for regularization.                                            |
| `epochs`                | 10                     | Number of training epochs.                                                  |
| `block_size`            | 64                     | Sequence length (number of tokens or time steps).                           |
| `tgt_vocab_size`        | 50262     | Size of the target vocabulary.                                              |
| `embeddings_dims`       | 384                    | Dimensionality of token embeddings.                                         |
| `attn_dropout`          | 0.1                    | Dropout rate for attention layers.                                          |
| `no_of_heads`           | 6                      | Number of attention heads in multi-head attention.                          |
| `no_of_decoder_layers`  | 6                      | Number of decoder layers in the model.                                      |
| `weight_decay_optim`    | 0.01                   | Weight decay for the optimizer.                                             |
| `log_mel_features`      | 80                     | Number of Mel spectrogram features.                                         |
| `kernel_size`           | 3                      | Kernel size for convolutional layers.                                       |
| `stride`                | 2             | Stride for convolutional layers.                                            |
| `sr`                    | 16000                  | Sampling rate of the audio.                                                 |
| `device`                | `'cuda:0'`             | Device to run the model on (e.g., GPU).                                     |
| `SAMPLING_RATE`         | 16000                  | Sampling rate of the audio.                                                 |
| `N_MELS`                | 80                     | Number of Mel bins in the spectrogram.                                      |
| `WINDOW_DURATION`       | 0.025                  | Duration of the analysis window in seconds (25 ms).                         |
| `STRIDE_DURATION`       | 0.010                  | Stride between consecutive windows in seconds (10 ms).                      |
| `max_t`                 | 500                    | Maximum time steps in the spectrogram.                                      |
| `n_channels`            | 80                     | Number of channels in the input spectrogram.                                |
| `hidden_dim`            | 4 * `embeddings_dims`  | Number of neurons in the feed-forward network (FFN).                        |
"""

### Dataset

[Gigaspeech](https://huggingface.co/datasets/speechcolab/gigaspeech)

Used the 'xs' snapshot.

### Frameworks:
**Pytorch**


### Epochs/Steps
Epochs (train) = 10

Val iterations = every epoch


### Loss Curves

![Train and Val loss curves](img/loss.jpg)



