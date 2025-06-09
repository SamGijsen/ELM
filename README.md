# EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping

**Authors:** Sam Gijsen, Kerstin Ritter

This repository provides pretrained models from the ICML 2025 paper "EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping".

**Note:** The complete code will be made available here within the next two weeks.

## Pretrained Models

We provide two pretrained EEG encoders:
*   `./pretrained/5s/`: Trained on 5-second EEG epochs and clinical text.
*   `./pretrained/60s/`: Trained on 60-second EEG epochs and clinical text.

### Preprocessing

Both models were trained on data preprocessed with the following minimal steps:
*   **Bandpass filter:** 0.1 - 49Hz
*   **Resampling:** 100Hz
*   **Amplitude clipping:** +/- 800µV
*   **Montage:** 20-channel longitudinal bipolar TCP montage using the channels below.

#### Channels

```
"Fp1-F7", "F7-T3", "T3-T5", "T5-O1",
"Fp2-F8", "F8-T4", "T4-T6", "T6-O2",
"T3-C3", "C3-Cz", "Cz-C4", "C4-T4",
"Fp1-F3", "F3-C3", "C3-P3", "P3-O1",
"Fp2-F4", "F4-C4", "C4-P4", "P4-O2"
```

## Usage

The following Python code demonstrates how to load a pretrained model and extract representations. This example uses the 5-second model, but the 60-second model can be loaded by changing the `config_path` and `weights_path`.

```python
import torch
import yaml

from models.models import EEG_ResNet

config_path = 'pretrained/5s/config_xy.yaml'
weights_path = 'pretrained/5s/model_0_checkpoint.pt'

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

mp = config["model"]
encoder = EEG_ResNet(
    in_channels=mp["in_channels"],
    conv1_params=mp["encoder_conv1_params"],
    n_blocks=mp["encoder_blocks"],
    res_params=mp["encoder_res_params"],
    res_pool_size=mp["encoder_pool_size"],
    dropout_p=mp["encoder_dropout_p"],
    res_dropout_p=mp["res_dropout_p"]
)

DDP = config["training"]["DDP"]
state_dict = torch.load(weights_path, device='cpu')
    
if DDP:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = "module." + key
        new_state_dict[new_key] = value
    state_dict = new_state_dict
encoder.load_state_dict(state_dict)

encoder.eval()

batch_size = 4
n_channels = mp["in_channels"]
n_time_samples = mp["n_time_samples"]

synth_data = torch.randn(batch_size, n_channels, n_time_samples)

with torch.no_grad():
    emb, proj_emb = encoder(synth_data)
        
print(f"Representation shape from encoder: {emb.shape}")
print(f"Projected representation shape from encoder: {proj_emb.shape}")

```


