# Pulse-Coupled-Filtering-for-Infrared-Small-Target-Detection
---
Yang Zhang, Meibin Qi, Kunyuan Li, Di Wang, Shuo Zhuang*

### September 4, 2026
Our paper has been accepted by TCSVT.

## Overall Framework
<img width="1321" height="749" alt="496c8ba7-6d5e-4a47-9c5c-e0fe0842bd3d" src="https://github.com/user-attachments/assets/2c42c09f-a2df-46a3-b406-18aba5288edf" />


## Main Contribution
1) We propose PCFNet that highlights feature purification  before interaction in cross-stage transmission. By introducing adaptive filtering into skip connections, PCFNet suppresses clutter propagation at the source and improves the quality of features used for subsequent decoding and fusion.
2) We design a DT-PCF module, which incorporates learnable threshold evolution, neighborhood-coupled pulse dynamics, and target-aware enhancement into skip connections. DT-PCF can adaptively suppress background clutter while preserving weak target responses, providing purified and discriminative feature representations.
3) We develop a CFAE module to alleviate the semantic gap between shallow spatial details and deep semantic representations. By using deep features to guide the enhancement of shallow features, the proposed module promotes more effective cross-level feature alignment and improves target-aware feature fusion.

### Our project has the following structure:
```
├──./datasets/
  │    ├── IRSTD-1K
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  |    |    |    ├── ········
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  |    |    |    ├── ········
  │    │    ├── train.txt
  │    │    │── test.txt
  │    ├── NUDT-SIRST
  │    │    ├── images
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  |    |    |    ├── ········
  │    │    ├── masks
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  |    |    |    ├── ········
  │    │    ├── train.txt
  │    │    │── test.txt
  │    ├── NUAA-SIRST
  │    │    ├── images
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  |    |    |    ├── ········
  │    │    ├── masks
  │    │    │    ├── Misc_1.png
  │    │    │    ├── Misc_2.png
  |    |    |    ├── ········
  │    │    ├── train.txt
  │    │    │── test.txt
```
## Special thanks：
This code is highly borrowed from [SCT](https://github.com/xdFai/SCTransNet).Thanks to Shuai Yuan.<br>


 
