# VoxShift: State-of-Art Voice Conversion Model
Welcome to VoxShift, a university project at KIT that delves into the fascinating world of voice conversion. Our model strives to alter the vocal characteristics from one speaker to another while maintaining the clarity of the spoken message. The results from our experiments have been sometimes surprising, occasionally educational, and always a learning experience. We encourage you to check out the 'Demo' section, run the code, and engage in the discussions.



## Table of Contents
- [Quick Start Guide](#quick-start-guide)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Demo](#demo)

## Quick Start Guide


## Dataset
For our project's training phase, we utilized the Voice Cloning Toolkit (VCTK) Dataset, courtesy of the Centre for Speech Technology Research at the University of Edinburgh. This dataset is well-suited for voice conversion endeavors but equally valuable for other research areas such as speaker identification, text-to-speech synthesis, and accent recognition. Comprising parallel data, the VCTK dataset ensures each speaker contributes a distinct set of newspaper sentences. These sentences were meticulously selected through a greedy algorithm to maximize contextual and phonetic diversity, enabling comprehensive coverage across various voice conversion scenarios.

<img width="1747" alt="Bildschirm­foto 2024-02-20 um 13 53 22" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/00dcb9b7-d19f-4ede-a888-eff16950bf8e">


## Model Architecture
<img width="1846" alt="Bildschirm­foto 2024-02-20 um 13 58 18" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/3cf4a2a4-353e-4505-a158-d0e33b3bcebd">
<img width="1872" alt="Bildschirm­foto 2024-02-20 um 14 03 14" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/ee99178d-4f90-4e04-a1ea-6c0a8fbcd6da">
<img width="1814" alt="Bildschirm­foto 2024-02-20 um 14 03 28" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/53c1c507-1592-4798-9a24-d5118949478e">

## Training
<img width="1812" alt="Bildschirm­foto 2024-02-20 um 13 59 15" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/95a84401-bf13-4c38-bf97-483824c25ea7">
<img width="1830" alt="Bildschirm­foto 2024-02-20 um 13 59 34" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/57bce6f7-2703-4e18-81e1-677c2118f8f4">


## Results
### Base Model Performance Evaluation
<img width="1792" alt="Bildschirm­foto 2024-02-20 um 14 15 42" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/d272031c-6ab8-49fa-ba8c-6a0d2dafb10a">


<img width="413" alt="Bildschirm­foto 2024-02-20 um 14 09 57" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/bb6c23ef-d512-4ed3-9829-822877f93c9e">

### Ablations Performance Evaluation
<img width="523" alt="Bildschirm­foto 2024-02-20 um 14 10 07" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/3f4acc87-284a-4ae2-9540-9bc20377916b">

### Influence of Gender on Basemodel Performance
<img width="395" alt="Bildschirm­foto 2024-02-20 um 14 10 27" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/3d95965f-4bc3-4371-83cb-9d6d236bc70a">
<img width="400" alt="Bildschirm­foto 2024-02-20 um 14 10 34" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/cdfe041f-8780-4711-8dd5-e78b110a7715">

### Influence of Target Audio Length on Basemodel Performance
<img width="584" alt="Bildschirm­foto 2024-02-20 um 14 10 44" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/7f9f0bf2-a195-4290-89f9-0915be7851f1">


## Demo
[Click here to visit our website](https://aleksandarilievski.github.io/voice-conversion/)
