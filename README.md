# VoxShift: State-of-Art Voice Conversion Model
Welcome to VoxShift, a university project at KIT that delves into the fascinating world of voice conversion. Our model strives to alter the vocal characteristics from one speaker to another while maintaining the clarity of the spoken message. The results from our experiments have been sometimes surprising, occasionally educational, and always a learning experience. We encourage you to check out the 'Demo' section, run the code, and engage in the discussions.



## Table of Contents
- :rocket: [Quick Start Guide](#quick-start-guide)
- :floppy_disk: [Dataset](#dataset)
- :building_construction: [Model Architecture](#model-architecture)
- :weight_lifting_man: [Training](#training)
- :bar_chart: [Results](#results)
- :headphones: [Demo](#demo)
- :books: [Sources](#sources)

## Quick Start Guide

We have run all our notebooks in Google Colab. To run them yourself, do the following:
- Access our [Google Drive folder](https://drive.google.com/drive/folders/1V36Ga7vsziCFcNr6wsyagdREbVFFMUI4?usp=sharing) which also contains all our code, as well as the VCTK dataset and model checkpoints. This puts our shared folder (the VC folder) into the "Shared with me" section of your Google Drive.
- Create a shortcut of the VC folder in the root of your Google Drive to ensure that everything loads correctly inside our notebooks (we mount Google Drive inside the notebook). See also [this](https://stackoverflow.com/questions/54351852/accessing-shared-with-me-with-colab) StackOverflow question.
- When mounting your Google Drive inside a notebook, make sure to check the boxes to grant colab permission to access your drive files.

For a quick test of training and evaluation, run the `Training.ipynb` and `Evaluation.ipynb` notebooks. 

For a deeper dive check out the README.md inside out Google Drive folder.

If you want to run our notebooks locally, take this GitHub repository as a starting point. The Dataset is only available in the Google Drive folder, but you can use our data preparation scripts to prepare the data. You might also have to adjust the code and paths to load data or checkpoints.

## Dataset
For our project's training phase, we utilized the Voice Cloning Toolkit (VCTK) Dataset, courtesy of the Centre for Speech Technology Research at the University of Edinburgh. This dataset is well-suited for voice conversion endeavors but equally valuable for other research areas such as speaker identification, text-to-speech synthesis, and accent recognition. Comprising parallel data, the VCTK dataset ensures each speaker contributes a distinct set of newspaper sentences. These sentences were meticulously selected through a greedy algorithm to maximize contextual and phonetic diversity, enabling comprehensive coverage across various voice conversion scenarios.

<img width="1747" alt="Bildschirm­foto 2024-02-20 um 13 53 22" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/00dcb9b7-d19f-4ede-a888-eff16950bf8e">


## Model Architecture
In developing VoxShift, we leveraged the versatility of the VCTK dataset, which provides us with the unique opportunity to explore both parallel and non-parallel training approaches. Parallel Training involves utilizing audio recordings that have identical linguistic content but are spoken by different individuals. This method focuses on mapping the source to the target spectrogram, necessitating temporal alignment to ensure the accuracy of the voice conversion. Non-Parallel Training, on the other hand, employs a more flexible approach. By autoencoding, the model reconstructs the Mel-Spectrogram directly from embeddings, eliminating the need for parallel data. This method does not rely on having two sets of the same spoken content by different speakers, making it significantly more versatile and suited to a wider range of applications. Given the challenges inherent in sourcing parallel training data, we opted for the a non-parallel approach. Non-parallel training stands at the forefront of voice conversion research due to its ability to navigate the absence of parallel data — a common hurdle in the field. This method not only aligns with the state-of-the-art in voice conversion technology but also presents an intriguing avenue for research, offering insights into more dynamic and adaptive model architectures.

A typical model architecture in this domain may incorporate various embeddings to capture the nuances of speech, including:
- Linguistic Embeddings: These encode the textual or phonetic aspects of speech, capturing the content without being influenced by the speaker's unique vocal characteristics.
- Speaker Embeddings: Focus on capturing the unique vocal traits of the speaker, allowing the model to maintain or change the speaker identity in the voice conversion process.
- Prosodic Embeddings: These are used to encapsulate the rhythm, stress, and intonation patterns of speech, contributing to the naturalness and expressiveness of the converted voice.

For VoxShift, we decided to streamline our model by not utilizing prosodic embeddings. This decision was made to reduce complexity and focus our efforts on mastering the core aspects of voice conversion — linguistic and content integrity. By simplifying our model, we aim to achieve a balance between performance and computational efficiency, making VoxShift a robust yet accessible tool for exploring voice conversion technologies.

<img width="1846" alt="Bildschirm­foto 2024-02-20 um 13 58 18" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/3cf4a2a4-353e-4505-a158-d0e33b3bcebd">

Continuing from our approach, we integrated two pretrained models to harness linguistic and speaker embeddings. HuBERT Soft, utilized for its linguistic embeddings, is pretrained on a diverse set of unlabeled audio, enabling the extraction of nuanced linguistic patterns crucial for voice conversion. For speaker characteristics, we leveraged WavLM, which excels in identifying vocal traits across languages and accents, due to its training on a wide-ranging speech corpus. Complementing these, our architecture incorporates HiFi-GAN, a vocoder trained on the LJSpeech dataset, chosen for its ability to produce high-fidelity speech from Mel spectrograms. The integration and final audio synthesis are achieved through a custom-implemented decoder, designed to merge linguistic and speaker embeddings effectively.

<img width="1872" alt="Bildschirm­foto 2024-02-20 um 14 03 14" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/ee99178d-4f90-4e04-a1ea-6c0a8fbcd6da">

Building upon our model's foundation, the decoder is structured as a sequence-to-sequence model. At its core, the encoder segment harnesses 1D convolutional layers, specifically tasked with handling the linguistic embeddings derived from HuBERT Soft. This processed output is then concatenated with the speaker embeddings from WavLM, forming a representation of both linguistic content and speaker identity. This is fed into the decoder segment, which is designed with three LSTM layers. These layers are pivotal in managing sequential data, ensuring that the temporal dynamics of speech are captured and accurately reproduced in the conversion process. An additional enhancement to this architecture is the inclusion of a PreNet. This component takes the target Mel spectrogram and subjects it to a series of linear layers. The PreNet acts as a form of teacher forcing, directly influencing the model with actual output data to refine its predictions, and it injects additional contextual information into the system.

<img width="1814" alt="Bildschirm­foto 2024-02-20 um 14 03 28" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/53c1c507-1592-4798-9a24-d5118949478e">

## Training
In the training of our voice conversion model, an autoencoder approach was employed. This involves an encoder-decoder structure where the encoder compresses the input into a lower-dimensional representation, and the decoder reconstructs the output from this representation. The training process begins by extracting content and speaker embeddings from the input audio. These embeddings are then fed into the decoder model, which aims to output a Mel-spectrogram that closely resembles the target Mel-spectrogram. The fidelity of the generated Mel-spectrogram to the target is measured by a loss function, specifically an L1 loss, or mean absolute error, which guides the optimization of the model parameters. For training, the Adam optimizer is a widely-used choice that computes adaptive learning rates for each parameter, helping to converge faster than traditional stochastic gradient descent. We trained with a batch size of 64, which balances the generalization benefits of larger batch sizes and the stochastic nature of smaller ones, and for 80 epochs to ensure that the model has ample opportunity to learn from the data without overfitting, given the complexity of the task at hand. A learning rate of 0.0004 is chosen as a starting point that is neither too large to overshoot minima nor too small to stall the training process. We also implemented regularization methods such as weight decay of 0.00001 to prevent overfitting by penalizing large weights as well as dropout and instance normalization to help the model generalize better to unseen data by reducing co-adaptation of neurons and stabilizing the distribution of inputs to a layer across the mini-batch.

<img width="1812" alt="Bildschirm­foto 2024-02-20 um 13 59 15" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/95a84401-bf13-4c38-bf97-483824c25ea7">

In our training regimen, a crucial aspect was establishing a robust train-validation-test split that would enable us to accurately gauge the model's performance on both many-to-many and any-to-any voice conversion tasks. The accompanying graphic delineates the distribution strategy for our datasets. We divided the dataset to ensure that the validation and test sets included utterances from both seen and unseen speakers during the training phase. This approach allows us to evaluate the model's ability to convert voices of speakers it has learned from (seen) and speakers it has never encountered during training (unseen). While the subset for unseen speakers is smaller, the flexibility of our model allows for generating multiple combinations between speakers, effectively expanding the test set and providing a comprehensive assessment of the model's generalization capabilities.

<img width="1830" alt="Bildschirm­foto 2024-02-20 um 13 59 34" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/57bce6f7-2703-4e18-81e1-677c2118f8f4">


## Results
In assessing the performance of our voice conversion model, we employed three key metrics:

- Word Error Rate (WER): This metric is derived from automatic speech recognition (ASR) and involves comparing the ASR transcript of the converted audio against the ground truth text of the source audio. WER is calculated as the number of insertions, deletions, and substitutions made by the ASR system, normalized by the number of words in the ground truth. It ranges from 0 to 1, where 0 indicates a perfect transcription with no errors.
- Speaker Similarity (Sim): We measured the speaker similarity by computing the cosine similarity between the speaker embeddings of the converted audio and the target audio. The cosine similarity is a value between -1 and 1, where 1 signifies perfect similarity, 0 indicates no similarity, and -1 represents perfect dissimilarity.
- Mean Opinion Score (MOS): For the subjective evaluation of audio quality, we used the Mean Opinion Score, which reflects human judgments of the converted audio's quality. Participants rate the quality on a scale from 1 to 5, with 5 being the highest quality. In addition to MOS, we also employed Speaker Mean Opinion Score (SMOS) to rate the similarity of speaker identity between the source and target audio.
  
These metrics provided us with a comprehensive understanding of our model's performance, evaluating both the objective accuracy of voice conversion and the subjective quality as perceived by human listeners.

### Base Model Performance Evaluation
For our baseline model, the initial expectation was a potential overfit as the training progressed. Contrary to this, the validation loss continued to converge, suggesting that the model was learning generalizable patterns rather than memorizing the training data. However, to specifically account for the model's performance on unseen speakers, we altered our validation approach. By evaluating solely on unseen speakers, the validation loss exhibited an increase, indicating that while the model was adept at handling seen speakers, its performance on any-to-any voice conversion was less robust.

<img width="1792" alt="Bildschirm­foto 2024-02-20 um 14 15 42" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/d272031c-6ab8-49fa-ba8c-6a0d2dafb10a">
<img width="413" alt="Bildschirm­foto 2024-02-20 um 14 09 57" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/bb6c23ef-d512-4ed3-9829-822877f93c9e">

### Ablations Performance Evaluation
In our ablation study, we incrementally introduced components to the baseline model and observed their impact on performance. First, we incorporated dropout to mitigate overfitting, ensuring the model's generalizability. Next, we increased the hidden dimension of the LSTM layers, which enhanced the model's capacity for the more complex any-to-any voice conversion task. Finally, we added a PostNet module, which further refined the generated Mel-Spectrogram, resulting in improved audio quality. Each step was methodically assessed to measure its contribution to the overall efficacy of the model. 

The ablation study yielded the following key insights: Introducing additional dropout to the model led to improved Word Error Rate (WER) for both many-to-many (m2m) and any-to-any (a2a) voice conversion tasks. Expanding the hidden dimension of the LSTM layers resulted in better Speaker Similarity (Sim) scores for both m2m and a2a scenarios. However, the integration of PostNet did not significantly enhance performance, as evidenced by the overall similar metrics before and after its inclusion.

<img width="523" alt="Bildschirm­foto 2024-02-20 um 14 10 07" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/3f4acc87-284a-4ae2-9540-9bc20377916b">

### Influence of Gender on Basemodel Performance
Our investigation into the influence of gender on model performance revealed a notable trend: conversions where the target speaker was female generally resulted in higher speaker similarity scores and lower WER scores, suggesting a more accurate capture of vocal characteristics and linguistic content. 

<img width="395" alt="Bildschirm­foto 2024-02-20 um 14 10 27" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/3d95965f-4bc3-4371-83cb-9d6d236bc70a">
<img width="400" alt="Bildschirm­foto 2024-02-20 um 14 10 34" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/cdfe041f-8780-4711-8dd5-e78b110a7715">

### Influence of Target Audio Length on Basemodel Performance
In our final set of experiments, we explored the impact of using more extensive audio inputs for creating speaker embeddings. The 'Base + single' configuration used the baseline model with embeddings from individual audio files. We then experimented with 'Base + window', where the embeddings were derived from concatenated audio sequences, including the target and its adjacent utterances, to form a more comprehensive audio context. This was further extended to 'BaseW + window', where the model was not only tested with windowed embeddings but also trained on them. Lastly, 'BaseA + agg' represented a model that used aggregated embeddings, averaging the embeddings from all utterances for each speaker in the VCTK dataset. The results indicated that using windowed and aggregated strategies for speaker embeddings led to improved Speaker Similarity scores for both many-to-many and any-to-any conversion tasks. However, this was accompanied by a deterioration in Word Error Rate (WER), suggesting that while speaker characteristics were captured more accurately, the intelligibility of the speech in the converted audio was somewhat compromised.

<img width="584" alt="Bildschirm­foto 2024-02-20 um 14 10 44" src="https://github.com/AleksandarIlievski/voice-conversion/assets/75620360/7f9f0bf2-a195-4290-89f9-0915be7851f1">

## Demo
We have prepared an interactive demo website that showcases a variety of audio file conversions generated by our baseline model. This demonstration includes examples of both any-to-any and many-to-many voice conversions, organized in descending order of speaker similarity scores — with the highest scores at the top. We invite users to listen and determine for themselves whether they concur with the given scores.

Please note, it is advisable to approach listening with caution, particularly when using headphones. Some of the converted audio files may contain loud or unpleasant artifacts. To prevent potential hearing damage, we recommend starting at a low volume and adjusting as necessary.

[Click here to visit our demo website](https://aleksandarilievski.github.io/voice-conversion/)

## Sources
Our project's foundation is built upon the work done in the soft-vc and Tacotron 2 repository as well as the [Medium article by Piero Esposito](https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091), which served as the initial codebase for our voice conversion model. Additionally, we've explored other resources that implement similar voice conversion approaches, enriching our understanding and capabilities in the domain. These resources have provided us with various perspectives and techniques that complement our primary framework.
- **soft-vc**: A voice conversion model utilizing a soft attention mechanism, created by bshall. This repository contains tools and pre-trained models for voice conversion. Available at: <https://github.com/bshall/soft-vc>
- **Tacotron 2 by NVIDIA**: An implementation of the Tacotron 2 speech synthesis model. This repository includes a robust and flexible framework for text-to-speech systems. Available at: <https://github.com/NVIDIA/tacotron2>
- **knn-vc**: Implements a k-nearest neighbors approach to voice conversion, also developed by bshall. It showcases an alternative method using non-parallel data. Available at: <https://github.com/bshall/knn-vc>
- **AutoVC**: A repository by auspicious3000, featuring an Autoencoder for Voice Conversion. It demonstrates a method for disentangling speaker characteristics from linguistic content. Available at: <https://github.com/auspicious3000/autovc>
- **fastVC**: Developed by fmiotello, this repository provides an efficient voice conversion framework. It is designed for rapid voice conversion without compromising on quality. Available at: <https://github.com/fmiotello/fastVC>

