# README

- `delete_mic2.sh` removes the audiofiles from mic 2 from the vctk audios
- `create_split_json.py` creates a train-val-test split of the dataset
- `extend_split_json.py` extends a train-val-test split with subsets of the validation part
- `create_mel_data.ipynb` creates mel-spectrogram data for all utterances
- `hubert_create_content_embeddings.ipynb` creates content embeddings for all utterances using hubert
- `wavlm_create_speaker_embeddings.ipynb` creates single or windowed speaker embeddings for all utterances using wavlm x vectors
- `create_aggregated_speaker_embeddings.ipynb` creates aggregated speaker embeddings for all speakers
