import datetime
import os

import librosa
import numpy as np
import pandas as pd
from resemblyzer import VoiceEncoder, preprocess_wav
from speechbrain.pretrained import EncoderClassifier
import torchaudio

base_path = "/home/rishav/repo/ml/speaker-recognizer"
data_dir = os.path.join(base_path, "data/shivam_shukla")


def get_file_path_and_speaker(data_dir):
    speaker_name_and_ids = {}
    speaker_counter = 0
    file_path_and_speaker = {}
    for root, dirs, files in os.walk(data_dir):
        path = root.split(os.sep)
        for file in files:
            speaker, file_path = file.split("_")[0].lower(), os.path.join('/'.join(path), file)
            if speaker not in speaker_name_and_ids:
                speaker_counter += 1
                speaker_name_and_ids[speaker] = speaker_counter

            speaker_id = speaker_name_and_ids[speaker]
            file_path_and_speaker[file_path] = speaker_id

    speaker_df = pd.DataFrame.from_dict(file_path_and_speaker, orient='index')
    speaker_df['file_path'] = speaker_df.index
    speaker_df.reset_index(drop=True, inplace=True)
    speaker_df.rename(columns={0: 'speaker_id'}, inplace=True)

    return speaker_df, speaker_name_and_ids


speaker_df, speaker_name_and_ids = get_file_path_and_speaker(data_dir)

print(f'''No. of speakers in the dataset : {len(speaker_df['speaker_id'].unique())}''')

encoder = VoiceEncoder()
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


def extract_features(file_path):
    print(f'''File Path : {file_path}''')
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    # len 40

    # resemblyzer embedding
    wav = preprocess_wav(file_path)
    resemblyzer_embedding = encoder.embed_utterance(wav)
    # len 256

    # ecapa-tdnn embedding
    signal, fs = torchaudio.load(file_path)
    ecapa_embeddings = classifier.encode_batch(signal)
    ecapa_embeddings_flatten = np.array(ecapa_embeddings)[0][0]
    # len 192

    return np.concatenate((resemblyzer_embedding, ecapa_embeddings_flatten, mfccs))


sample_file = os.path.join(base_path, "data/shivam_shukla/TrainingAudio/Aadiksha-007/Aadiksha_1.wav")
sample_feature = extract_features(sample_file)
speaker_df['features'] = speaker_df['file_path'].apply(lambda file_path : extract_features(file_path))
pickle_path = os.path.join(base_path, 'embeddings', f'embedding_resem_ecapa_mfcc_{str(datetime.datetime.today()).split(" ")[0]}.pkl')
speaker_df.to_pickle(pickle_path)

print()
