import os
import pandas as pd

data_dir = "../data/shivam_shukla"

print()


def extract_classical_features(file_path):
    print(f'''File Path : {file_path}''')
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return np.concatenate((mfccs, chroma, mel, contrast, tonnetz))

def get_file_path_and_speaker(data_dir):
    file_path_and_speaker = {}
    for root, dirs, files in os.walk(data_dir):
        path = root.split(os.sep)
        for file in files:
            speaker, file_path = file.split("_")[0].lower(), os.path.join(*path, file)
            file_path_and_speaker[file_path] = speaker

    speaker_df = pd.DataFrame.from_dict(file_path_and_speaker, orient='index')
    speaker_df['file_path'] = df.index
    speaker_df.reset_index(drop=True, inplace=True)
    speaker_df.rename(columns={0: 'speaker_name'}, inplace=True)

    return speaker_df


speaker_df = get_file_path_and_speaker(data_dir)
print()

# def split_stratified_into_train_val_test(df_input, stratify_colname='y',
#                                          frac_train=0.6, frac_val=0.15, frac_test=0.25,
#                                          random_state=None):
#     '''
#     Splits a Pandas dataframe into three subsets (train, val, and test)
#     following fractional ratios provided by the user, where each subset is
#     stratified by the values in a specific column (that is, each subset has
#     the same relative frequency of the values in the column). It performs this
#     splitting by running train_test_split() twice.

#     Parameters
#     ----------
#     df_input : Pandas dataframe
#         Input dataframe to be split.
#     stratify_colname : str
#         The name of the column that will be used for stratification. Usually
#         this column would be for the label.
#     frac_train : float
#     frac_val   : float
#     frac_test  : float
#         The ratios with which the dataframe will be split into train, val, and
#         test data. The values should be expressed as float fractions and should
#         sum to 1.0.
#     random_state : int, None, or RandomStateInstance
#         Value to be passed to train_test_split().

#     Returns
#     -------
#     df_train, df_val, df_test :
#         Dataframes containing the three splits.
#     '''

#     if frac_train + frac_val + frac_test != 1.0:
#         raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
#                          (frac_train, frac_val, frac_test))

#     if stratify_colname not in df_input.columns:
#         raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

#     X = df_input # Contains all columns.
#     y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

#     # Split original dataframe into train and temp dataframes.
#     df_train, df_temp, y_train, y_temp = train_test_split(X,
#                                                           y,
#                                                           stratify=y,
#                                                           test_size=(1.0 - frac_train),
#                                                           random_state=random_state)

#     # Split the temp dataframe into val and test dataframes.
#     relative_frac_test = frac_test / (frac_val + frac_test)
#     df_val, df_test, y_val, y_test = train_test_split(df_temp,
#                                                       y_temp,
#                                                       stratify=y_temp,
#                                                       test_size=relative_frac_test,
#                                                       random_state=random_state)

#     assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

#     return df_train, df_val, df_test




# df_train, df_val, df_test = \
#     split_stratified_into_train_val_test(speaker_df, stratify_colname='speaker_name', frac_train=0.60, frac_val=0.20, frac_test=0.20)



