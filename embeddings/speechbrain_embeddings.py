import torchaudio
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
signal, fs = torchaudio.load('debussy.wav')

# Compute speaker embeddings
embeddings = classifier.encode_batch(signal)
print(embeddings.shape)
print(embeddings)
