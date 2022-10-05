import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

import torch.nn.functional as F

def map_to_array(example):
    speech, _ = librosa.load(example["file"], sr=16000, mono=True)
    example["speech"] = speech
    return example

# load a demo dataset and read audio files
dataset = load_dataset("anton-l/superb_demo", "si", split="test")
dataset = dataset.map(map_to_array)

model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid", num_labels=1251)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")

# compute attention masks and normalize the waveform if needed
inputs = feature_extractor(dataset[:6]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")

print("Type of inputs: ", type(inputs))
labels = torch.tensor(dataset[:6]['label'])
print("Labels shape: ", labels.shape)
outputs = model(**inputs, labels=torch.tensor([2, 2, 2, 2, 2, 2]))
print("Type of outputs: ", type(outputs))
logits = outputs.logits
print("Loss: ", outputs.loss.item())
loss = F.cross_entropy(logits, labels)
print("Other loss: ",loss)
print("Logits: ", logits)
for i in logits:
    print("Len of logits: ", len(i))
predicted_ids = torch.argmax(logits, dim=-1)
print("Predicted IDs: ", predicted_ids)
labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
print("Labels: ", labels)