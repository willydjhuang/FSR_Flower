from collections import OrderedDict
import warnings

import os
import flwr as fl
import torch
import librosa
import pprint
from datasets import load_dataset, load_metric, Dataset

from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from transformers import AdamW

from tqdm import tqdm
from argparse import ArgumentParser, Namespace

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

net = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")

def map_to_array(example):
    speech, _ = librosa.load(example["file"], sr=16000, mono=True)
    example["speech"] = speech
    return example

def collate_fn(batch):

    inputs, targets = [], []

    for data in batch:
        inputs += [data['speech']]
        targets += [torch.tensor(data['label'])]

    # Group the list of tensors into a batched tensor
    inputs = feature_extractor(inputs, sampling_rate=16000, padding=True, return_tensors="pt")
    targets = torch.stack(targets)

    return inputs, targets

def train(net, trainloader, epochs, accum_iter=32):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch_idx, (data, target) in enumerate(tqdm(trainloader)):
            outputs = net(**data.to(DEVICE))

            # release gpu memory
            del outputs.hidden_states
            torch.cuda.empty_cache()
            
            loss = F.cross_entropy(outputs.logits.to('cpu'), target)
            loss = loss / accum_iter
            loss.backward()
            if ((batch_idx + 1) % accum_iter == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()

def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for data, target in testloader:
        with torch.no_grad():
            outputs = net(**data.to(DEVICE))

        # release gpu memory
        del outputs.hidden_states
        torch.cuda.empty_cache()
        
        logits = outputs.logits.to('cpu')
        loss += F.cross_entropy(logits, target)
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=target)

    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy

def main(args):
    dataset = load_dataset("superb", 
                            "si", 
                            split="train",
                            cache_dir="/tmp2/b08902028/cache/",
                            data_dir="/tmp2/b08902028/SR_federated/VoxCeleb1")
    dataset = dataset.map(map_to_array)
    dataset = dataset.remove_columns(["file", "audio"])

    train_set = []
    val_set = []
    # 2 is the stating index of VoxCeleb
    # Other speakers(clients) in federated system
    other_speakers = [i for i in range(2, args.speaker_num + 2) if i != args.speaker_id]

    for i, data in enumerate(dataset):
        if data['label'] <= args.dataset_size:
            if data['label'] not in other_speakers:
                train_set.append(data)
            val_set.append(data)
        else: 
            break

    print("Finishg trimming dataset...")

    trainloader = DataLoader(
        train_set,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=True
    )

    print("Finish loading training data...")

    testloader = DataLoader(
        val_set, 
        batch_size=1, 
        collate_fn=collate_fn
    )


    net.to(DEVICE)
    # train(net, trainloader, epochs=args.epoch)

    # Flower client
    class IMDBClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("Training Started...")
            train(net, trainloader, epochs=args.epoch)
            print("Training Finished.")
            loss, accuracy = test(net, testloader)
            print(f"Accuracy for local model with client id {args.speaker_id}: ", accuracy)
            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=IMDBClient())

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--speaker_id",
        help="Speaker index in VoxCeleb dataset",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--speaker_num",
        help="Number of speakers in federated system",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--dataset_size",
        help="Number of speakers in dataset",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--epoch",
        help="Number of epochs",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    return args


# 2, 3, 4, 5, 6, 7
# 2, 6, 7
# 3, 6, 7
# 4, 6, 7
# 5, 6, 7


if __name__ == "__main__":
    args = parse_args()
    main(args)
