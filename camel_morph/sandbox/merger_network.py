import argparse
import time
import re
import os
import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from vocab import Vocab

class Merger(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.bigram_encoder = Merger.BigramCharRNN(
            args,
            output_vocab_size=2,
            input_vocab=vocab.src.char2id)

    def forward(self, bigrams, lengths_bigram):
        outputs_bigram = self.bigram_encoder(bigrams, lengths_bigram)
        return outputs_bigram


    class BigramCharRNN(nn.Module):
        def __init__(self,
                    args,
                    output_vocab_size,
                    input_vocab):
            super().__init__()
            self.input_vocab = input_vocab
            self.hidden_dim = args.rnn_dim_char
            self.num_layers = args.rnn_layers
            self.word_embeddings = nn.Embedding(len(input_vocab), args.ce_dim)
            self.lstm = nn.LSTM(args.ce_dim,
                            args.rnn_dim_char,
                            num_layers=args.rnn_layers,
                            bidirectional=True)
            self.fc0 = nn.Linear(args.rnn_dim_char, 64)
            self.fc1 = nn.Linear(64, output_vocab_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)

        def forward(self, bigrams, lengths):
            embeds = self.dropout(self.word_embeddings(bigrams))
            packed_embeds = pack_padded_sequence(
                embeds, lengths, enforce_sorted=False)
            _, hidden = self.lstm(packed_embeds)
            hidden = sum(hidden[i][j, :, :]
                            for i in range(2) for j in range(self.num_layers))
            tag_space = self.relu(self.fc0(hidden))
            tag_space = self.fc1(tag_space)
            return tag_space


class Network:
    def __init__(self, args) -> None:
        self.device = torch.device(
            f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
        self.vocab = Vocab.load(args.vocab_path)
        self.load_data(args)
        self.model = Merger(args, vocab=self.vocab).to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3)


    class BigramData(Dataset):
        def __init__(self, data) -> None:
            self.data = data

        def __getitem__(self, index):
            src = torch.tensor(self.data[index][0], dtype=torch.long)
            label = torch.tensor(self.data[index][1], dtype=torch.long)
            return src, label

        def __len__(self):
            return len(self.data)


    def load_data(self, args, train_split=0.8):
        with open(args.data) as f:
            data = [line.split('\t') for line in f.readlines()]
            src = self.vocab.src.words2indices([list(line[0].strip()) for line in data])
            labels = [0 if bigram[1].strip() == '0' else 1 for bigram in data]
            src, labels = src[:args.data_size], labels[:args.data_size]
            data = list(zip(src, labels))
            lengths = [int(len(src)*train_split), int(len(src)*(1-train_split))]
            if sum(lengths) != len(src):
                lengths[0] += len(src) - sum(lengths)
            train_data, dev_data = random_split(data, lengths)

            train_data = Network.BigramData(train_data)
            dev_data = Network.BigramData(dev_data)

        def generate_batch(data_batch):
            src_batch, labels_batch = [], []
            lengths_bigram = []
            for src_item, label_item in data_batch:
                lengths_bigram.append(len(src_item))
                src_batch.append(src_item)
                labels_batch.append(label_item)

            src_batch = pad_sequence(
                src_batch, padding_value=self.vocab.src.word2id['<pad>'])
            labels_batch = torch.stack(labels_batch)
            return (src_batch, labels_batch), lengths_bigram

        self.train_iter = DataLoader(train_data, batch_size=args.batch_size,
                                     shuffle=True, collate_fn=generate_batch)
        self.dev_iter = DataLoader(train_data, batch_size=args.batch_size,
                                     collate_fn=generate_batch)


    @staticmethod
    def epoch_time(start_time: int,
                   end_time: int):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train(self, args):
        metrics = {'train_loss': [], 'dev_loss': [], 'dev_acc': []}
        for epoch in range(args.epochs):
            self.model.train()
            print(f'Epoch {epoch+1}/{args.epochs}')
            epoch_loss = 0
            start_time = time.time()
            for iteration, train_batch in enumerate(self.train_iter):
                (bigrams, labels), lengths = train_batch
                self.model.zero_grad()
                bigrams, labels = bigrams.to(self.device), labels.to(self.device)
                tag_scores = self.model(bigrams, lengths)
                tag_scores = tag_scores.view(-1, tag_scores.shape[-1])
                labels = labels.view(-1)
                loss = self.loss_function(tag_scores, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if iteration and iteration % 100 == 0 and len(self.train_iter) - iteration > 10 \
                        or iteration + 1 == len(self.train_iter):
                    for param_group in self.optimizer.param_groups:
                        lr = param_group['lr']
                    print(
                        f'Batch {iteration}/{len(self.train_iter)-1}\t| Loss {loss.item():.7f} | lr {lr}')
                    metrics['train_loss'].append(epoch_loss / iteration)
            end_time = time.time()
            epoch_mins, epoch_secs = Network.epoch_time(start_time, end_time)
            val_metrics = self.evaluate()
            for m in metrics:
                if m != 'train_loss':
                    metrics[m].append(val_metrics[m])
            print(
                f'Epoch {epoch+1}/{args.epochs} | Time: {epoch_mins}m {epoch_secs}s')
            print(
                f"\tTrain Loss: {metrics['train_loss'][-1]:.7f} | Dev. Loss: {metrics['dev_loss'][-1]:.7f} | Dev. Acc.: {metrics['dev_acc'][-1]:.1%}")
            print()
            self.scheduler.step(metrics['dev_loss'][-1])
        
        self.save_model(args)
        return metrics

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            epoch_loss = 0
            for (bigrams, labels), lengths in self.dev_iter:
                # Loss
                bigrams, labels = bigrams.to(self.device), labels.to(self.device)
                output = self.model(bigrams, lengths)
                loss = self.loss_function(output, labels)
                epoch_loss += loss.item()

                # Accuracy
                output = output.argmax(-1)
                correct += torch.sum(output == labels)
                total += labels.shape[0]
        metrics = {}
        metrics['dev_acc'] = correct / total
        metrics['dev_loss'] = epoch_loss / len(self.dev_iter)
        return metrics

    def predict(self):
        self.model.eval()
        pred, gold = [], []
        inputs_bigram = []
        with torch.no_grad():
            for (bigrams, labels), lengths in self.dev_iter:
                bigrams, labels = bigrams.to(self.device), labels.to(self.device)
                output = self.model(bigrams, lengths)
                output = output.argmax(-1)
                pred += list(output.detach().cpu().numpy())
                gold += list(labels.detach().cpu().numpy())
                inputs_bigram += list(bigrams.permute(1, 0).detach().cpu().numpy())

        return inputs_bigram, gold, pred

    @staticmethod
    def load_model(model_path: str):
        params = torch.load(model_path)
        args = params['args']
        network = Network(args)
        network.model.load_state_dict(params['state_dict'])
        return network

    def save_model(self, args):
        save_path = os.path.join(args.cpt, args.logdir) + '.pt'
        print('Saving model parameters to [%s]\n' % save_path)
        params = {
            'args': args,
            'state_dict': self.model.state_dict()
        }
        torch.save(params, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1048,
                        type=int, help="Batch size.")
    parser.add_argument("--epochs", default=50, type=int,
                        help="Number of epochs.")
    parser.add_argument("--ce_dim", default=32, type=int,
                        help="Word embedding dimension.")
    parser.add_argument("--we_dim", default=64, type=int,
                        help="Word embedding dimension.")
    parser.add_argument("--rnn_dim_char", default=64,
                        type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_dim", default=128,
                        type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_layers", default=1,
                        type=int, help="Number of RNN layers.")
    parser.add_argument("--data_size", default=10000, type=int,
                        help="Maximum number of examples to load.")
    parser.add_argument("--gpu_index", default=0, type=int,
                        help="Index of GPU to be used.")
    parser.add_argument("--vocab", dest='vocab_path',
                        default="/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/root_generator/data_nn_vocab.json", type=str,
                        help="Path to vocab JSON file.")
    parser.add_argument("--data",
                        default="/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/root_generator/data_nn.tsv", type=str,
                        help="Path to file with bigrams dataset.")
    parser.add_argument("--cpt", default='/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/root_generator/model_weights', type=str,
                        help="Directory to save the model checkpoints to.")
    parser.add_argument("--logs", default='/Users/chriscay/Library/Mobile Documents/com~apple~CloudDocs/NYUAD/camel_morph/sandbox_files/root_generator/logs', type=str,
                        help="Directory to save the model checkpoints to.")
    parser.add_argument("--load", default='', type=str,
                        help="Directory to save the model checkpoints to.")
    
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.logdir = "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook").split('.')[0]),
        datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        ",".join(("{}={}".format(re.sub(
            "(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items()) if isinstance(value, int)))
    )

    if not args.load:
        network = Network(args)
        metrics = network.train(args)
        print(metrics)
    else:
        network = Network.load(args.load)
        inputs, gold, pred = network.predict()
        with open(os.path.join(args.logs, args.logdir), 'w') as f:
            for i, result in enumerate(pred):
                bigram = ''.join(
                    list(map(lambda x: network.vocab.src.id2char[x], inputs[0][i])))
                print(re.sub(r'(<pad>)+?', r'', bigram), file=f, end=' | ')
                print(result, 'gold:', gold[i], file=f, end= ' | ')


if __name__ == '__main__':
    main()
