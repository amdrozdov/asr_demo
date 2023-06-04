import flask
from flask import request, jsonify
import torch
import spacy
import torch.nn as nn
import pickle

app = flask.Flask(__name__)
app.config["DEBUG"] = False

with open('embeddings.pickle', 'rb') as f:
    TEXT = pickle.load(f)


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx = pad_idx
        )
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'))
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(
            torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 16#256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(
    INPUT_DIM, 
    EMBEDDING_DIM, 
    HIDDEN_DIM, 
    OUTPUT_DIM, 
    N_LAYERS, 
    BIDIRECTIONAL, 
    DROPOUT, 
    PAD_IDX
)

MODEL_OUTPUT = 'sentiment_lstm_glove.pt'
model.load_state_dict(torch.load(MODEL_OUTPUT))
nlp = spacy.load('en_core_web_sm')
DEVICE_NAME = 'cpu'
device = torch.device(DEVICE_NAME)

def predict_sentiment(model, sentence):
    model.eval()

    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]

    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

def segmented_sentiment(text):
    data = nlp(text)
    scores = []
    for sent in data.sents:
        scores.append(predict_sentiment(model, str(sent)))
    score = sum(scores) / float(len(scores))
    return score * 2.0 - 1.0


@app.route('/speech', methods=['POST'])
def inference():
    data = request.get_json()
    text = data.get('text', '')
    score = segmented_sentiment(text)
    print("Recognized speech: '%s' with sentiment score: %.2f" % (text, score))
    return jsonify({"status": "OK", "score": score})

app.run()
