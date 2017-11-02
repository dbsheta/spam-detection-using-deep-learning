import pandas as pd
from keras.preprocessing import text
import pickle
import os
from bs4 import BeautifulSoup
from email.parser import Parser

parser = Parser()


# Load Data
def process_dataset():
    data = pd.read_csv("data/enron.csv")
    print(f"Total emails: {len(data)}")
    emails = data['msg'].values
    labels = [1 if x == "spam" else 0 for x in data['label'].values]

    # Pre-process Data
    # tokenizer = text.Tokenizer(char_level=True)
    # tokenizer.fit_on_texts(emails)
    # sequences = tokenizer.texts_to_sequences(emails)
    # word2index = tokenizer.word_index
    # num_words = len(word2index)
    # print(f"Found {num_words} unique tokens")
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:"
    char2index = {}
    for i, c in enumerate(alphabet):
        char2index[c] = i + 1

    sequences = []
    for email in emails:
        seq = []
        for c in email:
            if c in char2index:
                seq.append(char2index[c])
        sequences.append(seq)

    with open("data/dataset.pkl", 'wb') as f:
        pickle.dump([sequences, labels, char2index], f)


process_dataset()


def process_email(filename):
    with open(filename) as f:
        email = parser.parse(f)
    cleantext = ""
    if email.is_multipart():
        for part in email.get_payload():
            soup = BeautifulSoup(part.as_string(maxheaderlen=1))
            txt = soup.get_text()
            txt = ' '.join(txt.split())
            i = txt.find("Content-Transfer-Encoding")
            txt = txt[i + len("Content-Transfer-Encoding"):].split(maxsplit=2)[2]
            cleantext += txt

    else:
        soup = BeautifulSoup(email.get_payload())
        txt = soup.get_text()
        txt = ' '.join(txt.split())
        i = txt.find("Content-Transfer-Encoding")
        txt = txt[i + len("Content-Transfer-Encoding"):].split(maxsplit=2)[2]
        cleantext += txt
    print(cleantext)

# for filename in os.listdir("data/infy_spam_emails"):
#     process_email(f"data/infy_spam_emails/{filename}")
