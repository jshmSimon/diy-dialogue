import os
import nltk
import numpy as np
from config import Config


# number of line is 12100
# number of the first utterance is 5

def read_daily_samples():
    ret = []
    file = os.path.join('data', 'samples.txt')
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n','')
            sents = line.split('__eou__')
            conv = []
            for sent in sents:
                sent = sent.lower().strip()
                words = nltk.word_tokenize(sent)
                conv.append(words)
            ret.append(conv)
    return ret


def read_vocab():
    word_list = []
    file = os.path.join('data', 'vocab.txt')
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '').lower().strip()
            word_list.append(line)
    word_to_id = {}
    word_to_id['<pad>'] = 0
    word_to_id['<sos>'] = 1
    word_to_id['<eos>'] = 2
    word_to_id['<unk>'] = 3
    id_to_word = {}
    id_to_word[-1] = '-1'
    id_to_word[0] = '<pad>'
    id_to_word[1] = '<sos>'
    id_to_word[2] = '<eos>'
    id_to_word[3] = '<unk>'
    for i, w in enumerate(word_list):
        word_to_id[w] = i + 4
        id_to_word[i + 4] = w

    return word_to_id, id_to_word


def tokenize_data(data, word_to_id):
    unk_id = word_to_id["<unk>"]
    ret = []
    for conv in data:
        padded_conv = []
        for turn in conv:
            words = [word_to_id.get(w, unk_id) for w in turn]
            padded_conv.append(words)
        ret.append(padded_conv)
    return ret


class DataLoader(object):
    def __init__(self, max_utterance, max_length, split_ratio, batch_size):
        print()
        print(" # Loading DailyDialog Dataset.")
        self.max_utterance = max_utterance
        self.max_length = max_length
        self.split_ratio = split_ratio
        self.batch_size = batch_size

        self.word_to_id, self.id_to_word = read_vocab()
        self.PAD_ID = self.word_to_id["<pad>"]
        self.EOS_ID = self.word_to_id["<eos>"]
        self.SOS_ID = self.word_to_id["<sos>"]
        self.UNK_ID = self.word_to_id["<unk>"]
        self.vocab_size = len(self.word_to_id)
        print(" # Vocab Size: {}\n".format(self.vocab_size))

        data = read_daily_samples()
        self.data = tokenize_data(data, self.word_to_id)
        self.data_size = len(self.data)
        print(" # Data Size: {}\n".format(self.data_size))

        self.train_data, self.test_data = self.split_corpus()
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)
        self.train_batch_num = self.train_size // self.batch_size
        self.test_batch_num = self.test_size // self.batch_size
        print(" # Train Size: {}".format(self.train_size))
        print(" # Test Size: {}\n".format(self.test_size))



    def split_corpus(self):
        train_size = int(self.data_size * self.split_ratio)
        train_data = self.data[:train_size]
        test_data = self.data[train_size:]
        return train_data, test_data


    def create_batches(self, batch_data):
        num_utterance = self.max_utterance + 1 # 等于3
        batch_size = len(batch_data)

        new_batch_data = [] # 每一条数据取出前num_utterance个对话的数据
        for i in range(batch_size):
            conv = batch_data[i][:num_utterance] # 0到2，一共三个utterance
            while len(conv) < num_utterance:
                conv.append([self.UNK_ID])
            assert len(conv) == num_utterance
            new_batch_data.append(conv)

        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []

        for u in range(num_utterance - 1): # 0 到 1
            enc_inp_batches = []
            dec_inp_batches = []
            dec_tar_batches = []
            for b in range(batch_size):
                prev = new_batch_data[b][u][:self.max_length - 1]
                prev = prev[::-1]
                past = new_batch_data[b][u + 1][:self.max_length - 1]
                enc_inp = prev + [self.EOS_ID] + [self.PAD_ID] * (self.max_length - 1 - len(prev))
                dec_inp = [self.SOS_ID] + past + [self.PAD_ID] * (self.max_length - 1 - len(past))
                dec_tar = past + [self.EOS_ID] + [self.PAD_ID] * (self.max_length - 1 - len(prev))
                enc_inp_batches.append(enc_inp)
                dec_inp_batches.append(dec_inp)
                dec_tar_batches.append(dec_tar)

            encoder_inputs.append(enc_inp_batches)
            decoder_inputs.append(dec_inp_batches)
            decoder_targets.append(dec_inp_batches)
        return np.array(encoder_inputs), np.array(decoder_inputs), np.array(decoder_targets)


    def data_generator(self, flag):
        if flag == 'train':
            batch_num = self.train_batch_num
            data = self.train_data
        elif flag == 'test':
            batch_num = self.test_batch_num
            data = self.test_data
        else:
            batch_num = self.data_size // self.data_size
            data = self.data
        for i in range(batch_num):
            start_index = i * self.batch_size
            end_index = min(start_index + self.batch_size, len(data))
            batch_data = data[start_index:end_index]
            encoder_inputs, decoder_inputs, decoder_targets = self.create_batches(batch_data)
            yield encoder_inputs, decoder_inputs, decoder_targets




if __name__ == '__main__':
    # data_loader = DataLoader(max_utterance=Config.num_utterance,
    #                          max_length=Config.max_length,
    #                          split_ratio=Config.split_ratio,
    #                          batch_size=Config.batch_size)
    # count = 0
    # for (encoder_inputs, decoder_inputs, decoder_targets) in data_loader.data_generator(flag='train'):
    a = np.array([12,3,4,5])
    b = a[::-1]
    print(b)
    print(type(b))
