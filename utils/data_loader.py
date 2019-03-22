import torchtext
import pickle
from utils.globals import Globals as glob
import os
import torch




class data_row():

    def __init__(self, sample):

        self.s1 = [s for s in sample[0].replace(',', ' ,').split(' ')]
        try:
            self.s1[-1] = self.s1[-1] if self.s1[-1][-1] != '.' else self.s1[-1][:-1]
        except IndexError:
            pass

        self.s2 = [s for s in sample[1].replace(',', ' ,').split(' ')]
        try:
            self.s2[-1] = self.s2[-1] if self.s2[-1][-1] != '.' else self.s2[-1][:-1]
        except IndexError:
            pass
        self.label = float(sample[2])


class DataLoader():

    def load_data(self, corpus_names, is_train,debug=None):

        data = []
        cutoffs = []
        prev_cutoff = 0
        for corpus_name in corpus_names:
            path = os.path.join(glob.data_dir, corpus_name)
            this_data = pickle.load(open(path, 'rb'), encoding='utf-8')

            data.extend([data_row(sample) for sample in this_data])
            cutoffs.append((prev_cutoff, len(data)))

            prev_cutoff = len(data)
        if debug:
            data = data[0:debug]

        tokenize = lambda x: x.split(' ')
        sort_key = lambda x: sum([len(x.s1.split(' ')), len(x.s2.split(' '))]) / 2.0

        text_field = torchtext.data.Field(tokenize=tokenize)
        label_field = torchtext.data.Field(sequential=False,
                                           use_vocab=False,
                                           dtype=torch.float)

        fields = {'s1': text_field,
                  's2': text_field,
                  'label': label_field}

        data = torchtext.data.Dataset(examples=data, fields=fields)

        text_field.build_vocab(data)
        vectors = torchtext.vocab.Vectors(name='GoogleNews-vectors-negative300.txt', cache=glob.data_dir)
        text_field.vocab.load_vectors(vectors=vectors)

        batch_list = []
        for _ in range(len(corpus_names)):
            if debug:
                this_data = data
            else:
                this_data = torchtext.data.Dataset(examples=data[cutoffs[_][0]:cutoffs[_][1]], fields=fields)

            if 'pre' in corpus_names[_]:
                batches = torchtext.data.BucketIterator(dataset=this_data,
                                                  batch_size=glob.pre_train_batch_size,
                                                  shuffle=True,
                                                  sort_key=sort_key)
            else:
                batches = torchtext.data.BucketIterator(dataset=this_data,
                                                  batch_size=glob.train_batch_size if is_train else glob.test_batch_size,
                                                  shuffle=True,
                                                  sort_key=sort_key)

            batch_list.append(batches)


        return (text_field.vocab, batch_list)