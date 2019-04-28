import torchtext
import pickle
from utils.globals import Globals as glob
from unit_tests.test_data import TestData
import os
import torch

with open('data/dwords.p', 'rb') as f:
    w2v_cleaner = pickle.load(f, encoding='utf-8')
    # just happened to catch these
    w2v_cleaner['receiveing'] = 'receiving'
    w2v_cleaner['homossexuality'] = 'homosexuality'
    w2v_cleaner['promiscous'] = 'promiscuous'
    w2v_cleaner['rightfuly'] = 'rightfully'
    w2v_cleaner['celabrate'] = 'celebrate'
    w2v_cleaner['unafecting'] = 'unaffecting'
    w2v_cleaner['senteces'] = 'sentences'
    w2v_cleaner['deaht'] = 'death'

class data_row():

    def clean_sent(self,sent):
        clean_sent = []
        for word in sent:
            # not in, so try to clean as much as possible
            if word not in w2v_cleaner:
                clean = ''
                for char in word:
                    if char.isalnum():
                        clean += char
                if len(clean) > 0:
                    if clean in w2v_cleaner:
                        clean_sent.append(w2v_cleaner[clean])
                    elif clean.lower() in w2v_cleaner:
                        clean_sent.append(w2v_cleaner[clean.lower()])
                    else:
                        clean_sent.append(clean)
            else:
                clean_sent.append(w2v_cleaner[word])

        return clean_sent

    def __init__(self, sample):

        sent = list(filter(None, sample[0].split(' ')))
        self.s1 = self.clean_sent(sent)

        sent = list(filter(None, sample[1].split(' ')))
        self.s2 = self.clean_sent(sent)

        self.label = torch.tensor(float(sample[2]),dtype=torch.float32).to(glob.device)

class DataLoader():

    def __init__(self):
        # keep vectors cached
        self.vectors = torchtext.vocab.Vectors(name='GoogleNews-vectors-negative300.txt', cache=glob.data_dir)
        self.tests = TestData()

    def load_data(self, corpus_names, for_reg_model = False):
        words = set()
        data = []
        cutoffs = []
        prev_cutoff = 0
        glob.data_integrity_table = dict()
        val_integrity_table = dict()

        # load all data

        for corpus in corpus_names:
            path = os.path.join(glob.data_dir, corpus)
            # load data
            d = pickle.load(open(path, 'rb'), encoding='utf-8')
            # add samples
            for row in d:
                sample = data_row(row)
                key = ' '.join(sample.s1) + ' : ' + ' '.join(sample.s2)
                label = sample.label

                # original data has duplicate entries with different similarity labels
                if key not in glob.data_integrity_table:
                    words.update(sample.s1)
                    words.update(sample.s2)
                    glob.data_integrity_table[key] = label
                    data.append(sample)
                if 'val' in corpus:
                    if key not in val_integrity_table:
                        words.update(sample.s1)
                        words.update(sample.s2)
                        val_integrity_table[key] = label
                        data.append(sample)

            # record cutoffs
            cutoffs.append((prev_cutoff, len(data)))
            prev_cutoff = len(data)

        if glob.overfit_data_size:
            data = data[0:glob.overfit_data_size]

        if glob.run_data_integrity_tests:
            self.tests.test_vanillaDataKeys(data=data,
                                            gold_dict=glob.data_integrity_table,
                                            proc_name='vanilla data keys ' + str(corpus_names))
            self.tests.test_vanillaDataLabels(data=data,
                                              gold_dict=glob.data_integrity_table,
                                              proc_name='vanilla data labels ' + str(corpus_names))


        # define parameters for creating dataset
        tokenize = lambda x: x.split(' ')
        sort_key = lambda x: sum([len(x.s1), len(x.s2)]) / 2.0

        # define fields
        text_field = torchtext.data.Field(tokenize=tokenize)
        label_field = torchtext.data.Field(sequential=False,
                                           use_vocab=False,
                                           dtype=torch.float32,
                                           is_target=True)

        fields = {'s1' : text_field,
                  's2' : text_field,
                  'label': label_field}

        # make dataset (entries contain words, not indices or vectors)
        data = torchtext.data.Dataset(examples=data,
                                      fields=fields)

        if glob.run_data_integrity_tests:
            self.tests.test_ttextDatasetKeys(data=data,
                                             gold_dict=glob.data_integrity_table,
                                             proc_name='full ttext.Dataset keys: ' + str(corpus_names))
            self.tests.test_ttextDatasetLabels(data=data,
                                               gold_dict=glob.data_integrity_table,
                                               proc_name='full ttext.Dataset labels: ' + str(corpus_names))

        # build vocabulary from data
        text_field.build_vocab(data)
        text_field.vocab.load_vectors(vectors=self.vectors,
                                      dtype=torch.float32)

        if glob.overfit_data_size is None:
            assert len(text_field.vocab.stoi) - 2 == len(words)

        # batch every corpus
        batch_dict = dict()
        for corpus_idx in range(len(corpus_names)):
            if glob.overfit_data_size:
                this_data = data
            else:
                this_data = torchtext.data.Dataset(examples=data[ cutoffs[corpus_idx][0] : cutoffs[corpus_idx][1] ], fields=fields)


            # pre-train
            if 'pre' in corpus_names[corpus_idx]:
                batches = torchtext.data.BucketIterator(dataset=this_data,
                                                  batch_size=glob.pre_train_batch_size,
                                                  sort=False,
                                                  #sort_key=sort_key,
                                                  device=glob.device)

                batch_dict['pre_train'] = batches

                # unit test
                if glob.run_data_integrity_tests:

                    corpus_name = corpus_names[corpus_idx]
                    # ttext dataset
                    self.tests.test_ttextDatasetKeys(data=this_data,
                                                     gold_dict=glob.data_integrity_table,
                                                     proc_name='pre-train ttext.Dataset keys: {0}'.format(corpus_name))
                    self.tests.test_ttextDatasetLabels(data=this_data,
                                                       gold_dict=glob.data_integrity_table,
                                                       proc_name='pre-train ttext.Dataset labels: {0}'.format(corpus_name))

                    # batches
                    self.tests.test_batchKeys(batch= batches,
                                              vocab = text_field.vocab,
                                              gold_dict= glob.data_integrity_table,
                                              proc_name='pre-train batch keys: {0}'.format(corpus_name))
                    self.tests.test_batchLabels(batch= batches,
                                                vocab = text_field.vocab,
                                                gold_dict= glob.data_integrity_table,
                                                proc_name='pre-train batch labels: {0}'.format(corpus_name))

            # train
            elif 'train' in corpus_names[corpus_idx]:
                batches = torchtext.data.BucketIterator(dataset=this_data,
                                                  batch_size=glob.train_batch_size if not for_reg_model else glob.test_batch_size,
                                                  sort=False,
                                                        # sort_key=sort_key,
                                                        device=glob.device)
                batch_dict['train'] = batches

                # unit test
                if glob.run_data_integrity_tests:

                    corpus_name = corpus_names[corpus_idx]
                    # ttext dataset
                    self.tests.test_ttextDatasetKeys(data=this_data,
                                                     gold_dict=glob.data_integrity_table,
                                                     proc_name='train ttext.Dataset keys: {0}'.format(corpus_name))
                    self.tests.test_ttextDatasetLabels(data=this_data,
                                                       gold_dict=glob.data_integrity_table,
                                                       proc_name='train ttext.Dataset labels: {0}'.format(corpus_name))
                    # batches
                    self.tests.test_batchKeys(batch= batches,
                                              vocab = text_field.vocab,
                                              gold_dict= glob.data_integrity_table,
                                              proc_name='train batch keys: {0}'.format(corpus_name))
                    self.tests.test_batchLabels(batch= batches,
                                                vocab = text_field.vocab,
                                                gold_dict= glob.data_integrity_table,
                                                proc_name='train batch labels: {0}'.format(corpus_name))

            # test
            elif 'test' in corpus_names[corpus_idx]:
                batches = torchtext.data.BucketIterator(dataset=this_data,
                                                        batch_size=glob.test_batch_size,
                                                  sort=False,
                                                        # sort_key=sort_key,
                                                        device=glob.device)
                batch_dict['test'] = batches

                # unit test
                if glob.run_data_integrity_tests:

                    corpus_name = corpus_names[corpus_idx]
                    # ttext dataset
                    self.tests.test_ttextDatasetKeys(data=this_data,
                                                     gold_dict=glob.data_integrity_table,
                                                     proc_name='test ttext.Dataset keys: {0}'.format(corpus_name))
                    self.tests.test_ttextDatasetLabels(data=this_data,
                                                       gold_dict=glob.data_integrity_table,
                                                       proc_name='test ttext.Dataset labels: {0}'.format(corpus_name))
                    # batches
                    self.tests.test_batchKeys(batch= batches,
                                              vocab = text_field.vocab,
                                              gold_dict= glob.data_integrity_table,
                                              proc_name='test batch keys: {0}'.format(corpus_name))
                    self.tests.test_batchLabels(batch= batches,
                                                vocab = text_field.vocab,
                                                gold_dict= glob.data_integrity_table,
                                                proc_name='test batch labels: {0}'.format(corpus_name))

            # val
            elif 'val' in corpus_names[corpus_idx]:
                batches = torchtext.data.BucketIterator(dataset=this_data,
                                                        batch_size=glob.test_batch_size,
                                                  sort=False,
                                                        # sort_key=sort_key,
                                                        device=glob.device)
                batch_dict['val'] = batches

                # unit test
                if glob.run_data_integrity_tests:

                    corpus_name = corpus_names[corpus_idx]
                    # ttext dataset
                    self.tests.test_ttextDatasetKeys(data=this_data,
                                                     gold_dict=glob.data_integrity_table,
                                                     proc_name='val ttext.Dataset keys: {0}'.format(corpus_name))
                    self.tests.test_ttextDatasetLabels(data=this_data,
                                                       gold_dict=glob.data_integrity_table,
                                                       proc_name='val ttext.Dataset labels: {0}'.format(corpus_name))
                    # batches
                    self.tests.test_batchKeys(batch= batches,
                                              vocab = text_field.vocab,
                                              gold_dict= glob.data_integrity_table,
                                              proc_name='val batch keys: {0}'.format(corpus_name))
                    self.tests.test_batchLabels(batch= batches,
                                                vocab = text_field.vocab,
                                                gold_dict= glob.data_integrity_table,
                                                proc_name='val batch labels: {0}'.format(corpus_name))

        if glob.run_data_integrity_tests:
            print('\n--------------------')
            print('--------------------> data integrity tests passed')
            print('--------------------\n')

        return (text_field.vocab, batch_dict)


# probably don't need this, leaving here for now
# self.s1 = [w for w in sample[0].split(' ')]
# try:
#     self.s1[-1] = self.s1[-1] if self.s1[-1][-1] != '.' else self.s1[-1][:-1]
# except IndexError:
#     pass

# self.s2 = [w for w in sample[1].split(' ')]
# try:
#     self.s2[-1] = self.s2[-1] if self.s2[-1][-1] != '.' else self.s2[-1][:-1]
# except IndexError:
#     pass