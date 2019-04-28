import unittest
import torch
from utils.globals import Globals as glob
from unit_tests.testUtils import testUtils
import urllib.request

class TestEmbeds(unittest.TestCase):

    # test if each index gets same vector from word2vec and ttext cache
    def test_embedLookup(self, embeds, words_idx,vocab):
        print('embed test')
        for i in range(len(words_idx)):
            # from lstm
            embed1 = embeds[i]
            word = vocab.itos[words_idx[i]]
            if word not in ['<pad>']:
                urlFail = False
                embed2 = vocab.vectors[vocab.stoi[word]].to(glob.device)
                try:
                    contents = urllib.request.urlopen("http://localhost:9000/?vec={0}".format(word if word != '&' else '%26')).read()
                except:
                    print('bad url: ' + "http://localhost:9000/?vec={0}".format(word if word != '&' else '%26'))
                    urlFail = True
                if not urlFail:
                    l = []
                    for b in str(contents).split(','):
                        l.append(float(b.replace('\'', '').replace('b', '')))
                    embed3 = torch.tensor(l).to(glob.device)
                    self.addTypeEqualityFunc(type(embed1), self.assert3TensorEquals(embed1,embed2,embed3,word))

    def test_batchKeys(self, batch, vocab, gold_dict, proc_name):
        print('Running {0} tests'.format(proc_name))
        batch_s1 = torch.transpose(batch.s1,0,1)
        batch_s2 = torch.transpose(batch.s2, 0, 1)
        for i in range(batch_s1.size(0)):
            t1 = batch_s1[i]
            t2 = batch_s2[i]
            s1 = testUtils.sent_from_index_vec(t1, vocab)
            s2 = testUtils.sent_from_index_vec(t2, vocab)
            key = s1 + ' : ' + s2
            self.assertTrue(key in gold_dict, msg='{0} not in dict'.format(key))
        print('..... OK')

    def test_batchLabels(self, batch, vocab, gold_dict, proc_name):
        print('Running {0} tests'.format(proc_name))
        batch_s1 = torch.transpose(batch.s1,0,1)
        batch_s2 = torch.transpose(batch.s2, 0, 1)
        for i in range(batch_s1.size(0)):
            t1 = batch_s1[i]
            t2 = batch_s2[i]
            s1 = testUtils.sent_from_index_vec(t1, vocab)
            s2 = testUtils.sent_from_index_vec(t2, vocab)
            key = s1 + ' : ' + s2
            label = batch.label[i]

            # item
            self.addTypeEqualityFunc(type(gold_dict[key]), self.assertTensorItemEquals(gold=gold_dict[key],
                                                                                       label=label,
                                                                                       labels=batch.label,
                                                                                       key=key))
            # self.assertEqual(gold_dict[key], label)

            # type
            self.addTypeEqualityFunc(type(gold_dict[key]), self.assertTensorTypeEquals(gold=gold_dict[key],
                                                                                       label=label))
            # self.assertEqual(gold_dict[key], label)

            # device
            self.addTypeEqualityFunc(type(gold_dict[key]), self.assertTensorDeviceEquals(gold=gold_dict[key],
                                                                                         label=label))
            # self.assertEqual(gold_dict[key], label)
        print('..... OK')

    def assertTensorItemEquals(self,gold,label,labels,key):
        if gold.item() != label.item():
            print('ERROR: ')
            print(labels)
            print(key)
            print('item: gold {0} != ttext {1}'.format(gold, label))
            self.fail(msg='item: gold {0} != ttext {1}'.format(gold, label))

    def assertTensorTypeEquals(self,gold,label):
        if gold.device != label.device:
            self.fail(msg='device: gold {0} != ttext {1}'.format(gold, label))

    def assertTensorDeviceEquals(self,gold,label):
        if gold.dtype != label.dtype:
            self.fail(msg='dtype: gold {0} != ttext {1}'.format(gold, label))

    def assertTensorEquals(self,t1,t2):
        # torch.all(tens_a: eq(tens_b))
        if not torch.all(torch.eq(t1,t2)):
            self.fail(msg='{0} != {1}'.format(t1,t2))

    def assert3TensorEquals(self,t1,t2,t3,word):
        eq = True
        eq = eq and torch.all(torch.eq(t1,t2))
        eq = eq and torch.all(torch.eq(t2,t3))
        if not eq:
            self.fail(msg='{0}\nlstm:\n{0}\n ttext:\n{1}\n w2v\n{2}\n'.format(word,t1,t2,t3))