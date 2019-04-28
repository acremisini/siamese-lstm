import unittest
import torch
from unit_tests.testUtils import testUtils

# all tests passed on 04/25/19 @ 16:54

class TestData(unittest.TestCase):

    def test_vanillaDataKeys(self, data, gold_dict, proc_name):
        print('Running {0} tests'.format(proc_name))
        for s in data:
            s1 = ' '.join(s.s1)
            s2 = ' '.join(s.s2)
            key = s1 + ' : ' + s2
            self.assertTrue(key in gold_dict, msg='{0} not in dict'.format(key))
        print('..... OK')

    def test_vanillaDataLabels(self, data, gold_dict, proc_name):
        print('Running {0} tests'.format(proc_name))
        for s in data:
            s1 = ' '.join(s.s1)
            s2 = ' '.join(s.s2)
            label = s.label
            key = s1 + ' : ' + s2
            # print('{0}\n--->{1}'.format(key,label))
            # item
            self.addTypeEqualityFunc(type(gold_dict[key]), self.assertTensorItemEquals(gold=gold_dict[key],
                                                                                       label=label,
                                                                                       key=key))
            self.assertEqual(gold_dict[key], label)

            # type
            self.addTypeEqualityFunc(type(gold_dict[key]), self.assertTensorTypeEquals(gold=gold_dict[key],
                                                                                       label=label))
            self.assertEqual(gold_dict[key], label)

            # device
            self.addTypeEqualityFunc(type(gold_dict[key]), self.assertTensorDeviceEquals(gold=gold_dict[key],
                                                                                         label=label))
            self.assertEqual(gold_dict[key], label)
        print('..... OK')

    def test_ttextDatasetKeys(self, data, gold_dict, proc_name):
        print('Running {0} tests'.format(proc_name))
        for e in data.examples:
            s1 = ' '.join(e.s1)
            s2 = ' '.join(e.s2)
            key = s1 + ' : ' + s2
            self.assertTrue(key in gold_dict, msg='{0} not in dict'.format(key))
        print('..... OK')

    def test_ttextDatasetLabels(self, data, gold_dict, proc_name):
        print('Running {0} tests'.format(proc_name))
        for e in data.examples:
            s1 = ' '.join(e.s1)
            s2 = ' '.join(e.s2)
            key = s1 + ' : ' + s2
            label = e.label

            # item
            self.addTypeEqualityFunc(type(gold_dict[key]), self.assertTensorItemEquals(gold=gold_dict[key],
                                                                                       label=label,
                                                                                       key=key))

            # type
            self.addTypeEqualityFunc(type(gold_dict[key]), self.assertTensorTypeEquals(gold=gold_dict[key],
                                                                                       label=label))

            # device
            self.addTypeEqualityFunc(type(gold_dict[key]), self.assertTensorDeviceEquals(gold=gold_dict[key],
                                                                                         label=label))
        print('..... OK')


    def test_batchKeys(self, batch, vocab, gold_dict, proc_name):
        print('Running {0} tests'.format(proc_name))
        for b in batch:
            t1 = torch.transpose(b.s1, 0, 1)
            t2 = torch.transpose(b.s2, 0, 1)
            for i in range(len(t1)):
                s1 = testUtils.sent_from_index_vec(t1[i], vocab)
                s2 = testUtils.sent_from_index_vec(t2[i], vocab)
                key = s1 + ' : ' + s2
                self.assertTrue(key in gold_dict, msg='{0} not in dict'.format(key))
        print('..... OK')

    def test_batchLabels(self, batch, vocab, gold_dict, proc_name):
        print('Running {0} tests'.format(proc_name))
        for b in batch:
            t1 = torch.transpose(b.s1, 0, 1)
            t2 = torch.transpose(b.s2, 0, 1)
            for i in range(len(t1)):
                s1 = testUtils.sent_from_index_vec(t1[i], vocab)
                s2 = testUtils.sent_from_index_vec(t2[i], vocab)
                key = s1 + ' : ' + s2
                label = b.label[i]

                # item
                self.addTypeEqualityFunc(type(gold_dict[key]), self.assertTensorItemEquals(gold= gold_dict[key],
                                                                                           label=label,
                                                                                           key=key))

                # type
                self.addTypeEqualityFunc(type(gold_dict[key]), self.assertTensorTypeEquals(gold=gold_dict[key],
                                                                                           label=label))

                # device
                self.addTypeEqualityFunc(type(gold_dict[key]), self.assertTensorDeviceEquals(gold=gold_dict[key],
                                                                                           label=label))
        print('..... OK')

    def assertTensorItemEquals(self,gold,label,key):
        if gold.item() != label.item():
            self.fail(msg='\n{0}\nitem: gold {1} != ttext {2}'.format(key,gold, label))

    def assertTensorTypeEquals(self,gold,label):
        if gold.device != label.device:
            self.fail(msg='device: gold {0} != ttext {1}'.format(gold, label))

    def assertTensorDeviceEquals(self,gold,label):
        if gold.dtype != label.dtype:
            self.fail(msg='dtype: gold {0} != ttext {1}'.format(gold, label))