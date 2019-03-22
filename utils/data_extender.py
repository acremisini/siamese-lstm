import pickle
import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from collections import defaultdict
import sys
import pkg_resources
# Can't install Pattern on FIU GPU
if 'Pattern' in [p.project_name for p in pkg_resources.working_set]:
    from pattern.text.en import conjugate, PARTICIPLE, PAST, PRESENT, PROGRESSIVE, INFINITIVE
import os
from utils.globals import Globals as glb

class STSExtender():

    def __init__(self,selectivity_rate):
        #percentage of words that have to match on worndet domain
        self.rate=selectivity_rate
        self.nlp = spacy.load('en')
        self.nlp.add_pipe(WordnetAnnotator(self.nlp.lang), after='tagger')

    #expand a sentence
    def expand_sentence(self,s,):
        sentence = self.nlp(s)
        domain_to_count = dict()
        for token in sentence:
            if token.ent_type == 0 and any(substring in token.tag_ for substring in ['VB','RB','JJ']):
                domains = set(token._.wordnet.wordnet_domains())
                for d in domains:
                    if d not in domain_to_count:
                        domain_to_count[d] = 0
                    domain_to_count[d] += 1
        domains = []
        for d,c in domain_to_count.items():
            #less is less selective, larger dataset
            if c > len(sentence.text.split(' ')) * self.rate:
                domains.append(d)


        enriched_sentence = []
        # For each token in the sentence
        morph_dict={'part':PARTICIPLE,
                    'pres':PRESENT,
                    'past':PAST,
                    'prog':PROGRESSIVE,
                    'inf':INFINITIVE}
        for token in sentence:
            if token.lemma_ != 'be' and any(substring in token.tag_ for substring in ['V','RB','JJ']):
                # We get those synsets within the desired domains
                synsets = token._.wordnet.wordnet_synsets_for_domain(domains)
                morphs = self.nlp.vocab.morphology.tag_map[token.tag_]
                if synsets:
                    lemmas_for_synset = []
                    new_words = set()
                    for s in synsets:
                        lemmas_for_synset.extend(s.lemma_names())
                        for l in lemmas_for_synset:
                            if '_' in l:
                                continue
                            if l == token.lemma_:
                                new_words.add(token.text_with_ws)
                            #conjugate
                            # elif morphs:
                            #     if 'VerbForm' in morphs and 'fin' not in morphs['VerbForm']:
                            #         conj = conjugate(verb=l, tag=token.tag_, tense=morph_dict[morphs['VerbForm']])
                            #     else:
                            #         conj = conjugate(verb=l, tag=token.tag_)
                            #     w = conj if conj is not None else ''
                            #     if w == '':
                            #         continue
                            #     new_words.add(w + ' ')
                            else:
                                new_words.add(l + ' ')
                    enriched_sentence.append(list(new_words))
                else:
                    enriched_sentence.append([token.text_with_ws])
            else:
                enriched_sentence.append([token.text_with_ws])
        return enriched_sentence

    #generate all possible combinations of en enriched sentence
    def unroll(self, enriched_sentence):
        idx = 0
        for i in range(len(enriched_sentence)):
            for j in range(len(enriched_sentence[i])):
                idx += 1
                enriched_sentence[i][j] = str(idx) + 'delim_' + enriched_sentence[i][j]

        nodes = dict()
        nodes['ROOT'] = None

        for i in range(len(enriched_sentence)):
            for j in range(len(enriched_sentence[i])):
                if i == 0:
                    nodes[enriched_sentence[i][j]] = ['ROOT']
                else:
                    nodes[enriched_sentence[i][j]] = []
                    for k in range(len(enriched_sentence[i - 1])):
                        nodes[enriched_sentence[i][j]].append(enriched_sentence[i - 1][k])

        tree = defaultdict(list)
        for node, parent in nodes.items():
            if parent == None:
                tree[parent].append(node)
            else:
                for child in parent:
                    tree[child].append(node)
        # Remove "parent" of the root node

        root = tree.pop(None)[0]
        paths = []

        def traverse(root, tree, path=[]):
            if root != 'ROOT':
                path.append(root)
            if len(tree[root]) == 0:
                p = ''.join([''.join(w.split('delim_')[1:]) for w in path])
                paths.append(p)
                path.pop()
                return p
            else:
                for child in tree[root]:
                    traverse(child, tree)
                if len(path) > 0:
                    path.pop()
            return paths

        def print_tree(root, tree, prefix=''):
            print(prefix + str(root))
            for child in tree[root]:
                print_tree(child, tree, prefix + '  ')
        paths = traverse(root, tree)

        s = 1
        for word_list in enriched_sentence:
            s = s * len(word_list)
        if s != len(paths):
            print('expected:', s, 'got:', len(paths))
            sys.exit()
        return paths

    def extend_data(self,corpus_name):
        train = pickle.load(open(os.path.join(glb.data_dir,corpus_name), 'rb'), encoding='utf-8')
        aug_train = []
        i = 1
        for row in train:
            x1 = row[0]
            x2 = row[1]
            target = str(row[2])
            x1 = self.unroll(self.expand_sentence(x1))
            x2 = self.unroll(self.expand_sentence(x2))
            for s1 in x1:
                for s2 in x2:
                    r = [s1, s2, target,row[3],row[4]]
                    aug_train.append(r)
            if i % 100 == 0:
                print('augmented ' + str(i/len(train)) + '% of ' + corpus_name)
            i += 1
        outfile_name = '{0}_aug_{1}.p'.format(str(corpus_name),str(self.rate))

        with open(os.path.join(glb.data_dir,outfile_name),'wb') as file:
            pickle.dump(aug_train, file)
