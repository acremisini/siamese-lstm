

class testUtils():

    @staticmethod
    def sent_from_index_vec(vec, vocab):
        sent = ' '.join([vocab.itos[vec[i]] for i in range(len(vec)) if vec[i] != 1])
        return sent