import sys
import numpy as np
from scipy.sparse import coo_matrix

class Data(object):

    def __init__(self, path, verbose=True):
        decpars = {}
        i = 0
        n = {}
        with open(path, 'r') as file:
            for line in file:
                decpar = self.parse_decpar(line)
                for ID in decpar:
                    if ID not in n:
                        n[ID] = 1
                    else:
                        n[ID] += 1
                    if ID not in decpars:
                        decpars[ID] = decpar[ID]
                    else:
                        for condition in decpar[ID]:
                            if condition not in decpars[ID]:
                                decpars[ID][condition] = decpar[ID][condition]
                            else:
                                for consequent in decpar[ID][condition]:
                                    if consequent not in decpars[ID][condition]:
                                        decpars[ID][condition][consequent] = float(decpar[ID][condition][consequent])
                if verbose and i > 0 and i%100000 == 0:
                    sys.stderr.write('%s lines processed\n' %i)
                i += 1

        self.dp = decpars

        for ID in self.dp:
            m = Model(self.dp[ID])
            setattr(self, ID, m)

    def parse_decpar(self, dp_line):
        tokens = dp_line.strip().split()
        assert len(tokens) == 6
        model_id = tokens[0]
        condition = tuple(tokens[1].split(','))
        consequent = tokens[3]
        count = float(tokens[5])

        decpar = {model_id: {condition: {consequent: count}}}

        return decpar




class Model(object):
    def __init__(self, dp):
        n = 0
        condition2ix = {}
        ix2condition = []
        consequent2ix = {}
        ix2consequent = []
        for condition in dp:
            for sub_condition in condition:
                if sub_condition not in condition2ix:
                    condition2ix[sub_condition] = len(ix2condition)
                    ix2condition.append(sub_condition)
            for consequent in dp[condition]:
                if consequent not in consequent2ix:
                    consequent2ix[consequent] = len(ix2consequent)
                    ix2consequent.append(consequent)
                n += 1

        self.dp = dp
        self.n = n
        self.condition2ix = condition2ix
        self.ix2condition = ix2condition
        self.n_condition = len(ix2condition)
        self.consequent2ix = consequent2ix
        self.ix2consequent = ix2consequent
        self.n_consequent = len(ix2consequent)

    def to_matrices(self):
        X = [None, [[],[]]]
        y = np.zeros(self.n, dtype='int32')
        w = np.zeros(self.n, dtype='float32')

        i = 0
        for condition in self.dp:
            condition_ix = []
            for sub_condition in condition:
                condition_ix.append(self.condition2ix[sub_condition])
            for consequent in self.dp[condition]:
                for ix in condition_ix:
                    X[1][0].append(i)
                    X[1][1].append(ix)
                y[i] = self.consequent2ix[consequent]
                w[i] = self.dp[condition][consequent]
                i += 1

        X[0] = np.ones((len(X[1][0]),), dtype='uint8')

        assert len(X[0]) == len(X[1][0]) == len(X[1][1])

        X[1][0] = np.array(X[1][0], dtype='int64')
        X[1][1] = np.array(X[1][1], dtype='int64')
        X[1] = tuple(X[1])
        X = tuple(X)

        X = coo_matrix(X, (self.n, self.n_condition))
        X = X.tocsr()

        return X, y, w
    

