import nltk
from nltk.corpus import brown
from nltk.tokenize.moses import MosesDetokenizer
import pandas as pd
from tqdm import tqdm


class DataLoader:
    def __init__(self):
        self.corpus_allowed = ['romance', 'news']

    def fetch_data(self, corpus: str = ''):
        """
            :param corpus: corpus to get the data for
            :return: List of tokenizes sentences (Used for Word2Vec), List of sentences (Used for Tfidf)
        """

        if corpus in self.corpus_allowed:
            if corpus == 'romance':
                tokenized_sentences = brown.sents(categories='romance')

                mdetok = MosesDetokenizer()
                sentences = [
                    mdetok.detokenize(' '.join(sent).replace('``', '"').replace("''", '"').replace('`', "'").split(),
                                      return_str=True) for sent in brown.sents(categories='romance')
                ]
                return tokenized_sentences, sentences
            elif corpus == 'news':
                tokenized_sentences = brown.sents(categories='news')

                mdetok = MosesDetokenizer()
                sentences = [
                    mdetok.detokenize(' '.join(sent).replace('``', '"').replace("''", '"').replace('`', "'").split(),
                                      return_str=True) for sent in brown.sents(categories='news')
                ]
                return tokenized_sentences, sentences
        else:
            raise Exception('Corpus not in allowed corpuses')


# Setting up simlex values
class SimLexLoader:
    def __init__(self, file_loc: str = "data/SimLex-999.txt"):
        self.file_loc = file_loc

    def load(self):
        """
        :return: df with simlex-word and synonym pairings
        """
        result1 = [x.split('\t')[0] for x in open(self.file_loc).readlines()]
        result2 = [x.split('\t')[1] for x in open(self.file_loc).readlines()]
        result3 = [x.split('\t')[3] for x in open(self.file_loc).readlines()]


        result1.remove('word1')
        result2.remove('word2')
        result3.remove('SimLex999')

        self.df = pd.DataFrame(list(zip(result1, result2, result3)),
                       columns =['word1', 'word2', 'SimLex999'])
        return self.df

    @staticmethod
    def return_one_level_similar_words(temp, i_sim, result_dict):
        i_sim_old = i_sim
        for j in list(i_sim.keys()):
            if j in temp:
                j_dict = result_dict.get(j)
                for k in list(j_dict.keys()):
                    if k not in list(i_sim.keys()):
                        i_sim.update({k: j_dict[k]})
        return i_sim, i_sim_old

    def get_top_words(self):
        result_dict = {}

        for word1, word2, simi in zip(self.df['word1'], self.df['word2'], self.df['SimLex999']):
            if word1 not in result_dict.keys():
                result_dict[word1] = {word2: simi}

            else:
                if word2 not in result_dict.get(word1):
                    result_dict[word1].update({word2: simi})

        temp = result_dict.keys()

        for i in tqdm(temp, desc='Combining outputs: '):
            i_sim = result_dict.get(i)
            i_sim_old = {}
            while len(i_sim) < 10 and list(i_sim.keys()) != list(i_sim_old.keys()):
                i_sim, i_sim_old = self.return_one_level_similar_words(temp, i_sim, result_dict)
            result_dict[i] = i_sim

        final_dict = dict()

        for key, val in result_dict.items():
            val = {k: float(v) for k, v in val.items()}
            sorted_val = {k: v for k, v in sorted(val.items(), key=lambda item: item[1], reverse=True)}
            final_dict[key] = sorted_val

        return final_dict
