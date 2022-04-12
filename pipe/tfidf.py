import numpy as np
import os
import os.path as osp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

from utils.common_utils import load_sparse_csr, save_sparse_csr
from utils.evaluation import PyTrecEvaluator, prepare_dict_for_evaluation


class TfidfTextContextSimilarWord:
    def __init__(self, text: list):
        self.text = text
        self.X = sparse.csr_matrix(np.array([]))
        self.vocabulary = []

    def tune_min_df(self, min_df: list, val_list_of_words: list, evaluator: PyTrecEvaluator):
        best_result = -1
        best_min_df = None
        scores = np.zeros((len(min_df)))
        vocab_sizes = []
        # region HYPER-PARAMETER TUNING
        # a valid parameter of the one's returned by evaluator
        selection_parameter = 'ndcg average'
        for i, min_df_ in enumerate(min_df):
            # fit the model
            self.fit(min_df_)
            # make predictions using this model
            predictions = self.predict(val_list_of_words)
            predictions = prepare_dict_for_evaluation(predictions)

            vocab_sizes.append(len(self.vocabulary))
            # delete the model to remove the previous training effect
            self.X = sparse.csr_matrix(np.array([]))
            self.vocabulary = []
            # get metrics from the evaluator
            m_s = evaluator.evaluate(predictions)[selection_parameter]
            if m_s > best_result:
                best_result = m_s
                best_min_df = min_df_
            scores[i] = m_s
        # endregion
        return best_min_df, scores, vocab_sizes

    def fit(self, min_df: float):
        vectorizer = TfidfVectorizer(min_df=min_df, stop_words='english', ngram_range=(1, 1))
        X = vectorizer.fit_transform(self.text)

        # converting the doctument matrix to word-context matrix
        self.X = X.T
        self.X = sparse.csr_matrix(self.X)
        self.vocabulary = list(vectorizer.get_feature_names_out())

    def predict(self, list_of_words):
        tfidf_similar = {}
        for i in list_of_words:
            ix = [i_ for i_, w in enumerate(self.vocabulary) if w == i]
            if len(ix) == 1:
                ix_ = ix[0]
                w1 = self.X[ix_]
                sim_scores = [(w, p) for w, p in zip(self.vocabulary,
                                                     list(enumerate(cosine_similarity(w1, self.X)))[0][1])]
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                # the most similar word is going to be the word itself so taking words other than the word itself
                sim_scores = sim_scores[1:11]

                dict_ = {}
                for elem in sim_scores:
                    dict_[elem[0]] = elem[1]
                    tfidf_similar[i] = dict_
            else:
                # OOV
                pass
        return tfidf_similar

    def save_model(self, model_loc: str):
        if os.path.exists(model_loc):
            save_sparse_csr(osp.join(model_loc, 'tfidf_word_context.pkl'), self.X)
            with open(osp.join(model_loc, 'tfidf_vocabulary.txt'), "w") as file:
                file.write(str(self.vocabulary))
        else:
            try:
                os.makedirs(model_loc, exist_ok=True)
                save_sparse_csr(osp.join(model_loc, 'tfidf_word_context.pkl'), self.X)
                with open(osp.join(model_loc, 'tfidf_vocabulary.txt'), "w") as file:
                    file.write(str(self.vocabulary))
            except TypeError:
                print('Cant save the model !')

    def load_model(self, model_loc: str):
        if os.path.exists(model_loc):
            self.X = load_sparse_csr(osp.join(model_loc, 'tfidf_word_context.pkl'))
            with open(osp.join(model_loc, 'tfidf_vocabulary.txt'), "r") as file:
                self.vocabulary = eval(file.readline())
        else:
            raise Exception('Cant load the model !')
