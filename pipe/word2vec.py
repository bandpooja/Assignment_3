from gensim.models import Word2Vec
import numpy as np
import os
import os.path as osp
import warnings

from utils.evaluation import PyTrecEvaluator, prepare_dict_for_evaluation


class Word2VecSimilarWord:
    def __init__(self, text):
        self.text = text
        self.model = Word2Vec()

    def tune_window_n_size(self, windows: list, vector_sizes: list, val_list_of_words: list,
                           evaluator: PyTrecEvaluator):
        """
            performs hyper-parameter tuining.
            Similar to GridSearcCV for scikit-learn

            :param windows: list of window sizes
            :param vector_sizes: list os vector sizes
            :param val_list_of_words: list of words to evalute the final model on
            :param evaluator: evaluator to evaluate the model performance
            :return: best hyper-parameters and the score for all the grid points
        """
        best_result = -1
        best_window = None
        best_vector_size = None

        scores = np.zeros((len(windows), len(vector_sizes)))
        # region HYPER-PARAMETER TUNING
        # a valid parameter of the one's returned by evaluator
        selection_parameter = 'ndcg average'
        for i, window in enumerate(windows):
            for j, vector_size in enumerate(vector_sizes):
                # fit the model
                self.fit(window, vector_size)
                # make predictions using this model
                predictions = self.predict(val_list_of_words)
                predictions = prepare_dict_for_evaluation(predictions)
                # make the model NULL to remove the previous training
                self.model = Word2Vec()

                # get metrics from the evaluator
                m_s = evaluator.evaluate(predictions)[selection_parameter]
                if m_s > best_result:
                    best_result = m_s
                    best_window = window
                    best_vector_size = vector_size
                scores[i][j] = m_s
        # endregion
        return best_window, best_vector_size, scores

    def fit(self, window: int, vector_size: int):
        """
            :param window: context window of Word2Vec model
            :param vector_size: vector size of Word2Vec model
        """
        # vector_size argument is called size in older version
        try:
            self.model = Word2Vec(sentences=self.text, window=window, min_count=1, vector_size=vector_size,
                                  epochs=1000)
        except Warning('Using an older version of Word2Vec'):
            self.model = Word2Vec(sentences=self.text, window=window, min_count=1, size=vector_size,
                                  epochs=1000)

    def save_model(self, model_loc: str):
        """
            :param model_loc: location to save the model in
        """
        if os.path.exists(model_loc):
            self.model.save(osp.join(model_loc, "word2vec.model"))
        else:
            try:
                os.makedirs(model_loc, exist_ok=True)
                self.model.save(osp.join(model_loc, "word2vec.model"))
            except TypeError:
                print('Cant save the model !')

    def load_model(self, model_loc: str):
        """
            :param model_loc: location to load the model from (dir path)
        """
        if osp.exists(model_loc):
            self.model.save(osp.join(model_loc, "word2vec.model"))
        else:
            raise Exception('Cant load the model !')

    def predict(self, list_of_words: list):
        """
            :param list_of_words: list of words
            :return: prediction for a list of words in a nested-dictonary format
        """
        try:
            # for gensim 4.0.0 +
            vocab = list(self.model.wv.index_to_key)
        except Warning:
            vocab = list(self.model.wv.vocab.keys())
        word2vec_similar = {}
        for i in list_of_words:
            if i in vocab:
                sims = self.model.wv.most_similar(i, topn=10)
                dict_ = {}
                for elem in sims:
                    dict_[elem[0]] = elem[1]
                word2vec_similar[i] = dict_
            else:
                # OOV
                pass
        return word2vec_similar
