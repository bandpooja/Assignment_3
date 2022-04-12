import nltk

nltk.download('brown')
nltk.download('perluniprops')
import os
import os.path as osp
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm

from preprocess.data_loader import DataLoader, SimLexLoader
from pipe.tfidf import TfidfTextContextSimilarWord
from pipe.word2vec import Word2VecSimilarWord
from utils.evaluation import PyTrecEvaluator, prepare_dict_for_evaluation
from utils.plotter import hyper_parameter_CM, line_plots


class Assignment3:
    def __init__(self, model_loc: str, corpus: str, file_loc: str):
        self.model_loc = model_loc
        self.corpus = corpus
        self.file_loc = file_loc

    def load_simlex_data(self):
        simlex_loader = SimLexLoader(file_loc=self.file_loc)
        df = simlex_loader.load()
        print('#' * 8 + ' Simlex - loaded dataframe' + '#' * 8)
        print(df.head())
        self.result_golden = simlex_loader.get_top_words()
        self.test_words = list(self.result_golden.keys())

    def initiate_evaluator(self):
        # initiate the evaluator
        self.result_golden_for_evaluator = prepare_dict_for_evaluation(self.result_golden)
        self.evaluator = PyTrecEvaluator(res_golden=self.result_golden_for_evaluator)

    def load_data(self):
        loader = DataLoader()
        self.tokenized_sentences, self.sentences = loader.fetch_data(self.corpus)

    def tfidf(self):
        # TF-IDF Model
        tfidf_model = TfidfTextContextSimilarWord(self.sentences)

        # grid-search tf-idf
        min_dfs = [1e-4, 1e-3, 1e-2]
        best_min_df, scores, vocab_sizes = tfidf_model.tune_min_df(min_df=min_dfs, val_list_of_words=self.test_words,
                                                                   evaluator=self.evaluator)

        # plot to show model-hyper-parameter-search-visualization
        line_plots(arrays=[scores, vocab_sizes], labels=['(ndcg average) Score', 'vocab size'], xticks=min_dfs,
                   xlabel='minimum document frequency', ylabel='',
                   title=f'(ndcg average) TF-IDF Score for different hyperparameters on {corpus}',
                   img_loc=osp.join(self.model_loc, f'hyper-parameter-search-TFIDF-{corpus}.png'))

        # fit the final model
        tfidf_model.fit(best_min_df)

        # save it and load the best model
        tfidf_model.save_model(model_loc=self.model_loc)
        tfidf_model.load_model(model_loc=self.model_loc)

        # test the final model
        pred_dict = tfidf_model.predict(list_of_words=self.test_words)
        pred_dict_for_evaluator = prepare_dict_for_evaluation(pred_dict)
        ms = self.evaluator.evaluate(pred_dict_for_evaluator)
        return ms

    def word2vec(self):
        # Word2Vec Model
        word2vec_model = Word2VecSimilarWord(self.tokenized_sentences)

        windows = [1, 2, 5, 10]
        vector_sizes = [10, 50, 100, 300]
        best_window_size, best_vector_size, scores = word2vec_model.tune_window_n_size(windows=windows,
                                                                                       vector_sizes=vector_sizes,
                                                                                       val_list_of_words=self.test_words,
                                                                                       evaluator=self.evaluator)
        # plot to show model-hyper-parameter-search-visualization
        hyper_parameter_CM(array=scores, index=windows, columns=vector_sizes, ylabel='Context windows', xlabel='Vector sizes',
                           title=f'(ndcg average) Word2Vec Score for different hyperparameters on {corpus}',
                           img_loc=osp.join(self.model_loc, f'hyper-parameter-search-word2vec-{corpus}.png'))

        word2vec_model.fit(window=best_window_size, vector_size=best_vector_size)

        word2vec_model.save_model(model_loc=self.model_loc)
        word2vec_model.load_model(model_loc=self.model_loc)

        pred_dict = word2vec_model.predict(list_of_words=self.test_words)
        pred_dict_for_evaluator = prepare_dict_for_evaluation(pred_dict)
        ms = self.evaluator.evaluate(pred_dict_for_evaluator)
        return ms


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    model_loc = 'result'
    corpora = ['news', 'romance']
    # load the simlex data

    results = {}
    for corpus in tqdm(corpora, desc='Fitting for corpus'):
        results[corpus] = {}
        model_loc_ = osp.join(model_loc, corpus)
        os.makedirs(model_loc_, exist_ok=True)

        exp = Assignment3(model_loc_, corpus=corpus, file_loc=r'result\SimLex-999.txt')
        exp.load_simlex_data()
        exp.initiate_evaluator()
        exp.load_data()
        ms = exp.tfidf()
        results[corpus]['tfidf'] = ms
        ms = exp.word2vec()
        results[corpus]['word2vec'] = ms

    # final bar-plot
    print(results)

    x_list = [1, 2, 3, 4]
    y_list = []
    for i in corpora:
        y_list.append(results[i]['tfidf']['ndcg average'])
        y_list.append(results[i]['word2vec']['ndcg average'])

    plt.bar(x_list, y_list)
    labels = ['TF-IDF news', 'Word2Vec news', 'TF-IDF romance', 'Word2Vec romance']
    plt.xticks(x_list, labels, rotation=20)
    plt.title('Tf-Idf and Word2vec on news and romance corpus')
    plt.xlabel('Models')
    plt.ylabel('mean nDCG score')
    plt.grid()
    plt.tight_layout()
    plt.savefig(osp.join(model_loc, f'result.png'))
    plt.close()
