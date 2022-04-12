import pytrec_eval


def prepare_dict_for_evaluation(res):
    """
        Converts the dictonary to be passed to PyTrec eval

        :param res: a dictonary with words
        :return: same in passable format to the PyTrec eval
    """
    res_for_eval = {}
    for k in res.keys():
        dict_ = {}
        sorted_gt = [(k, v) for k, v in sorted(res.get(k).items(), key=lambda item: item[1])]
        for idx, l in enumerate(sorted_gt[:10]):
            dict_[l[0]] = 1
        res_for_eval[k] = dict_
    return res_for_eval


class PyTrecEvaluator:
    def __init__(self, res_golden):
        self.evaluator = pytrec_eval.RelevanceEvaluator(res_golden, {'ndcg'})

    def evaluate(self, result_evaluation: dict):
        """
        :param result_evaluation: A dictornary with prediction results to compare against golden.
        :return: average performance.
        """
        result = self.evaluator.evaluate(result_evaluation)

        metrics = {}
        for measure in sorted(list(result[list(result.keys())[0]].keys())):
              metrics[f'{measure} average'] = \
                  pytrec_eval.compute_aggregated_measure(
                      measure, [query_measures[measure] for query_measures in result.values()])
        return metrics
