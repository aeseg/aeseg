import numpy as np
import itertools

from Encoder import Encoder
from sed_tools import evaluator

from multiprocessing import Pool

from collections.abc import Iterable
import tqdm


def evaluate(combination: tuple, keys: list, method: str, encoder: Encoder, y_true: np.array,
             y_pred: np.array, filenames: list) -> dict:
    """Compute the segments using the given parameters and then compute all the
    metrics using the sed_eval toolbox.

    Args:
        combination (tuple): The parameters for the encoder
        keys (list):
        method (str):
        nb_digit (int):
        encoder (Encoder):
        y_true (np.array):
        y_pred (np.array):
        filenames (list):
    """
    # Transform parameters from tuple to dictionary
    # If only one parameter --> transform into tuple before
    # if not isinstance(combination, list):
    #    combination = [combination]

    combination = dict(
        zip( keys, combination )
    )

    # Compute segment and transform into a csv string
    segments = encoder.encode(
        y_pred,
        method=method,
        **combination
    )
    to_evaluate = encoder.parse(segments, filenames)

    # evaluate using sed_eval
    return evaluator(y_true, to_evaluate).results()


class Optimizer:
    """The Dichotomic optimizer will search for the combination of parameters
    that will lead to the best score. For each parameters that must be
    optimized, a tuple is provided representing the range in between the
    algorithm must search.

    When the search is complete, the process is repeated in between to two
    best combination of parameters in order to yields more accurate results.
    """

    def __init__(self, param: dict, encoder: Encoder, step: int,
                 nb_recurse: int, nb_process: int = 4):
        """
        Args:
            param (dict): The parameters that should be optimized.
            encoder (Encoder): The encoder object that will be used to
            step (int): In how many step the range should be divided.
            nb_recurse (int): The number of recursion.
            nb_digit (int): The precision of the parameters.
            nb_process (int): The number of thread used for the optimization
                process.
        """

        self.param = param
        self.encoder = encoder
        self.step = step
        self.nb_recurse = nb_recurse

        self.original_param = param.copy()
        self.keys = list(param.keys())

        # Dataset parameters
        self._y_true = None
        self._y_pred = None
        self._filenames = None

        # Synchronization variables
        self.nb_process = nb_process
        self.process_pool = None
        # self.process_manager = Manager()
        self.history = dict()
        self.fitted = False

    def param_to_range(self, param: dict) -> dict:
        """Giving a tuple representing the minimum and the maximum value of a
        parameters, will generate, uniformly a list of value to test.

        Args:
            param (dict): The parameters to optimize
        """
        outputs = dict()

        for key in param:
            # is tuple --> range
            if isinstance(param[key], tuple):
                outputs[key] = np.linspace(
                    param[key][0], param[key][1],
                    self.step
                )

            # if list --> actual liste
            elif isinstance(param[key], list):
                outputs[key] = param[key]

            # if not iterable --> fix value, no changing
            elif not isinstance(param[key], Iterable):
                outputs[key] = param[key]

            # if str --> fix value, no changing
            elif isinstance(param[key], str):
                outputs[key] = [param[key]]

        return outputs

    def nb_iteration(self):
        extend_parameters = self.param_to_range(self.param)

        nb_iteration = 1
        for key in extend_parameters:
            nb_iteration *= len(extend_parameters[key])

        return nb_iteration

    def two_best(self, source: dict, keys: list) -> dict:
        """Return the new range (tuples) for each parameters based on the two
        best results

        Args:
            source (dict): The combination of parameters, the keys are a tuple
            keys (list): The list of combination
        """
        # Transform the dictionary into a list of tuple where the first element
        # is the combination of parameters and the second the score
        tuples = list(zip(source.keys(), source.values()))

        # Sort the combination by the score
        tuples.sort(key=lambda elem: elem[1])

        # Create a dictionary for each key and a tuple representing the new
        # research space for each parameters
        tmp_param = dict(zip(keys, list(zip(tuples[-1][0], tuples[-3][0]))))

        # In some case, like with str param or unique value param,
        # The tuple created is no suitable, so we take only the first element to go back to a unique parameter
        for key in tmp_param:
            # if suppose to be str --> str
            if isinstance(self.param[key], str):
                tmp_param[key] = tmp_param[key][0]

            if isinstance(self.param[key], list):
                tmp_param[key] = self.param[key]

        return tmp_param

    def fit(self, y_true: np.array, y_pred: np.array, filenames: list, verbose: int = 1, method: str = "threshold"):
        """Initialize the thread pool and perform the optimization.*

        Args:
            y_true (np.array): The ground truth, Must be in one of the format
            y_pred (np.array): The prediction that must be segmented by the
            filenames (list): A list of filename in the same order than y_pred
            verbose (int): 0 --> No verbose at all, 1 --> tqdm terminal progress, 2 --> tqdm notebook progress bar
        """
        self._y_true = y_true
        self._y_pred = y_pred
        self._filenames = filenames

        self.process_pool = Pool(processes=self.nb_process)

    @property
    def results(self) -> dict:
        if not self.fitted:
            raise RuntimeWarning("No optimization done yet")
        return self.history

    # Necessary to make the class (serialisable)
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict["process_pool"]
        return self_dict

    def __setstate__(self, state):
        """
        Args:
            state:
        """
        self.__dict__.update(state)


class ThresholdOptimizer(Optimizer):
    def __init__(self, param: dict, encoder: Encoder, step: int,
                 nb_recurse: int, nb_process: int = 4):
        """
        Args:
            param (dict):
            encoder (Encoder):
            step (int):
            nb_recurse (int):
            nb_digit (int):
            nb_process (int):
        """
        super().__init__(param, encoder, step, nb_recurse, nb_process)

    def fit(self, y_true: np.array, y_pred: np.array, filenames: list,
            monitor: str = "f_measure", verbose: int = 1, method: str = "threshold"):
        """
        Args:
            y_true (np.array):
            y_pred (np.array):
            filenames (list):
            monitor (str):
            verbose (int):
        """
        super().fit(y_true, y_pred, filenames)
        self.history = dict()

        _param = self.param.copy()

        if verbose == 1:
            progress = tqdm.tqdm(total = self.nb_iteration() * self.nb_recurse)
        if verbose == 2:
            progress = tqdm.tqdm_notebook(total = self.nb_iteration() * self.nb_recurse)

        for _ in range(self.nb_recurse):

            # Create all the combination
            search_space = self.param_to_range(_param)
            all_combination = itertools.product(*search_space.values())

            # Add all combination to the thread pool
            works = []
            for combination in all_combination:
                works.append((
                    combination,
                    self.process_pool.apply_async(evaluate, kwds={
                        "combination": combination,
                        "keys": self.keys,
                        "method": method,
                        "encoder": self.encoder,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "filenames": filenames
                    })
                ))

            # wait for the pool to finish working on the current recursion
            results = []
            for combination, res in works:
                results.append((combination, res.get()))

                if verbose != 0:
                    progress.update()

            # Save all results in the history. The combination will be the key
            for combination, res in results:
                self.history[combination] = res

            # The best parameters combination is evaluate using the monitored
            # metric
            results_monitor = dict()
            for combination, r in results:
                results_monitor[combination] = \
                    r["class_wise_average"]["f_measure"][monitor]

            # Find the two best results among this recursion
            two_best = self.two_best(results_monitor, list(_param.keys()))

            # Set the new parameters range for the next recursion
            _param = two_best

        self.fitted = True
