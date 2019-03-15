import numpy as np
import tqdm
import itertools

from Encoder import Encoder
from sed_tools import evaluator
import sed_eval

from multiprocessing import Manager
from multiprocessing.pool import ThreadPool


class Optimizer:
    def __init__(self, param: dict, encoder: Encoder, step: int,
                 nb_recurse: int, nb_digit:int, nb_process: int = 4):
        """
        Args:
            param (dict):
            encoder (Encoder):
            step (int):
            nb_recurse (int):
            nb_digit (int):
            nb_process (int):
        """

        self.param = param
        self.encoder = encoder
        self.step = step
        self.nb_recurse = nb_recurse
        self.nb_digit = nb_digit

        self.original_param = param.copy()
        self.keys = list(param.keys())

        # multiprocess parameters
        self.nb_process = nb_process
        self.thread_pool = None

        # Dataset parameters
        self._y_true = None
        self._y_pred = None
        self._filenames = None

    def param_to_range(self, param: dict) -> dict:
        """
        Args:
            param (dict):
        """
        outputs = dict()

        for key in param:
            outputs[key] = np.linspace(
                param[key][0], param[key][1],
                self.step
            )

        return outputs

    def two_best(self, source: dict) -> dict:
        """Return the new range (tuples) for each parameters based on the two
        best results

        Args:
            source (dict):
            keys:
        """
        tuples = list(zip(source.keys(), source.values()))
        tuples.sort(key=lambda elem: elem[1])

        # Create a dictionary for each key and a tuple representing the new
        # research space for each paramters
        return dict(
            zip(
                source.keys, list(
                    zip(
                        tuples[-1][0], tuples[-3][0]
                    )
                )
            )
        )

    def fit(self, y_true: np.array, y_pred: np.array, filenames: list):
        """
        Initialize the thread pool and perform the optimization.*

        Args:
            y_true:
            y_pred: The content of a csv file (
            filenames:

        Returns:

        """
        self._y_true = y_true
        self._y_pred = y_pred
        self._filenames = filenames

        self.thread_pool = ThreadPool(self.nb_process)

    def evaluate(self, combination: tuple):
        """

        Args:
            combination:

        Returns:

        """
        raise NotImplementedError


class ThresholdOptimizer(Optimizer):
    def __init__(self, param: dict, step: int, nb_recurse: int, nb_digit: int,
                 nb_process: int = 4):
        """
        Args:
            param (dict):
            step (int):
            nb_recurse (int):
            nb_digit (int):
            nb_process (int):
        """
        super().__init__(param, step, nb_recurse, nb_digit, nb_process)

    def fit(self, y_true: np.array, y_pred: np.array, filenames: list,
            monitor: str = "f_measure"):
        """

        Args:
            y_true:
            y_pred:
            filenames:

        Returns:

        """
        super().fit()
        _param = self.param.copy()

        for recurse in range(self.nb_recurse):

            # Create all the combination
            search_space = self.param_to_range(_param)
            all_combination = itertools.product(*search_space.values())

            # Add all combination to the thread pool
            works = []
            for combination in all_combination:
                works.append(
                    self.thread_pool.apply_async(self.__evaluate, combination)
                )

            # wait for the pool to finish working on the current recursion
            results = [res.get() for res in works]


            # The best parameters combination is evaluate using the monitored
            # metric
            results_monitor =

            # Find the two best results among this recursion
            two_best =


    def __evaluate(self, combination: tuple ) -> dict:
        """
        Compute the segments using the given parameters and then compute all
        the metrics using the sed_eval toolbox.

        Args:
            combination: The parameters for the encoder

        Returns:

        """
        # Transform parameters from tuple to dictionary
        combination = dict(
            zip(
                self.keys,
                map(
                    lambda x: round(x, self.nb_digit),
                    combination
                )
            )
        )

        # Compute segment and transform into a csv string
        segments = self.encoder.encode(
            self._y_pred,
            method="threshold",
            **combination
        )
        to_evaluate = self.encoder.parse(segments, self._filenames)

        # evaluate using sed_eval
        return evaluator(self._y_true, self._y_pred).results()


