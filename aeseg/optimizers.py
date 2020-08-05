import functools
import itertools
import operator
from collections.abc import Iterable
from multiprocessing import Pool

import numpy as np
import tqdm

from Encoder import Encoder


class Optimizer:
    def __init__(self, step: int, nb_recurse: int, nb_process: int = 4):
        """
        Abstract class for the different optimizer available

        .. seealso::
            - :class:`optimizers.DichotomicOptimizer`
            - :class:`optimizers.GenOptimizer`

        Args:
            param (dict):
                The parameters that should be optimized.
            optimize_fn (Callable):
                The encoder object that will be used to compute the score
            step (int):
                In how many step the range should be divided.
            nb_recurse (int):
                The number of recursion.
            nb_process (int):
                The number of processes used for the optimization.
        """

        self.step = step
        self.nb_recurse = nb_recurse

        # Synchronization variables
        self.nb_process = nb_process
        self.process_pool = None
        self.results = dict()
        self.fitted = False

        # Monitoring
        self.progress = None

    def find(self, monitor: tuple, data: dict):
        """ Find the metric to monitor in the nested dictionary return by the evaluator.
        """
        return functools.reduce(operator.getitem, monitor, data)

    def nb_iteration(self, parameters):
        raise NotImplementedError

    def dict_nan_to_num(self, d: dict):
        """ For every np.nan within the dictionary and the nested dictionary, will convert every nan to 0.0.
        """
        for k in d:
            if isinstance(d[k], dict):
                self.dict_nan_to_num(d[k])
            else:
                d[k] = np.nan_to_num(d[k])

    def fit(self, kwds=dict(), verbose=1):
        """ Perform the optimization of the parameters provided by kwds.
        kwds (dict):
            The dictionnary key represents the parameters of the optimization function
            Each key value must be either a list, a tuple or a single value.
            A tuple will be interpreted as a range of value possible, automatially split into <steps>
            A list will be interpreted as an ensemble fix of value to test
            A single value as a fixed value (no optimization on this parameter
        """
        self.fitted = True
        self.keys = list(kwds.keys())

        # start the pool
        self.process_pool = Pool(self.nb_process)
        
        if verbose == 1:
            self.progress = tqdm.tqdm(total=self.nb_iteration(kwds))
        if verbose == 2:
            self.progress = tqdm.tqdm_notebook(total=self.nb_iteration(kwds))

    @property
    def history(self) -> dict:
        """
        Return the complete history of the optimization process, useful for visualization of the optimization process.
        Work only if an optimization process have been already done.

        Returns:
            a list of dict, see sed_eval documentation for further detail
        """
        if not self.fitted:
            raise RuntimeWarning("No optimization done yet")
        return self.results

    @property
    def best(self) -> tuple:
        """
        Return the combination of parameters that yield the best score using the monitored metric.

        Returns:
            tuple (parameters, score)
        """
        if not self.fitted:
            raise RuntimeWarning("Optimization not perform yet")

        out = []

        for k in self.results.keys():
            value = self.results[k]
            out.append((k, value))

        out = sorted(out, key=lambda x: x[1])

        # pair each value with it parameter
        return dict(zip(self.keys, out[-1][0])), out[-1][1]


class DichotomicOptimizer(Optimizer):
    """ The dichotomous optimizer aims to drastically reduce the search time for the combination of parameters
    that gives the best score.

    For each parameter that needs to be optimized, the user provides a search interval. Using a "step" parameter
    defined at creation, the optimizer will then search for all possible combinations. The best combination is
    selected, and new intervals are calculated. The optimizer then resumes its search in these new intervals.
    The optimization stops when a fixed number of recursions specified at creation is reached.
    """
    def __init__(self, step: int,
                 nb_recurse: int, nb_process: int = 4):
        """
        Args:
            param (dict):
                The parameters that should be optimized.
            optimize_fn (Callable):
                The function used to compute the metric to optimize
            step (int):
                In how many step the range should be divided.
            nb_recurse (int):
                The number of recursion.
            nb_process (int):
                The number of processes used for the optimization.

        :Exemple:
        To use the Dichotomous optimizer on the absolute threshold method, 3 step are needed.
        - Define the list of class that are predicted by the system.
        - Create an encoder with a specified temporal prediction and a minimum segmentatio separation.
        - Create an optimizer that will use this encoder and perform best combination search.

        .. code-block:: python

            class_list = ['Alarm_bell_ringing', 'Speech', 'Dog', 'Cat', 'Vacuum_cleaner', 'Dishes', 'Frying', 'Electric_shaver_toothbrush', 'Blender', 'Running_water']

            # Create the encoder that will be used
            encoder = Encoder(
                classes=class_list,
                temporal_precision = 200,  # ms
                clip_length = 10,          # s
                minimal_segment_step = 200 # ms
            )

            parameter_to_optimize = {
                    "threshold": (0.1, 0.9),
                    "smooth": "smoothMovingAvg",
                    "window_len": (5, 27)
                }

            optimizer = DichotomicOptimizer(
                **parameter_to_optimize,
                method="threshold",
                encoder = encoder,

                step = 6,
                nb_recurse = 10,
                nb_process = 20
            )
        """
        super().__init__(step, nb_recurse, nb_process)

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

    def nb_iteration(self, parameters):
        extend_parameters = self.param_to_range(parameters)

        nb_iteration = 1
        for key in extend_parameters:
            nb_iteration *= len(extend_parameters[key])

        return nb_iteration * self.nb_recurse

    def two_best(self, source: dict, params: dict, nb_recurse: int = 1, mode="max")-> dict:
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

        # best parameters combination
        pick = -1 if mode == "max" else 0
        best_combination = dict(zip(params.keys(), tuples[-1][0]))

        # For each parameters that need it (tuple), build the new search
        # boundaries
        for key in best_combination:
            # only for tuple range
            if isinstance(params[key], tuple):
                best = best_combination[key]
                half = best / (1 + nb_recurse)

                # boundaries
                low = best - half
                high = best + half

                # clip
                low = low if low > params[key][0] else params[key][0]
                high = high if high < params[key][1] else params[key][1]

                best_combination[key] = (low, high)

        # in some case, like with str param or unique value param,
        # the tuple created is no suitable, so we take only the first element
        # to go back to a unique parameter
        for key in best_combination:
            # if suppose to be str --> str
            if isinstance(params[key], str):
                best_combination[key] = best_combination[key]

            if isinstance(params[key], list):
                best_combination[key] = params[key]

        return best_combination

    def fit(self, optimize_fn, kwds=dict(), mode="max", verbose=1):
        super().fit(kwds, verbose)
        _param = kwds.copy()
        _args = kwds.keys()

        for recurse in range(self.nb_recurse):

            # Create all the combination
            search_space = self.param_to_range(_param)
            all_combination = itertools.product(*list(search_space.values()))

            # Add all combination to the thread pool
            works = []
            for combination in all_combination:
                fn_args = dict(zip(_args, combination))
                works.append((
                    combination,
                    self.process_pool.apply_async(optimize_fn, kwds=fn_args)
                ))

            # wait for the pool to finish working on the current recursion
            # Save all results in the history. The combination will be the key
            for combination, res in works:
                self.results[combination] = res.get()

                if verbose != 0:
                    self.progress.update()

            # Find the two best results among this recursion
            two_best = self.two_best(
                self.results,
                _param,
                recurse,
                mode
            )

            # Set the new parameters range for the next recursion
            _param = two_best

        # close the pool
        self.process_pool.close()
        self.process_pool.join()
