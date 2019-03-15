"""
Optimizer_copy.py
version 1.0
author Leo Cances leo.cances@irit.fr

Optimization process for the different encoding methods used.
The dichotonomous search will, given boudaries for each parameters, start by roughtly try a small amount of combination.
The two best will be used as new boundaries and the process will be repeated again.

The genetical algorithm will perform succesive random change and will compute every time a "score". If one random
combination give better performance, then the set of parameters is kept and the process is done again.
"""
import numpy as np
import tqdm
import time
import dcase_util as dcu
import itertools
from Encoder import Encoder
from evaluation_measures import event_based_evaluation  # evaluation system from DCASE

from multiprocessing import Manager
from multiprocessing.pool import ThreadPool



# ======================================================================================================================
#
#   Genetical algorithm based optimization
#
# ======================================================================================================================
class OptimizeParameterError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)



class Gen_Optimizer:
    def __init__(self, param: dict, reference_path: str):
        self.param = param
        self.param_keys = list(param.keys())
        self._current_combination = dict()
        self.reference_path = reference_path

        self.__check_param()

    def __check_param(self):
        for key in self.param.keys():
            p = self.param[key]

            if not isinstance(p, (tuple, list)):
                raise OptimizeParameterError("Parameters must be tuple of length two or list of length at lest 1")
            else:
                if isinstance(p, tuple):
                    if len(p) != 2:
                        raise OptimizeParameterError("Parameters must be tuple of length two or list of length at lest 1")
                if isinstance(p, list):
                    if len(p) < 2:
                        raise OptimizeParameterError("Parameters must be tuple of length two or list of length at lest 1")

    def random_parameters(self):
        for k in self.param.keys():
            p = self.param[k]

            if isinstance(p, tuple):
                self._current_combination[k] = self.__random_tuple_parameters(p)

            if isinstance(p, list):
                self._current_combination[k] = self.__ranndom_list_parameters(p)

        return self._current_combination

    def __ranndom_list_parameters(self, l: list):
        return np.random.choice(l)

    def __random_tuple_parameters(self, t: tuple):
        return np.random.uniform(t[0], t[1])

    def slight_change(self, current_param):
        # choose one parameters randomly
        picked_param = np.random.choice(self.param_keys)
        new_param = dict(current_param)

        new_value = new_param[picked_param]
        if isinstance(self.param[picked_param], tuple):
            new_value = self.__slight_change_tuple(new_param[picked_param], self.param[picked_param])

        if isinstance(self.param[picked_param], list):
            new_value = self.__slight_change_list(new_param[picked_param], self.param[picked_param])

        new_param[picked_param] = new_value

        return new_param

    def __slight_change_tuple(self, current_value, r):
        print("change in tuple")
        range_size = max(r) - min(r)

        # get a random delta between 0 % and 1 % of the range size
        random_delta = np.random.uniform(min(r), max(r))

        new_value = random_delta
        if new_value < min(r): new_value = min(r)
        if new_value > max(r): new_value = max(r)

        return new_value


    def __slight_change_list(self, current_value, l):
        next = np.random.randint(0, 1)

        if next:
            return current_value[l.index(current_value) + 1]
        else:
            return current_value[l.index(current_value) - 1]


    def optimize(self, nb_iteration: int, nb_recurse: int):
        # 1 - initialize random parameters
        # 2 - evaluate current score
        # 3 - Choose one parameters randomly
        # 4 - slightly move it in the authorized range (tuple) or pick next / previous (list)
        # 5 - Evaluate
        # 6 - Compare
        # 7 - Update
        raise NotImplementedError

    def __to_listDict(self, to_evaluate):
        list_dict = []
        for line in to_evaluate.split("\n")[:-1]:
            info = line.split("\t")
            list_dict.append({
                "filename": str(info[0]),
                "onset": float(info[1]),
                "offset": float(info[2]),
                "event_label": str(info[3])
            })

        return list_dict

    def evaluate(self, to_evaluate):
        # Critical memory access, synchronization mechanisme required
        # self.semaphore.acquire()

        # with open(file_path, "w") as f:
        #    f.write("filename\tonset\toffset\tevent_label\n")
        #    f.write(to_evaluate)

        # perso_event_list = dcu.containers.MetaDataContainer()
        perso_event_list = dcu.containers.MetaDataContainer(self.__to_listDict(to_evaluate))
        # perso_event_list.load(filename=file_path)
        # self.semaphore.release()

        ref_event_list = dcu.containers.MetaDataContainer()
        ref_event_list.load(filename=self.reference_path)

        event_based_metric = event_based_evaluation(ref_event_list, perso_event_list)

        return event_based_metric


class Threshold_Optimizer(Gen_Optimizer):
    def __init__(self, param: dict, reference_path: str, time_prediction, name_list):
        super().__init__(param, reference_path)
        self.time_prediction = time_prediction
        self.name_list = name_list
        self.encoder = Encoder()

    def get_eval(self, param: dict):
        ths = [param["high"]] * 10
        segments = self.encoder.encode(self.time_prediction, method="threshold",
                                       padding="same",
                                       smooth="smoothMovingAvg",
                                       thresholds=ths,
                                       window_len=19
                                       )

        toEvaluate = self.encoder.parse(segments, self.name_list)
        return toEvaluate

    def optimize(self, nb_iteration: int, nb_recurse: int):
        _history = []

        _best_param = self.random_parameters()
        _best_score = self.evaluate(self.get_eval(_best_param)).results()

        progress = tqdm.tqdm(total=nb_iteration * nb_recurse)

        for recurse_counter in range(nb_recurse):

            _current_param = dict(_best_param)
            for iteration_counter in range(nb_iteration):
                _current_param = self.slight_change(_best_param)

                _current_score = self.evaluate(self.get_eval(_current_param)).results()

                _current_f1 = _current_score["class_wise_average"]["f_measure"]["f_measure"]
                _best_f1 = _best_score["class_wise_average"]["f_measure"]["f_measure"]

                if _current_f1 >= _best_f1:
                    _best_score = _current_score.copy()
                    _best_param = dict(_current_param)
                    _history.append((_current_f1, _best_param))

                progress.update()

        return _history


# ======================================================================================================================
#
#   evaluator pipeline
#
# ======================================================================================================================
class EvaluateJob(pipe.Job):
    def __init__(self, time_prediction, name_list,
                 name: str = "evaluate",
                 reference_path="/baie/corpus/dcase2018/task4/metadata/test.csv", ):
        super().__init__(name)
        self.time_prediction = time_prediction
        self.name_list = name_list
        self.reference_path = reference_path
        self.encoder = Encoder()

    def set_param(self, param: dict):
        super().set_param(param)

    def to_list_dict(self, to_evaluate):
        list_dict = []
        for line in to_evaluate.split("\n")[:-1]:
            info = line.split("\t")
            list_dict.append({
                "filename": str(info[0]),
                "onset": float(info[1]),
                "offset": float(info[2]),
                "event_label": str(info[3])
            })

        return list_dict

    def strong_evaluate(self, to_evaluate, file_path):
        # Critical memory access, synchronization mechanisme required
        # self.semaphore.acquire()

        # with open(file_path, "w") as f:
        #    f.write("filename\tonset\toffset\tevent_label\n")
        #    f.write(to_evaluate)

        # perso_event_list = dcu.containers.MetaDataContainer()
        perso_event_list = dcu.containers.MetaDataContainer(self.to_list_dict(to_evaluate))
        # perso_event_list.load(filename=file_path)
        # self.semaphore.release()

        ref_event_list = dcu.containers.MetaDataContainer()
        ref_event_list.load(filename=self.reference_path)

        event_based_metric = event_based_evaluation(ref_event_list, perso_event_list)

        return event_based_metric

    def exec(self, unique_param, **kwargs):
        raise NotImplementedError


class Evaluator(pipe.Pipeline):
    def __init__(self, param, step, nb_digit, writer: Writer, nb_process: int = 4,
                 granularity: int = 10):
        super().__init__(writer, "", "", nb_process, None, granularity)
        self.param = param
        self.step = step
        self.nb_digit = nb_digit

        self.keys = list(param.keys())

    def generate_all_range(self, ranges: dict):
        outputs = dict()

        for key in ranges:
            outputs[key] = np.linspace(ranges[key][0], ranges[key][1], self.step)

        return outputs

    def exec(self):
        # Generate the research space
        research_space = self.generate_all_range(self.param)
        all_combinaison = itertools.product(*research_space.values())

        # add all combinaison possible to the queue
        for combinaison in all_combinaison:
            unique_combinaison = dict(zip(self.keys, map(lambda x: round(x, self.nb_digit), combinaison)))
            self.to_process.put(unique_combinaison)


class Logger(pipe.Writer):
    def __init__(self, tmp_monitor: dict, total_score: dict, monitor, total: int = None, timeout: int = 1, ):
        super().__init__(total, timeout)

        self.monitor = monitor
        self.total_score = total_score
        self.tmp_monitor = tmp_monitor

        # for monitoring progress
        self.total = 0
        self.progress_value = 0
        self.timer_start = None
        self.timing = []
        self.timing_mean_size = 10

    def set_total(self, value: int):
        print("iteration for each recurse = ", value)
        self.total = value

    def calc_timing(self):
        if self.timer_start is None:
            self.timer_start = time.time()
            return 0
        else:
            self.timing.append(time.time() - self.timer_start)
            self.timing = self.timing[-self.timing_mean_size:]
            self.timer_start = time.time()
            return 1 / (sum(self.timing) / self.timing_mean_size)

    def update_progress(self):
        self.progress_value += 1
        speed = self.calc_timing()
        estimated = (self.total - self.progress_value) / (speed + 0.000001)
        print("\r%.2f %% %.2f it.s ~%.2f" % ((self.progress_value / self.total * 100), speed, estimated), end="")
        # use display(f) if you encounter performance issues

    def exec(self, metadata, **kwargs):
        job_name = metadata[0]
        results = metadata[1][0]
        unique_param = metadata[1][1]

        score_key = tuple(unique_param.values())
        self.total_score[score_key] = results
        self.tmp_monitor[score_key] = results["class_wise_average"]["f_measure"][self.monitor]

        self.update_progress()


class ThresholdJob(EvaluateJob):

    def __init__(self, time_prediction, name_list, name: str = "evaluate",
             reference_path="/baie/corpus/DCASE2018/task4/metadata/test.csv"):
        super().__init__(time_prediction, name_list, name, reference_path)

    def exec(self, unique_param, **kwargs):
        ths = [unique_param["high"]] * 10
        segments = self.encoder.encode(self.time_prediction, method="threshold",
                                       padding="same",
                                       smooth="smoothMovingAvg",
                                       thresholds=ths,
                                       window_len=19
                                       )

        toEvaluate = self.encoder.parse(segments, self.name_list)
        evaluator = self.strong_evaluate(toEvaluate, "/notebook_tmp/eval.csv")

        results = evaluator.results()

        return results, unique_param


class HysteresisJob(EvaluateJob):

    def __init__(self, time_prediction, name_list, name: str = "evaluate",
                 reference_path="/baie/corpus/DCASE2018/task4/metadata/test.csv"):
        super().__init__(time_prediction, name_list, name, reference_path)

    def exec(self, unique_param, **kwargs):
        segments = self.encoder.encode(self.time_prediction, method="hysteresis",
                                       padding="same",
                                       smooth="smoothMovingAvg",
                                       **unique_param,
                                       )

        toEvaluate = self.encoder.parse(segments, self.name_list)
        evaluator = self.strong_evaluate(toEvaluate, "/notebook_tmp/eval.csv")

        results = evaluator.results()

        return results, unique_param


class DerivativeJob(EvaluateJob):

    def __init__(self, time_prediction, name_list, name: str = "evaluate",
             reference_path="/baie/corpus/DCASE2018/task4/metadata/test.csv"):
        super().__init__(time_prediction, name_list, name, reference_path)

    def exec(self, unique_param, **kwargs):
        segments = self.encoder.encode(self.time_prediction, method="derivative",
                                       padding="same",
                                       smooth="smoothMovingAvg",
                                       **unique_param,
                                       )

        toEvaluate = self.encoder.parse(segments, self.name_list)
        evaluator = self.strong_evaluate(toEvaluate, "/notebook_tmp/eval.csv")

        results = evaluator.results()

        return results, unique_param


def eval(time_prediction, name_list,
         param, job_to_perform,
         reference_path="/baie/corpus/DCASE2018/task4/metadata/test.csv",
         nb_digit=3, step=5, recurse=2, nb_process=4):
    process_manager = Manager()
    _param = param
    _keys = list(_param.keys())
    shared_tmp_score = process_manager.dict()
    total_score = process_manager.dict()
    best_param = _param

    def two_best(source: dict, keys) -> dict:
        """Return the new range for each parameters based on the two best results. """
        tuples = list(zip(source.keys(), source.values()))
        tuples.sort(key=lambda elem: elem[1])
        # Create a dictionary for each key and a tuple representing the new research space for each parameters
        return dict(zip(keys, list(zip(tuples[-1][0], tuples[-3][0]))))

    for _ in range(recurse):
        writer = Logger(shared_tmp_score, total_score, "f_measure", timeout=10)

        pipeline = Evaluator(
            granularity=40,
            nb_process=nb_process,
            writer=writer,
            param=_param,
            step=step,
            nb_digit=nb_digit
        )

        # Prepare job to do
        job = job_to_perform(time_prediction, name_list, reference_path=reference_path)
        pipeline.append_job(job)

        # Execute pipeline
        writer.set_total(step ** len(_keys))
        pipeline.run()

        # Get two best results and loop over with new range
        # total_score = {**total_score, **shared_tmp_score}
        _param = two_best(shared_tmp_score, _keys)
        best_param = _param

    return total_score, best_param

def get_cross_evaluate_info(cross_eval):
    # thresholds_cross_eval = thresholds_cross_eval[()]
    f1 = []
    er = []
    dr = []
    ir = []
    for r in cross_eval:
        f1.append(r["class_wise_average"]["f_measure"]["f_measure"])
        er.append(r["class_wise_average"]["error_rate"]["error_rate"])
        dr.append(r["class_wise_average"]["error_rate"]["deletion_rate"])
        ir.append(r["class_wise_average"]["error_rate"]["insertion_rate"])

    return f1, er, dr, ir