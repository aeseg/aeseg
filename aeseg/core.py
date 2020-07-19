from collections.abc import Iterable

import dcase_util as dcu
import sed_eval


# TODO change eb to event base somethin
def eb_evaluator(y_true, y_pred,
              t_collar: float = 0.200,
              percentage_of_length: float = 0.2
              ) -> sed_eval.sound_event.EventBasedMetrics:
    # TODO In the sed_eval library, EventBasedMetrics have more attribute.
    """
    Return an evaluator function depending on the type of the data provided.
    Different type of data are possible to use. Sed_eval event based metrics.

    # Exemple
    # A list of string
    estimated_event_list = [
        ["file_1.wav\t1.00\t2.00\dog"],
        ["file_2.wav\t1.56\t1.92\cat"]
    ]

    # A string
    estimated_event_list = "file_1.wav\t1.00\t2.00\dog\n" + \
        "file_2.wav\t1.56\t1.92\cat"

    Args:
        y_true: The ground truth
        y_pred: The prediction (results of the encode function)
        t_collar (float): Time collar used when evaluating validity of the onset
        and offset, in seconds. Default value 0.2
        percentage_of_length (float): Second condition, percentage of the
        length within which the estimated offset has to be in order to be
        consider valid estimation. Default value 0.2

    Returns:
        EventBasedMetrics: **event_based_metric**
    """

    # Convert the data into dcase_util.containers.MetaDataContainer
    _y_true = convert_to_mdc(y_true)
    _y_pred = convert_to_mdc(y_pred)

    # return _y_true.unique_files, _y_pred.unique_files
    return event_based_evaluation(_y_true, _y_pred,
                                  t_collar, percentage_of_length)

    # print(_y_true.unique_event_labels)
    # event_based_metric = sed_eval.sound_event.EventBasedMetrics(
    #     event_label_list=_y_true.unique_event_labels,
    #     t_collar=t_collar,
    #     percentage_of_length=percentage_of_length,
    # )

    # return event_based_metric.evaluate(
    #     _y_true, _y_pred
    # )

def event_based_evaluation(reference_event_list, estimated_event_list,
                           t_collar: float, percentage_of_length: float):
    """Calculate sed_eval event based metric for challenge

    Args:
        reference_event_list (MetaDataContainer, list of referenced events):
        estimated_event_list (MetaDataContainer, list of estimated events):

    Returns:
        EventBasedMetrics: **event_based_metric**
    """

    files = {}
    for event in reference_event_list:
        files[event['filename']] = event['filename']

    evaluated_files = sorted(list(files.keys()))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
    )

    for file in evaluated_files:
        reference_event_list_for_current_file = []
        # events = []
        
        for event in reference_event_list:
            if event['filename'] == file:
                reference_event_list_for_current_file.append(event)
                # events.append(event.event_label)

        estimated_event_list_for_current_file = []
        for event in estimated_event_list:
            if event['filename'] == file:
                estimated_event_list_for_current_file.append(event)

        # Replace
        # replace every None by ""
        def none2empty(root):
            for d in root:
                for k in d:
                    if d["event_label"] is None:
                        d["event_label"] = "Speech"

        # none2empty(reference_event_list_for_current_file)
        # none2empty(estimated_event_list_for_current_file)

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    return event_based_metric


def sb_evaluator(y_true, y_pred,
                 time_resolution) -> sed_eval.sound_event.SegmentBasedMetrics:

    # TODO add doc
    # convert the data into dcase_util.containers.MetadataContainer
    _y_true = convert_to_mdc(y_true)
    _y_pred = convert_to_mdc(y_pred)

    return segment_based_evaluation(_y_true, _y_pred, time_resolution)

def segment_based_evaluation(reference_event_list, estimated_event_list,
                             time_resolution):
    # TODO add documentation
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        time_resolution=time_resolution
    )

    for filename in reference_event_list.unique_files:
        reference_event_list_for_current_file = reference_event_list.filter(
            filename=filename
        )

        estimated_event_list_for_current_file = estimated_event_list.filter(
            filename=filename
        )

        segment_based_metrics.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    return segment_based_metrics

# ==============================================================================
#
#       CONVERTION FUNCTIONS
#
# ==============================================================================
def __detect_separator(exemple: str) -> str:
    """
    Automatically detect the separator use into a string and return it

    Args:
        A String exemple

    Returns:
        separator character
    """
    known_sep = [",", ";", ":", "\t"]

    for sep in known_sep:
        if len(exemple.split(sep)) > 1:
            return sep

    return "\t"


def convert_to_mdc(event_list) -> dcu.containers.MetaDataContainer:
    """
    Since the user can provide the reference and estimates event list in
    different format, We must convert them into MetaDataContainer.

    Args:
        event_list: The event list into one of the possible format

    Returns:
        MetaDataContainer
    """

    if isinstance(event_list, dcu.containers.MetaDataContainer):
        return event_list

    # A string
    elif isinstance(event_list, str):
        return string_to_mdc(event_list)

    # list of string
    elif isinstance(event_list, Iterable):
        if isinstance(event_list[0], str):
            return list_string_to_mdc(event_list)

    else:
        raise ValueError("This format %s can't be used. " % type(event_list))


def string_to_mdc(event_strings: str) -> dcu.containers.MetaDataContainer:
    """
    If the data is under the form of a long string with several line (\n).
    The information contain in each line must be separated using one of this
    separator : ",", ";", "\t". It will be automatically detected.

    Args:
         event_strings (str): The string to convert into a MetaDataContainer


    Returns:
        MetaDataContainer
    """
    list_json = []

    # Automatically find the separator
    sep = __detect_separator(event_strings.split("\n")[0])

    for line in event_strings.split("\n"):

        # TODO automatic separator
        info = line.split("\t")

        if len(info) != 4:
            continue

        info_dict = dict()
        info_dict["file"] = info[0]
        if info[1] != "": info_dict["event_onset"] = float(info[1])
        if info[2] != "": info_dict["event_offset"] = float(info[2])
        if info[3] != "": info_dict["event_label"] = info[3]

        list_json.append(info_dict)

    return dcu.containers.MetaDataContainer(list_json)


def list_string_to_mdc(event_list: list) -> dcu.containers.MetaDataContainer:
    """
    If the data is under the form of a list of strings. The information contain
    in each line must be separated using one of this separator : ",", ";",
    "\t". It will be automatically detected.

    Args:
         event_list (str): The string to convert into a MetaDataContainer


    Returns:
        MetaDataContainer
    """
    list_json = []

    # Automatically find the separator
    sep = __detect_separator(event_list[0])

    for line in event_list:
        info = line.split("\t")

        # TODO automatic separator
        info = line.split("\t")

        if len(info) != 4:
            continue

        info_dict = dict()
        info_dict["file"] = info[0]
        if info[1] != "": info_dict["event_onset"] = float(info[1])
        if info[2] != "": info_dict["event_offset"] = float(info[2])
        if info[3] != "": info_dict["event_label"] = info[3]

        list_json.append(info_dict)

    return dcu.containers.MetaDataContainer(list_json)


if __name__=='__main__':
    import numpy as np
    from aeseg.Encoder import Encoder

    # load baseline data
    strong_prediction_path = "/home/lcances/sync/Documents_sync/Projet" \
                        "/Threshold_optimization/data/baseline_strong_prediction.npy"
    strong_prediction = np.load(strong_prediction_path)

    # classes
    class_correspondance = {"Alarm_bell_ringing": 0, "Speech": 1, "Dog": 2,
                            "Cat": 3, "Vacuum_cleaner": 4,
                            "Dishes": 5, "Frying": 6,
                            "Electric_shaver_toothbrush": 7, "Blender": 8,
                            "Running_water": 9}
    class_list = list(class_correspondance.keys())

    encoder = Encoder(class_list, 200, 10, 200)

    import time
    def test(name):
        start = time.time()
        segments = encoder.encode(strong_prediction, method=name)
        end = time.time() - start


    test("threshold")
    test("hysteresis")
    test("derivative")
    test("mean_threshold")
    test("median_threshold")
    test("global_mean_threshold")
    test("global_median_threshold")
