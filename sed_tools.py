import sed_eval
import dcase_util as dcu


def evaluator(reference_event_list, estimated_event_list,
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
        reference_event_list: The ground truth
        estimated_event_list: The prediction (results of the encode function)
        t_collar (float): Time collar used when evaluating validity of the onset
        and offset, in seconds. Default value 0.2
        percentage_of_length (float): Second condition, percentage of the
        length within which the estimated offset has to be in order to be
        consider valid estimation. Default value 0.2

    Returns:
        EventBasedMetrics: **event_based_metric**
    """

    # Convert the data into dcase_util.containers.MetaDataContainer
    estimated_event_list = convert_to_mdc(estimated_event_list)
    reference_event_list = convert_to_mdc(reference_event_list)

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=reference_event_list.unique_event_labels,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
    )

    return event_based_metric.evaluate(
        reference_event_list, estimated_event_list
    )


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
        if len(exemple.split(sep)) > 0:
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
    estimated = event_list

    if isinstance(estimated, dcu.containers.MetaDataContainer):
        estimated = estimated

    # list of string
    elif isinstance(estimated, list):
        if isinstance(estimated[0], str):
            estimated = list_string_to_mdc(event_list)

    # A string
    elif isinstance(estimated, str):
        estimated = string_to_mdc(event_list)

    else:
        raise ValueError("This format %s can't be used. " % type(event_list))

    return estimated


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
        info = line.split(sep)

        list_json.append({
            "file": info[0],
            "event_onset": info[1],
            "event_offset": info[2],
            "event_label": info[3]
        })

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
        info = line.split(sep)

        list_json.append({
            "file": info[0],
            "event_onset": info[1],
            "event_offset": info[2],
            "event_label": info[3]
        })

    return dcu.containers.MetaDataContainer(list_json)


if __name__=='__main__':
    import numpy as np
    from Encoder import Encoder

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
        print("testing", name, end="")
        segments = encoder.encode(strong_prediction, method=name)
        end = time.time() - start

        print("done in %.2f seconds" % end)

    test("threshold")
    test("hysteresis")
    test("derivative")
    test("mean_threshold")
    test("median_threshold")
    test("global_mean_threshold")
    test("global_median_threshold")
