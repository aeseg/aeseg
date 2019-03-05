import numpy as np
import tqdm
import matplotlib.pyplot as plt

import dcase_util as dcu
from evaluation_measures import event_based_evaluation  # evaluation system from DCASE

from Encoder_copy import Encoder


class_correspondance = {
    "Alarm_bell_ringing": 0,
    "Speech": 1,
    "Dog": 2,
    "Cat": 3,
    "Vacuum_cleaner": 4,
    "Dishes": 5,
    "Frying": 6,
    "Electric_shaver_toothbrush": 7,
    "Blender": 8,
    "Running_water": 9
}


def load_thomas_dict(path: str):
    """The dictionary contain only the curve of the predicted class, therefore
    some adaptation is needed.

    the dict key will be used to create the filename list required for the
    segment parsing :param path: The path to the prediction

    Args:
        path (str):
    """
    data = np.load(path)
    tmp = dict()
    
    # Fill the missing curve with zero
    for filename in data:
        new_data = np.zeros((431, 10))
        
        for cls in data[filename]:
            new_data[:, class_correspondance[cls]] = data[filename][cls]
            
        tmp[filename] = new_data.copy()
    
    # Create a list of filename in the same order than the dict
    filename_list = ["%s.wav" % k for k in tmp.keys()]
    
    # Create a list
    output_data = np.array(list(tmp.values()))
    
    return output_data, filename_list



def strong_evaluate(to_evaluate,
                    reference_path="/baie/corpus/DCASE2018/task4/metadata/test.csv",
                    file_path=".ref.csv"):
    """Perform the evaluation on the strong prediction.

    Args:
        to_evaluate: The csv string that must be evaluate
        reference_path: The path to the metadata reference
        file_path: The path to where the temporary file will be saved (tmpfs
            directory is appreciated)
    """

    with open(file_path, "w") as f:
        f.write("filename\tonset\toffset\tevent_label\n")
        f.write(to_evaluate)

    perso_event_list = dcu.containers.MetaDataContainer()
    perso_event_list.load(filename=file_path)

    ref_event_list = dcu.containers.MetaDataContainer()
    ref_event_list.load(filename=reference_path)
    #"/baie/corpus/DCASE2018/task4/metadata/test.csv"

    event_based_metric = event_based_evaluation(ref_event_list, perso_event_list)
    return event_based_metric


# Find best hysteresis threshold parameters
def eval_hysteresis(time_prediction, name_list, low = (0, 0.5), high = (-0.5, 1), 
                   nb_digit=3, step=5, recurse=2,
                   monitor: str = "f_measure"):
    """Find best hysteresis threshold parameters.

    Args:
        time_prediction: The time prediction that will be convert into segments
            and evaluate
        name_list: The list of file name (must be in the same order than
            time_prediction)
        low: The search boudaries for the "low" parameter (low -> bottom
            threshold)
        high: the search boudaries for the "high" parameter (high -> top
            threshold)
        nb_digit: The threshold max precision
        step: The number of value that will be test in between the boudaries
            (for each parameters)
        recurse: The number of time the search will be done (each time in
            between the two best boundaries of the previous execution)
        monitor (str): The metrics to focus on for comparing the results of each
            parameters combination.
    """
    
    def two_best(source: dict) -> tuple:
        """Find the two combination of parameters that yields the best scores"""
        tuples = list(zip(source, source.values()))
        tuples.sort(key=lambda elem: elem[1])
        
        return tuples[-1][0], tuples[-2][0]
        
    total_scores = {}
    _low, _high = low, high
    encoder = Encoder()
    
    progress = tqdm.tqdm(total=step*recurse)
    
    for recurse_number in range(recurse):
        # Create all the combination possible in between the boudaries of the parameters
        research_space = zip(
            np.linspace(_low[0], _high[0], step),
            np.linspace(_low[1], _high[1], step)
        )
                
        tmp_monitor = {}
        
        #l -> low ; h -> high
        for l, h in research_space:
            l = round(l, nb_digit)
            h = round(h, nb_digit)
            
            # Encode the time prediction into segments
            segments = encoder.encode(
                time_prediction,
                method="hysteresis",
                low=l, high=h,
                smooth="smoothMovingAvg")
            
            # Parse into a csv string
            toEvaluate = encoder.parse(segments, name_list)
            
            # Evaluate using the evaluation function from DCASE2018 task 4
            evaluator = strong_evaluate(toEvaluate)
            
            results = evaluator.results()
            total_scores[(l, h)] = results
            tmp_monitor[(l, h)] = results["class_wise_average"]["f_measure"][monitor]
            
            progress.update()
            
        # Get the two best combination of parameters and search again in between those parameters
        _high, _low = two_best(tmp_monitor)
    
    return total_scores




if __name__=='__main__':
    max_abs_cos_01_9_path = "/baie/travail/thomas/dcase2018/thomas_GLU_models_alpha/ten_models_max_abs_cos_0.1/9/dico_prob_curves_for_predicted_classes.pkl"
    data, filenames = load_thomas_dict(max_abs_cos_01_9_path)
    print(data.shape, data.mean(), data.std())

    hysteresis_evaluation = eval_hysteresis(data, name_list=filenames,
        low=(0.01, 0.05), high=(0.05, 0.1), # Ici je le force un peu à ce concentrer sur les valeur base parce ce a tendence à tomber dans des minimum locaux
        nb_digit=4,
        step = 10,
        recurse=4)

    print(hysteresis_evaluation)

    # Pour récupérer toutes les f_measures
    ths = list(hysteresis_evaluation.keys())
    print(ths)
    
    f_measures = [hysteresis_evaluation[k]["class_wise_average"]["f_measure"]["f_measure"] for k in ths]

    # Pour les afficher
    x = np.linspace(0, len(f_measures), len(f_measures))

    plt.figure(0, figsize=(15, 5))
    plt.plot(f_measures, marker="x", color="C0", linewidth=0)
    plt.xticks(range(0, len(ths)), ths, rotation=80)
    plt.show()