import numpy as np


class Encoder:
    """
    In a sound event detection task, the output of the prediction model is
    often a temporal prediction. Different segmentation algorithm exist in order
    to translate this curves into a list of segment.
    """
    def __init__(self, classes: list, temporal_precision: int, clip_length: int,
                 minimal_segment_step: int):
        """
        Args:
            classes (list):
            temporal_precision (int):
            clip_length (int):
            minimal_segment_step (int):
        """
        self.classes = classes
        self.temporal_precision = temporal_precision
        self.clip_length = clip_length
        self.minimal_segment_step = minimal_segment_step

    def __encode_using_threshold(self, temporal_prediction: np.array,
                                 **kwargs) -> list:
        """
        Args:
            temporal_prediction (np.array):
            **kwargs:
        """
        output = []

        # Recover kwargs arguments
        thresholds = kwargs.get('thresholds', None)

        # Binarize if requested using the given thresholds
        if thresholds is None:
            thresholds = [0.5] * len(self.classes)

        bin_prediction = temporal_prediction.copy()
        bin_prediction[bin_prediction > thresholds] = 1
        bin_prediction[bin_prediction <= thresholds] = 0

        # Merging "hole" that are smaller than 200 ms
        step_length = self.clip_length / temporal_prediction.shape[1] * 1000
        max_hole_size = int(self.minimal_segment_step / step_length)

        for clip in bin_prediction:
            labeled = dict()

            cls = 0
            for bin_prediction_per_class in clip.T:
                # convert the binarized list into a list of tuple representing
                # the element and it's number of occurrence. The order is
                # conserved and the total sum should be equal to 10s

                # first pass --> Fill the holes
                for i in range(len(bin_prediction_per_class) - max_hole_size):
                    window = bin_prediction_per_class[i: i+max_hole_size]

                    if window[0] == window[-1] == 1:
                        window[:] = [window[0]] * max_hole_size

                # second pass --> split into segments
                converted = []
                cpt = 0
                nb_segment = 0
                previous_elt = None
                for element in bin_prediction_per_class:
                    if previous_elt is None:
                        previous_elt = element
                        cpt += 1
                        nb_segment = 1
                        continue

                    if element == previous_elt:
                        cpt += 1

                    else:
                        converted.append((previous_elt, cpt))
                        previous_elt = element
                        nb_segment += 1
                        cpt = 1

                # case where the class is detect during the whole clip
                #                 if nbSegment == 1:
                converted.append((previous_elt, cpt))

                labeled[cls] = converted.copy()
                cls += 1

            output.append(labeled)

        return output
