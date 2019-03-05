import numpy as np


class Encoder:
    """In a sound event detection task, the output of the prediction model is
    often a temporal prediction. Different segmentation algorithm exist in order
    to translate this curves into a list of segment.
    """

    def __init__(self, classes: list, temporal_precision: int, clip_length: int,
                 minimal_segment_step: int):
        """
        Args:
            classes (list):
            temporal_precision (int):
            clip_length (int): The length of the audio file (in seconds)
            minimal_segment_step (int):
        """
        self.classes = classes
        self.temporal_precision = temporal_precision
        self.clip_length = clip_length
        self.minimal_segment_step = minimal_segment_step

        # Attribute that are not initialize with the constructor
        self.frame_length = None
        self.nb_frame = None

    def encode(self, temporal_prediction: np.array, method: str = "threshold",
               smooth: str = None, **kwargs) -> str:
        """Perform the localization of the sound event present in the file.

        Using the temporal prediction provided y the last step of the system,
        it will "localize" the sound event inside the file under the form of a
        strongly annotated line. (see DCASE2018 task 4 strong label exemple).
        There is two methods implemented here, one using a simple threshold
        based segmentation and an other using a modulation system based on the
        variance of the prediction over the time.

        # Exemples
        ```YOTsn73eqbfc_10.000_20.000.wav  0.163   0.665   Alarm_bell_ringing```

        Args:
            temporal_prediction (np.array): A 3-dimension numpy array (<nb
                clip>, <nb frame>, <nb class>)
            method (str): The segmentation method to use [threshold | hysteresis
                | derivative | primitive]
            smooth (str): The smoothing method to use [smoothMovingAvg]
            kwargs: See the segmentation method parameters

        Returns:
            Return a list of positive and negative segments with their size. A
            segment is a tuple where the first value that represent the segment
            value (1) for positive, (0) for negative and the second values is
            the width of the segment (number of frame)
        """
        # parameters verification
        _methods = ["threshold", "hysteresis", "derivative", "primitive",
                    "mean_threshold", "median_threshold", "dynamic_threshold",
                    "global_mean_threshold", "global_median_threshold"]

        if method not in _methods:
            raise ValueError("Method %s doesn't exist. Only %s are available" %
                             (method, _methods))

        # Depending on the method selected, the proper function will be selected
        if method == _methods[0]:
            encoder = self.__encode_using_threshold
        elif method == _methods[2]:
            encoder = self.__encodeUsingDerivative
        elif method == _methods[1]:
            encoder = self.__encodeUsingHysteresis
        elif method == _methods[3]:
            encoder = self.__encodeUsingPrimitive
        elif method == _methods[4]:
            encoder = self.__encode_using_mean_threshold
        elif method == _methods[7]:
            encoder = self.__encode_using_gmean_threshold
        elif method == _methods[5]:
            encoder = self.__encode_using_mean_threshold()
        elif method == _methods[6]:
            encoder = self.__encodeUsingDynamicThreshold
        elif method == _methods[8]:
            encoder = self.__encode_using_gmedian_threshold
        else:
            encoder = None

        # Apply smoothing if requested
        if smooth is not None:
            temporal_prediction = self.__smooth(temporal_prediction,
                                                method=smooth, **kwargs)

        # Now that we have the strong prediction, we can assign the value to the
        # two attributes nb_frame and frame_length
        self.nb_frame = temporal_prediction.shape[1]
        self.frame_length = self.clip_length / self.nb_frame

        # Execute the selected segmentation algorithm and recover its results
        return encoder(temporal_prediction, **kwargs)

    def __encode_using_threshold(self, temporal_prediction: np.array,
                                 **kwargs) -> list:
        """
        A basic threshold algorithm. Each value that are above the threshold
        will be part of a valid segment, an invalid one otherwise.

        Args:
            temporal_prediction (np.array):
            thresholds (list): The list of threshold to apply. It must
            contain as much threshold that there is classes. (one threshold for
            each class.
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
                    window = bin_prediction_per_class[i: i + max_hole_size]

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

    def __encode_using_gmean_threshold(self, temporal_prediction: np.array,
                                       **kwargs) -> list:
        """
        Using all the temporal prediction, the mean of each curve and for
        each class is computed and will be choose as threshold. Then call the
        `__encode_using_threshold` function to apply it.

        Args:
            temporal_prediction (np.array):
            global (bool): If the threshold must be global (one for all
            classes) or independent (one for each class)
        """

        # Recover the kwargs arguments
        _global = kwargs.get("global", False)

        total_thresholds = []

        for clip in temporal_prediction:
            total_thresholds.append([curve.mean() for curve in clip.T])

        total_thresholds = np.array(total_thresholds)

        if _global:
            return self.__encode_using_threshold(
                temporal_prediction,
                thresholds=total_thresholds.mean(axis=0),
                **kwargs)
        else:
            return self.__encode_using_threshold(
                temporal_prediction,
                thresholds=[total_thresholds.mean()] * len(self.classes),
                **kwargs)

    def __encode_using_gmedian_threshold(self, temporalPrediction: np.array, **kwargs) -> list:
        """
        Using all the temporal prediction, the mean of each curve and for
        each class is compted and will be choose as threshold. Then call the
        `__encoder_using_threshold` function to apply it.

        Args:
            temporalPrediction (np.array):
            global (bool): If the threshold must be global (one for all
            classes) or independent (one for each class)
        """

        # Recover the kwargs arguments
        _global = kwargs.get("global", False)

        total_thresholds = []
        for clip in temporalPrediction:

            # compute unique threshold for this file
            total_thresholds.append( [curve[len(curve) // 2] for curve in clip.T] )

        total_thresholds = np.array(total_thresholds)

        if _global:
            return self.__encode_using_threshold(
                temporalPrediction,
                thresholds=total_thresholds.mean(axis=0),
                **kwargs)
        else:
            return self.__encode_using_threshold(
                temporalPrediction,
                thresholds=[total_thresholds.mean()] * len(self.classes),
                **kwargs)

    def __encode_using_mean_threshold(self, temporalPrediction: np.array, **kwargs) -> list:
        """
        This algorithm is similar to the global mean threshold but will compute
        new threshold(s) (global or independent) for each files.

        Args:
            temporalPrediction (np.array): A 3-dimension numpy array (<nb clip>,
                <nb frame>, <nb class>)
            global (bool): If the threshold must be global (one for all
            classes) or independent (one for each class)

        Returns:
            the result of the system under the form of a strong annotation text
            where each line represent on timed event
        """

        # Recover the kwargs arguments
        _global = kwargs.get("global", False)

        output = []

        # Merging "hole" that are smaller than 200 ms
        step_length = self.clip_length / temporalPrediction.shape[1] * 1000
        max_hole_size = int(self.temporal_precision / step_length)

        for clip in temporalPrediction:
            labeled = dict()
            _clip = clip.copy()

            # compute unique threshold for this file globally or independent
            thresholds = np.array([curve.mean() for curve in _clip.T])

            if _global:
                thresholds = [thresholds.mean()] * len(self.classes)

            # Binarize using the given thresholds
            if thresholds is not None:
                _clip[_clip > thresholds] = 1
                _clip[_clip <= thresholds] = 0

            cls = 0
            for binPredictionPerClass in _clip.T:
                # convert the binarized list into a list of tuple representing
                # the element and it's number of # occurrence. The order is
                # conserved and the total sum should be equal to 10s

                # first pass --> Fill the holes
                for i in range(len(binPredictionPerClass) - max_hole_size):
                    window = binPredictionPerClass[i: i + max_hole_size]

                    if window[0] == window[-1] == 1:
                        window[:] = [window[0]] * max_hole_size

                # second pass --> split into segments
                converted = []
                cpt = 0
                nb_segment = 0
                previous_elt = None
                for element in binPredictionPerClass:
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
                #                 if nb_segment == 1:
                converted.append((previous_elt, cpt))

                labeled[cls] = converted.copy()
                cls += 1

            output.append(labeled)

        return output

    def __encode_using_median_treshold(self, temporal_prediction: np.array, **kwargs) -> list:
        """
        This algorithm is similar to the global median threshold but will
        compute new threshold(s) (global or independent) for each files.

        Args:
            temporal_prediction (np.array): A 3-dimension numpy array (nb clip,
                nb frame, nb class)
            global (bool): If the threshold must be global (one for all
            classes) or independent (one for each class)

        Returns:
            The result of the system under the form of a strong annotation text
            where each line represent one time event
        """

        # Recover the kwargs arguments
        _global = kwargs.get("global", False)

        output = []

        # Merging "hole" that are smaller than 200 ms
        step_length = self.clip_length / temporal_prediction.shape[1] * 1000
        max_hole_size = int(self.temporal_precision / step_length)

        for clip in temporal_prediction:
            labeled = dict()
            _clip = clip.copy()

            # compute unique threshold for this file
            thresholds = np.array([curve[len(curve) // 2] for curve in clip.T])

            if _global:
                thresholds = [thresholds.mean()] * len(self.classes)

            # Binarize using the given thresholds
            if thresholds is not None:
                _clip[_clip > thresholds] = 1
                _clip[_clip <= thresholds] = 0

            cls = 0
            for binPredictionPerClass in _clip.T:
                # convert the binarized list into a list of tuple representing
                # the element and it's number of # occurrence. The order is
                # conserved and the total sum should be equal to 10s

                # first pass --> Fill the holes
                for i in range(len(binPredictionPerClass) - max_hole_size):
                    window = binPredictionPerClass[i: i + max_hole_size]

                    if window[0] == window[-1] == 1:
                        window[:] = [window[0]] * max_hole_size

                # second pass --> split into segments
                converted = []
                cpt = 0
                nb_segment = 0
                previous_elt = None
                for element in binPredictionPerClass:
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
                #                 if nb_segment == 1:
                converted.append((previous_elt, cpt))

                labeled[cls] = converted.copy()
                cls += 1

            output.append(labeled)

        return output
