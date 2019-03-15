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
        self.class_correspondence = dict(zip(classes, range(len(classes))))
        self.class_correspondence_reverse = dict(zip(range(len(classes)), classes))

    def parse(self, all_segments: list, test_files_name: list) -> str:
        """
        Transform a list of segment into a string ready for evaluation with
        sed_eval.

        Args:
            all_segments (list): a list of dict of 10 key. the list length is
                equal to the number of file, the dict number
            test_files_name (list): The list of the file names in the same
            order than the temporal prediction used to perform the segmentation
        """
        output = ""

        for clipIndex in range(len(all_segments)):
            clip = all_segments[clipIndex]

            for cls in clip:
                start = 0

                for segment in clip[cls]:
                    if segment[0] == 1.0:
                        output += "%s\t%f\t%f\t%s\n" % (
                            test_files_name[clipIndex],
                            start * self.frame_length,
                            (start + segment[1]) * self.frame_length,
                            self.class_correspondence_reverse[cls]
                        )
                    start += segment[1]

        return output

    def encode(self, temporal_prediction: np.array, method: str = "threshold",
               smooth: str = None, **kwargs) -> list:
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
        _methods = ["threshold", "hysteresis", "derivative", "mean_threshold",
                    "median_threshold", "dynamic_threshold",
                    "global_mean_threshold", "global_median_threshold"]

        if method not in _methods:
            raise ValueError("Method %s doesn't exist. Only %s are available" %
                             (method, _methods))

        # Depending on the method selected, the proper function will be selected
        encoder = None

        if method == _methods[0]:
            encoder = self.__encode_using_threshold
        elif method == _methods[2]:
            encoder = self.__encode_using_derivative
        elif method == _methods[1]:
            encoder = self.__encode_using_hysteresis
        elif method == _methods[3]:
            encoder = self.__encode_using_mean_threshold
        elif method == _methods[6]:
            encoder = self.__encode_using_gmean_threshold
        elif method == _methods[4]:
            encoder = self.__encode_using_median_treshold
        elif method == _methods[5]:
            encoder = self.__encode_using_dynamic_threshold
        elif method == _methods[7]:
            encoder = self.__encode_using_gmedian_threshold

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

    def __encode_using_derivative(self, temporal_prediction: np.array,
                                  rising: float = 0.5, decreasing: float = -0.5,
                                  window_size: int = 5, high: float = 0.8,
                                  padding: str = "same") -> list:
        """
        The derivative create segment based on the intensity of the variation
        of the temporal prediction curve. If the prediction rise above a certain
        threshold `rising` then a valid segment start. If it decrease faster
        than the `decreasing` threshold, then a valid segment finish. If the
        prediction start with a high value, of rise slowly but high, then an
        absolute (and global) threshold `high` is used. (it works like a normal
        threhsold)


        Args:
            temporal_prediction (np.array): A 3-dimension numpy array (<nb
                clip>, <nb frame>, <nb class>)
            rising (float): Must be between 0 and 1, rising threshold. When the
            slope is steeper
            decreasing (float): Must be between 0 and 1, decreasing threshold
            that specifies the end of the current valid segment.
            window_size (int): size of the processing window
            high (float): minimum prediction value that trigger a valid
            segment, even if the condition are not fulfill.
            padding (str): The padding method to used on the curves

        Returns:
            The result of the system under the form of a strong annotation text
            where each represent on timed event
        """

        output = []

        for clip in temporal_prediction:
            cls = 0
            labeled = dict()

            for prediction_per_class in clip.T:
                padded_prediction_per_class = self.__pad(prediction_per_class,
                                                         window_size,
                                                         method=padding)

                nb_segment = 1
                segments = []
                segment = [0.0, 0]
                for i in range(len(padded_prediction_per_class) - window_size):
                    window = padded_prediction_per_class[i:i + window_size]
                    slope = (window[-1] - window[0]) / window_size

                    # first element
                    if i == 0:
                        segment = [1.0, 1] if window[0] > high else [0.0, 1]

                    # if on "high" segment
                    if segment[0] == 1:

                        # if above high threshol
                        if window[0] > high:
                            segment[1] += 1

                        else:
                            # if decreasing threshold is reach
                            if slope < decreasing:
                                segments.append(segment)
                                nb_segment += 1
                                segment = [0.0, 1]
                            else:
                                segment[1] += 1

                    # if on "low" segment
                    else:

                        # if above high threshold
                        if window[0] > high:
                            segments.append(segment)
                            nb_segment += 1
                            segment = [1.0, 1]

                        else:
                            if slope > rising:
                                segments.append(segment)
                                nb_segment += 1
                                segment = [1.0, 1]
                            else:
                                segment[1] += 1

                segments.append(segment.copy())

                labeled[cls] = segments
                cls += 1

            output.append(labeled)
        return output

    def __encode_using_hysteresis(self, temporal_prediction: np.array,
                                  **kwargs) -> list:
        """The hysteresis based segmentation algorithm require two threhsolds. A
        high value to decided when the segment should start and a low value to
        decided when to finish the segment. It perform better when the temporal
        prediction is noisy

        Args:
            temporal_prediction (np.array): A 3-dimension numpy array (<nb
                clip>, <nb frame>, <nb class>)
            kwargs: Extra arguments - "high" and "low" (thresholds for the
                hysteresis)

        Returns:
            the result of the system under the form of a strong annotation text
            where each line represent on timed event
        """
        low = kwargs["low"] if "low" in kwargs.keys() else 0.4
        high = kwargs["high"] if "high" in kwargs.keys() else 0.6
        prediction = temporal_prediction

        output = []

        for clip in prediction:
            labeled = dict()

            cls = 0
            for prediction_per_class in clip.T:
                converted = list()
                segment = [0, 0]
                nb_segment = 1
                for i in range(len(prediction_per_class)):
                    element = prediction_per_class[i]

                    # first element
                    if i == 0:
                        segment = [1.0, 1] if element > high else [0.0, 1]

                        # then
                    if element > high:
                        if segment[0] == 1:
                            segment[1] += 1
                        else:
                            converted.append(segment)
                            nb_segment += 1
                            segment = [1.0, 1]
                    elif low <= element:
                        segment[1] += 1
                    else:
                        if segment[0] == 0:
                            segment[1] += 1
                        else:
                            converted.append(segment)
                            nb_segment += 1
                            segment = [0.0, 0]

                converted.append(segment)

                labeled[cls] = converted.copy()
                cls += 1

            output.append(labeled)

        return output

    def __encode_using_threshold(self, temporal_prediction: np.array,
                                 **kwargs) -> list:
        """A basic threshold algorithm. Each value that are above the threshold
        will be part of a valid segment, an invalid one otherwise.

        Args:
            temporal_prediction (np.array):
            **kwargs:
        """
        output = []

        # Recover kwargs arguments
        thresholds = kwargs.get('thresholds', None)
        threshold = kwargs.get("threshold", None)

        # Binarize if requested using the given thresholds
        if thresholds is not None and threshold is not None:
            raise AttributeError("Can't have at the same time \"threshold\" "
                                 "and \"thresholds\" set")

        if threshold:
            thresholds = [threshold] * len(self.classes)

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
        """Using all the temporal prediction, the mean of each curve and for
        each class is computed and will be choose as threshold. Then call the
        `__encode_using_threshold` function to apply it.

        Args:
            temporal_prediction (np.array):
            **kwargs:
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

    def __encode_using_gmedian_threshold(self, temporal_prediction: np.array,
                                         **kwargs) -> list:
        """Using all the temporal prediction, the mean of each curve and for
        each class is compted and will be choose as threshold. Then call the
        `__encoder_using_threshold` function to apply it.

        Args:
            temporal_prediction (np.array):
            **kwargs:
        """

        # Recover the kwargs arguments
        _global = kwargs.get("global", False)

        total_thresholds = []
        for clip in temporal_prediction:
            # compute unique threshold for this file
            total_thresholds.append(
                [curve[len(curve) // 2] for curve in clip.T])

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

    def __encode_using_mean_threshold(self, temporal_prediction: np.array,
                                      **kwargs) -> list:
        """This algorithm is similar to the global mean threshold but will
        compute new threshold(s) (global or independent) for each files.

        Args:
            temporal_prediction (np.array): A 3-dimension numpy array (<nb
                clip>, <nb frame>, <nb class>)
            **kwargs:

        Returns:
            the result of the system under the form of a strong annotation text
            where each line represent on timed event
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

    def __encode_using_median_treshold(self, temporal_prediction: np.array,
                                       **kwargs) -> list:
        """This algorithm is similar to the global median threshold but will
        compute new threshold(s) (global or independent) for each files.

        Args:
            temporal_prediction (np.array): A 3-dimension numpy array (nb clip,
                nb frame, nb class)
            **kwargs:

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

    def __encode_using_dynamic_threshold(self, temporal_prediction: np.array,
                                         **kwargs) -> list:
        """
        Args:
            temporal_prediction (np.array):
            **kwargs:
        """
        raise NotImplementedError()

    # ==========================================================================
    #
    #       SMOOTHING AND UTILITIES
    #
    # ==========================================================================
    def __pad(self, array: np.array, window_size: int,
              method: str = "same") -> np.array:
        """Pad and array using the methods given and a window_size.

        Args:
            array (np.array): the array to pad
            window_size (int): the size of the working window
            method (str): methods of padding, two available "same" | "reflect"

        Returns:
            the padded array
        """

        output = array

        if method == "same":
            missing = int(window_size / 2)
            first = np.array([array[0]] * missing)
            last = np.array([array[-1]] * missing)

            output = np.concatenate((first, array, last))

        elif method == "valid":
            output = array

        elif method == "null":
            missing = int(window_size / 2)
            start, end = [0] * missing, [0] * missing

            output = np.concatenate((start, array, end))

        return output

    # ===============================================================================
    #
    #     SMOOTHING FUNCTIONS:
    #
    # ===============================================================================
    def __smooth(self, temporal_prediction: np.array,
                 method: str = "smoothMovingAvg",
                 **kwargs) -> np.array:
        """For smoothing the curve of the prediction curves.

        Args:
            temporal_prediction (np.array): The temporalPrediction of the second
                model (TimeDistributed Dense output)
            method (str): The algorithm to use for smoothing the curves
            kwargs: See argument list for the smoothing algorithm
        """

        # Check if methods asked exist
        _methods = ["smoothMovingAvg", "smoothMovingMedian"]
        if method not in _methods:
            raise ValueError("Method %s doesn't exist. Only %s available" %
                             (method, _methods))

        # Create smoother (select the algorithm)
        if method == _methods[0]:
            smoother = self.__smooth_moving_avg
        elif method == _methods[1]:
            smoother = self.__smooth_moving_median
        else:
            return

        return smoother(temporal_prediction, **kwargs)

    def __smooth_moving_median(self, temporal_prediction: np.array,
                               windows_len: int = 11, **kwargs):
        """
        Args:
            temporal_prediction (np.array):
            windows_len (int):
            **kwargs:
        """
        raise NotImplementedError()

    def __smooth_moving_avg(self, temporal_prediction: np.array,
                            window_len: int = 5, padding: str = "same",
                            **kwargs):
        """
        Args:
            temporal_prediction (np.array):
            window_len (int): The size of the smoothing window
            padding (str): The padding mode to use
        """

        def smooth(data, _window_len):
            _window_len = int(_window_len)

            if window_len < 3:
                return data

            s = np.r_[
                2 * data[0] - data[_window_len - 1::-1],
                data,
                2 * data[-1] - data[-1:-_window_len:-1]
            ]

            w = np.ones(_window_len, 'd')
            y = np.convolve(w / w.sum(), s, mode=padding)
            return y[_window_len:-_window_len + 1]

        # core
        smoothed_temporal_prediction = temporal_prediction.copy()

        for clip_ind in range(len(smoothed_temporal_prediction)):
            clip = smoothed_temporal_prediction[clip_ind]

            for cls_ind in range(len(clip.T)):
                clip.T[cls_ind] = smooth(clip.T[cls_ind], window_len)

        return smoothed_temporal_prediction
