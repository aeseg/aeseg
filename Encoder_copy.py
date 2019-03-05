import sys
import numpy as np
import copy


class Encoder:
    def __init__(self):
        self.frameLength = 0
        self.nbFrame = 0

    def __pad(self, array: np.array, window_size: int, method: str = "same"):
        """Pad and array using the methods given and a window_size.

        Args:
            array (np.array): the array to pad
            window_size (int): the size of the working window
            method (str): methods of padding, two available "same" | "reflect"

        Returns:
            the padded array
        """
                
        if method == "same":
            missing = int(window_size / 2)
            first = np.array([array[0]] * missing)
            last = np.array([array[-1]] * missing)
            
            output = np.concatenate((first, array, last))

        return output


    # ===============================================================================
    #
    #     SMOOTHING FUNCTIONS:
    #
    # ===============================================================================
    def __smooth(self, temporalPrediction: np.array, method: str = "smoothMovingAvg", **kwargs) -> np.array:
        """For smoothing the curve of the prediction curves.

        Args:
            temporalPrediction (np.array): The temporalPrediction of the second
                model (TimeDistributed Dense output)
            method (str): The algorithm to use for smoothing the curves
            kwargs: See argument list for the smoothing algorithm
        """
        
        _methods = ["smoothMovingAvg", "smoothMovingMedian"]
        if method not in _methods:
            print("method %s doesn't exist. Only ", _methods, " available")
            sys.exit(1)

        if method == _methods[0]: smoother = self.__smoothMovingAvg
        elif method == _methods[1]: smoother = self.__smoothMedian
        else:
            return

        return smoother(temporalPrediction, **kwargs)

    def __smoothMedian(self, temporalPrediction: np.array, windows_len: int = 11, **kwargs):
        """
        Args:
            temporalPrediction (np.array):
            windows_len (int):
            **kwargs:
        """
        pass

    def __smoothMovingAvg(self, temporalPrediction: np.array, window_len: int = 11, **kwargs):
        """
        Args:
            temporalPrediction (np.array):
            window_len (int):
            **kwargs:
        """
        def smooth(data, window_len = 11):
            window_len = int(window_len)

            if window_len < 3:
                return data

            s = np.r_[2 * data[0] - data[window_len - 1::-1], data, 2 * data[-1] - data[-1:-window_len:-1]]
            w = np.ones(window_len, 'd')
            y = np.convolve(w / w.sum(), s, mode='same')
            return y[window_len:-window_len + 1]

        # core
        smoothed_temporal_prediction = temporalPrediction.copy()
        
        for clipInd in range(len(smoothed_temporal_prediction)):
            clip = smoothed_temporal_prediction[clipInd]

            for clsInd in range(len(clip.T)):
                clip.T[clsInd] = smooth(clip.T[clsInd], window_len)
                
        return smoothed_temporal_prediction


    def encode(self, temporal_prediction: np.array, method: str = "threshold", smooth: str = None, **kwargs) -> str:
        """Perform the localization of the sound event present in the file.

        Using the temporal prediction provided y the last step of the system,
        it will "localize" the sound event inside the file under the form of a
        strongly annotated line. (see DCASE2018 task 4 strong label exemple).
        There is two methods implemented here, one using a simple threshold
        based segmentation and an other using a modulation system based on the
        variance of the prediction over the time.

        Args:
            temporal_prediction (np.array): A 3-dimension numpy array (<nb
                clip>, <nb frame>, <nb class>)
            method (str): The segmentation method to use [threshold | hysteresis
                | derivative | primitive]
            smooth (str): The smoothing method to use [smoothMovingAvg]
            kwargs: See the segmentation method parameters

        Returns:
            The result of the system under the form of a strong annotation text
            where each represent on timed event
        """
        # parameters verification
        _methods=["threshold", "hysteresis", "derivative", "primitive", "mean_threshold", "median_threshold", "dynamic_threshold",
                 "global_mean_threshold", "global_median_threshold"]
        if method not in _methods:
            print("method %s doesn't exist. Only", _methods, " available")
            sys.exit(1)

        if method == _methods[0]: encoder = self.__encodeUsingThreshold
        elif method == _methods[2]: encoder = self.__encodeUsingDerivative
        elif method == _methods[1]: encoder = self.__encodeUsingHysteresis
        elif method == _methods[3]: encoder = self.__encodeUsingPrimitive
        elif method == _methods[4]: encoder = self.__encodeUsingMeanThreshold
        elif method == _methods[7]: encoder = self.__encode_using_global_mean_threshold
        elif method == _methods[5]: encoder = self.__encodeUsingMedianTreshold
        elif method == _methods[6]: encoder = self.__encodeUsingDynamicThreshold
        elif method == _methods[8]: encoder = self.__encode_using_global_median_threshold
        else:
            sys.exit(1)

        # Apply smoothing if requested
        if smooth is not None:
            temporal_prediction = self.__smooth(temporal_prediction, method=smooth, **kwargs)

        self.nbFrame = temporal_prediction.shape[1]
        self.frameLength = 10 / self.nbFrame
        
        return encoder(temporal_prediction, **kwargs)
    
    def __encode_using_global_mean_threshold(self, temporalPrediction: np.array, **kwargs) -> list:
        """Global mean threshold based localization of the sound event in the
        clip using the temporal prediction.

        A mean threshold is computed globally over the whole dataset

        Args:
            temporalPrediction (np.array):
            **kwargs:
        """
        output = []
        temporalPrecision = 200  # ms

        # Merging "hole" that are smaller than 200 ms
        stepLength = DCASE2018.CLIP_LENGTH / temporalPrediction.shape[1] * 1000  # in ms
        maxHoleSize = int(temporalPrecision / stepLength)
        
        total_thresholds = []
        for clip in temporalPrediction:
            labeled = dict()
            _clip = clip.copy()

            # compute unique threshold for this file
            total_thresholds.append( [curve.mean() for curve in _clip.T] )
            
        total_thresholds = np.array(total_thresholds)
        return self.__encodeUsingThreshold(temporalPrediction,
                                          thresholds=total_thresholds.mean(axis=0),
                                          **kwargs)
    
    def __encode_using_global_median_threshold(self, temporalPrediction: np.array, **kwargs) -> list:
        """Global mean threshold based localization of the sound event in the
        clip using the temporal prediction.

        A mean threshold is computed globally over the whole dataset

        Args:
            temporalPrediction (np.array):
            **kwargs:
        """
        output = []
        temporalPrecision = 200  # ms

        # Merging "hole" that are smaller than 200 ms
        stepLength = DCASE2018.CLIP_LENGTH / temporalPrediction.shape[1] * 1000  # in ms
        maxHoleSize = int(temporalPrecision / stepLength)
        
        total_thresholds = []
        for clip in temporalPrediction:
            labeled = dict()
            _clip = clip.copy()

            # compute unique threshold for this file
            total_thresholds.append( [curve[len(curve) // 2] for curve in _clip.T] )
            
        total_thresholds = np.array(total_thresholds)
        return self.__encodeUsingThreshold(temporalPrediction,
                                          thresholds=total_thresholds.mean(axis=0),
                                          **kwargs)

    def __encodeUsingMeanThreshold(self, temporalPrediction: np.array, **kwargs) -> list:
        """Mean threshold based localization of the sound event in the clip
        using the temporal prediction.

        For each class and each file, the mean prediction is computed and
        used as threshold

        Args:
            temporalPrediction (np.array): A 3-dimension numpy array (<nb clip>,
                <nb frame>, <nb class>)
            **kwargs:

        Returns:
            the result of the system under the form of a strong annotation text
            where each line represent on timed event
        """
        output = []
        temporalPrecision = 200  # ms

        # Merging "hole" that are smaller than 200 ms
        stepLength = DCASE2018.CLIP_LENGTH / temporalPrediction.shape[1] * 1000  # in ms
        maxHoleSize = int(temporalPrecision / stepLength)

        for clip in temporalPrediction:
            labeled = dict()
            _clip = clip.copy()

            # compute unique threshold for this file
            thresholds = [curve.mean() for curve in _clip.T]

            # Binarize using the given thresholds
            if thresholds is not None:
                _clip[_clip > thresholds] = 1
                _clip[_clip <= thresholds] = 0

            cls = 0
            for binPredictionPerClass in _clip.T:
                # convert the binarized list into a list of tuple representing the element and it's number of
                # occurrence. The order is conserved and the total sum should be equal to 10s

                # first pass --> Fill the holes
                for i in range(len(binPredictionPerClass) - maxHoleSize):
                    window = binPredictionPerClass[i: i + maxHoleSize]

                    if window[0] == window[-1] == 1:
                        window[:] = [window[0]] * maxHoleSize

                # second pass --> split into segments
                converted = []
                cpt = 0
                nbSegment = 0
                previousElt = None
                for element in binPredictionPerClass:
                    if previousElt is None:
                        previousElt = element
                        cpt += 1
                        nbSegment = 1
                        continue

                    if element == previousElt:
                        cpt += 1

                    else:
                        converted.append((previousElt, cpt))
                        previousElt = element
                        nbSegment += 1
                        cpt = 1

                # case where the class is detect during the whole clip
                #                 if nbSegment == 1:
                converted.append((previousElt, cpt))

                labeled[cls] = copy.copy(converted)
                cls += 1

            output.append(labeled)

        return output

    def __encodeUsingMedianTreshold(self, temporal_prediction: np.array, **kwargs) -> list:
        """Median threshold based localization of the sound event in the clip
        using the temporal prediction.

        For each class and each file, the median of the prediction will be
        used as threhsold

        Args:
            temporal_prediction (np.array): A 3-dimension numpy array (nb clip,
                nb frame, nb class)
            **kwargs:

        Returns:
            The result of the system under the form of a strong annotation text
            where each line represent one time event
        """
        output = []
        temporalPrecision = 200  # ms

        # Merging "hole" that are smaller than 200 ms
        stepLength = DCASE2018.CLIP_LENGTH / temporal_prediction.shape[1] * 1000  # in ms
        maxHoleSize = int(temporalPrecision / stepLength)

        for clip in temporal_prediction:
            labeled = dict()
            _clip = clip.copy()

            # compute unique threshold for this file
            thresholds = [curve[int(len(curve) / 2)] for curve in clip.T]
            print(thresholds)

            # Binarize using the given thresholds
            if thresholds is not None:
                _clip[_clip > thresholds] = 1
                _clip[_clip <= thresholds] = 0

            cls = 0
            for binPredictionPerClass in _clip.T:
                # convert the binarized list into a list of tuple representing the element and it's number of
                # occurrence. The order is conserved and the total sum should be equal to 10s

                # first pass --> Fill the holes
                for i in range(len(binPredictionPerClass) - maxHoleSize):
                    window = binPredictionPerClass[i: i + maxHoleSize]

                    if window[0] == window[-1] == 1:
                        window[:] = [window[0]] * maxHoleSize

                # second pass --> split into segments
                converted = []
                cpt = 0
                nbSegment = 0
                previousElt = None
                for element in binPredictionPerClass:
                    if previousElt is None:
                        previousElt = element
                        cpt += 1
                        nbSegment = 1
                        continue

                    if element == previousElt:
                        cpt += 1

                    else:
                        converted.append((previousElt, cpt))
                        previousElt = element
                        nbSegment += 1
                        cpt = 1

                # case where the class is detect during the whole clip
                #                 if nbSegment == 1:
                converted.append((previousElt, cpt))

                labeled[cls] = copy.copy(converted)
                cls += 1

            output.append(labeled)

        return output

    def __encodeUsingDynamicThreshold(self, temporal_prediction: np.array, **kwargs) -> list:
        """
        Args:
            temporal_prediction (np.array):
            **kwargs:
        """
        pass



    def __encodeUsingHysteresis(self, temporalPrediction: np.array, **kwargs) -> list:
        """Hysteresys based localization of the sound event in the clip using
        the temporal prediction.

        Args:
            temporalPrediction (np.array): A 3-dimension numpy array (<nb clip>,
                <nb frame>, <nb class>)
            kwargs: Extra arguments - "high" and "low" (thresholds for the
                hysteresis)

        Returns:
            the result of the system under the form of a strong annotation text
            where each line represent on timed event
        """
        low = kwargs["low"] if "low" in kwargs.keys() else 0.4
        high = kwargs["high"] if "high" in kwargs.keys() else 0.6
        prediction = temporalPrediction

        output = [] 

        for clip in prediction: 
            labeled = dict() 

            cls = 0 
            for predictionPerClass in clip.T: 
                converted = list() 
                segment = [0, 0] 
                nbSegment = 1 
                for i in range(len(predictionPerClass)): 
                    element = predictionPerClass[i] 

                    # first element 
                    if i == 0: 
                        segment = [1.0, 1] if element > high else [0.0, 1] 

                    # then 
                    if element > high: 
                        if segment[0] == 1: 
                            segment[1] += 1 
                        else: 
                            converted.append(segment) 
                            nbSegment += 1 
                            segment = [1.0, 1] 
                    elif low <= element: 
                        segment[1] += 1 
                    else: 
                        if segment[0] == 0: 
                            segment[1] += 1 
                        else: 
                            converted.append(segment) 
                            nbSegment += 1 
                            segment = [0.0, 0] 

    #             if nbSegment == 1: 
                converted.append(segment) 

                labeled[cls] = copy.copy(converted) 
                cls += 1 

            output.append(labeled) 

        return output

    def __encodeUsingThreshold(self, temporalPrediction: np.array, **kwargs) -> list:
        """Threshold based localization of the sound event in the clip using the
        temporal prediction.

        Args:
            temporalPrediction (np.array): A 3-dimension numpy array (<nb clip>,
                <nb frame>, <nb class>)
            **kwargs:

        Returns:
            The result of the system under the form of a strong annotation text
            where each represent on timed event
        """

        output = []
        temporalPrecision = 200        # ms

        thresholds = kwargs["thresholds"] if "thresholds" in kwargs.keys() else None

        # Binarize if requested using the given thresholds
        if thresholds is not None:
            binPrediction = temporalPrediction.copy()
            binPrediction[binPrediction > thresholds] = 1
            binPrediction[binPrediction <= thresholds] = 0

        # Merging "hole" that are smaller than 200 ms
        stepLength = DCASE2018.CLIP_LENGTH / temporalPrediction.shape[1] * 1000     # in ms
        maxHoleSize = int(temporalPrecision / stepLength)

        for clip in binPrediction:
            labeled = dict()

            cls = 0
            for binPredictionPerClass in clip.T:
                # convert the binarized list into a list of tuple representing the element and it's number of
                # occurrence. The order is conserved and the total sum should be equal to 10s

                # first pass --> Fill the holes
                for i in range(len(binPredictionPerClass) - maxHoleSize):
                    window = binPredictionPerClass[i : i+maxHoleSize]

                    if window[0] == window[-1] == 1:
                        window[:] = [window[0]] * maxHoleSize

                # second pass --> split into segments
                converted = []
                cpt = 0
                nbSegment = 0
                previousElt = None
                for element in binPredictionPerClass:
                    if previousElt is None:
                        previousElt = element
                        cpt += 1
                        nbSegment = 1
                        continue

                    if element == previousElt:
                        cpt += 1

                    else:
                        converted.append((previousElt, cpt))
                        previousElt = element
                        nbSegment += 1
                        cpt = 1

                # case where the class is detect during the whole clip
#                 if nbSegment == 1:
                converted.append((previousElt, cpt))

                labeled[cls] = copy.copy(converted)
                cls += 1

            output.append(labeled)

        return output

#     def __encodeUsingDerivative(self, temporalPrediction: np.array, **kwargs) -> list:
#         """ Threshold based localization of the sound event in the clip using the temporal prediction.

#         :param temporalPrediction: A 3-dimension numpy array (<nb clip>, <nb frame>, <nb class>)
#         :param rising: the rising steep angle at which a segment should start
#         :param decreasing: the decreasing steep angle at which the segment should end
#         :param flat: The minimum angle to consider the slop as flat
#         :param window_size: The size of the sliding window
#         :param hig: A threshold that overide the steepness (exemple, even if it increase slowly, as soon as it goes over 0.8 the segment start)
#         :param padding: The padding method to apply
        
#         :return: The result of the system under the form of a strong annotation text where each represent on timed event
#         """

#         def futureIsFlat(prediction: np.array, currentPos: int, flat: float = 0.05, window_size: int = 5) -> bool:
#             """
#             Detect what is following is "kinda" flat.
#             :param prediction: The prediction values of the current class
#             :param currentPos: The current position of the window (left side)
#             :return: True is the near future of the curve is flat, False otherwise
#             """
#             slopes = 0

#             # if not future possible (end of the curve)
#             if (currentPos + 2 * window_size) > len(prediction):
#                 return False

#             # sum the slope value for the next <window_size> window
#             for i in range(currentPos, currentPos + 2 * window_size):
#                 window = prediction[i:i + window_size]
#                 slopes += window[-1] - window[0]

#             averageSlope = slopes / 2 * window_size

#             # is approximately flat, the return True, else False
#             return abs(averageSlope) < flat

#         # retreive the argument from kwargs
#         keys = kwargs.keys()
#         rising = kwargs["rising"] if "rising" in keys else 0.5
#         decreasing = kwargs["decreasing"] if "decreasing" in keys else -0.5
#         flat = kwargs["flat"] if "flat" in keys else 0.05
#         window_size = kwargs["window_size"] if "window_size" in keys else 5
#         high = kwargs["high"] if "high" in keys else 0.5
#         padding = kwargs["padding"] if "padding" in keys else "same"

#         output = []

#         for clip in temporalPrediction:
#             cls = 0
#             labeled = dict()

#             for predictionPerClass in clip.T:
#                 paddedPredictionPerClass = self.__pad(predictionPerClass, window_size, method=padding)

#                 nbSegment = 1
#                 segments = []
#                 segment = [0.0, 0]
#                 for i in range(len(paddedPredictionPerClass) - window_size):
#                     window = paddedPredictionPerClass[i:i+window_size]

#                     slope = window[-1] - window[0]

#                     # first element
#                     if i == 0:
#                         segment = [1.0, 1] if window[0] > high else [0.0, 1]

#                     # rising slope while on "low" segment --> changing segment
#                     if slope > rising and segment[0] == 0:
#                         segments.append(segment)
#                         nbSegment += 1
#                         segment = [1.0, 1]

#                     # rising slope while on "high" segment --> same segment
#                     elif slope > rising and segment[0] == 1:
#                         segment[1] += 1

#                     # decreasing slope while on "low" segment --> same segment
#                     elif slope < decreasing and segment[0] == 0:
#                         segment[1] += 1

#                     # decreasing slope while on "high" segment --> one extra condition, future is flat ?
#                     elif slope < decreasing and segment[0] == 1:
#                         # is there is no flat plateau right after --> same segment
#                         if not futureIsFlat(paddedPredictionPerClass, i, flat, window_size):
#                             segment[1] += 1

#                         # Otherwise --> change segment
#                         else:
#                             segments.append(segment)
#                             nbSegment += 1
#                             segment = [0.0, 1]


#                     else:
#                         segment[1] += 1

# #                 if nbSegment == 1:
#                 segments.append(copy.copy(segment))

#                 labeled[cls] = segments
#                 cls += 1

#             output.append(labeled)
#         return output

    def __encodeUsingDerivative(self, temporalPrediction: np.array, **kwargs) -> list:
        """Threshold based localization of the sound event in the clip using the
        temporal prediction.

        Args:
            temporalPrediction (np.array): A 3-dimension numpy array (<nb clip>,
                <nb frame>, <nb class>)
            **kwargs:

        Returns:
            The result of the system under the form of a strong annotation text
            where each represent on timed event
        """

        # retreive the argument from kwargs
        keys = kwargs.keys()
        rising = kwargs["rising"] if "rising" in keys else 0.5
        decreasing = kwargs["decreasing"] if "decreasing" in keys else -0.5
        flat = kwargs["flat"] if "flat" in keys else 0.05
        window_size = int(kwargs["window_size"]) if "window_size" in keys else 5
        high = kwargs["high"] if "high" in keys else 0.5
        padding = kwargs["padding"] if "padding" in keys else "same"

        output = []

        for clip in temporalPrediction:
            cls = 0
            labeled = dict()

            for predictionPerClass in clip.T:
                paddedPredictionPerClass = self.__pad(predictionPerClass, window_size, method=padding)

                nbSegment = 1
                segments = []
                segment = [0.0, 0]
                for i in range(len(paddedPredictionPerClass) - window_size):
                    window = paddedPredictionPerClass[i:i+window_size]
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
                                nbSegment += 1
                                segment = [0.0, 1]
                            else:
                                segment[1] += 1
                    
                    # if on "low" segment
                    else:
                        
                        # if above high threshold
                        if window[0] > high:
                            segments.append(segment)
                            nbSegment += 1
                            segment = [1.0, 1]
                        
                        else:
                            if slope > rising:
                                segments.append(segment)
                                nbSegment += 1
                                segment = [1.0, 1]
                            else:
                                segment[1] += 1

                segments.append(copy.copy(segment))

                labeled[cls] = segments
                cls += 1

            output.append(labeled)
        return output
    
    def __encodeUsingPrimitive(self, temporalPrediction: np.array, **kwargs) -> list:
        """Area under the curve based localization of the sound event using the
        temporal prediction.

        Given a sliding window, the area under the curve of the window is
        computed and, The area under the curve is computed and depending on
        whether it is above a threshold or not the segment will be considered.
        implementation based of the composite trapezoidal rule.

        Args:
            temporalPrediction (np.array): A 3-dimension numpy array (<nb clip>,
                <nb frame>, <nb class>)
            **kwargs:

        Returns:
            The result of the system under the form of a strong annotation text
            where each represent on timed event
        """

        def area(window: list) -> float:
            """ Compute the area under the curve inside a window

            :param window: the current window
            :return: the area under the curve
            """
            area = 0
            for i in range(len(window) - 1):
                area += (window[i+1] + window[i]) / 2

            return area

        # retreiving extra arguments
        keys = kwargs.keys()
        window_size = kwargs["window_size"] if "window_size" in keys else 5
        threshold = kwargs["threshold"] if "threshold" in keys else window_size / 4
        stride = kwargs["stride"] if "stride" in keys else 1
        padding = kwargs["padding"] if "padding" in keys else "same"

        output = []
        for clip in temporalPrediction:
            labeled = dict()
            cls = 0
            for predictionPerClass in clip.T:
                paddedPredictionPerClass = self.__pad(predictionPerClass, window_size, method=padding)
                
                nbSegment = 1
                segments = []
                segment = None
                for i in range(0, len(paddedPredictionPerClass) - window_size, stride):
                    window = paddedPredictionPerClass[i:i+window_size]
                    wArea = area(window)

                    # first element
                    if i == 0:
                        segment = [1.0, 1] if wArea > threshold else [0.0, 1]

                    # then
                    if wArea > threshold and segment[0] == 1:
                        segment[1] += 1

                    elif wArea > threshold and segment[0] == 0:
                        segments.append(segment)
                        nbSegment += 1
                        segment = [1.0, 1]

                    elif wArea <= threshold and segment[0] == 0:
                        segment[1] += 1

                    elif wArea <= threshold and segment[0] == 1:
                        segments.append(segment)
                        nbSegment += 1
                        segment = [0.0, 1]

#                 if nbSegment == 1:
                segments.append(segment)

                labeled[cls] = copy.copy(segments)
                cls += 1

            output.append(labeled)
        return output

    def parse(self, allSegments: list, testFilesName: list) -> str:
        """Transform a list of segment into a txt file ready for evaluation.

        of key to the number of class :param testFilesName: the list of
        filename in the same order than the list allSegments :return: a str file
        ready for evaluation using dcase_util evaluation_measure.py

        Args:
            allSegments (list): a list of dict of 10 key. the list length is
                equal to the number of file, the dict number
            testFilesName (list):
        """
        output = ""

        for clipIndex in range(len(allSegments)):
            clip = allSegments[clipIndex]

            for cls in clip:
                start = 0

                for segment in clip[cls]:
                    if segment[0] == 1.0:
                        output += "%s\t%f\t%f\t%s\n" % (
                            testFilesName[clipIndex],
                            start * self.frameLength,
                            (start + segment[1]) * self.frameLength,
                            DCASE2018.class_correspondance_reverse[cls]
                        )
                    start += segment[1]

        return output

if __name__=='__main__':
    import random
    e = Encoder()

    # create fake data (temporal prediction)
    def mRandom():
        r = random.random()
        return r

    def fakeTemporalPrediction():
        prediction = []
        for i in range(10):
            clip = []
            for j in range(200):
                score = [mRandom() for k in range(10)]
                clip.append(score)
            prediction.append(clip)

        prediction = np.array(prediction)

        #o = e.encode(prediction)       # basic thresold with hold filling
        o = e.encode(prediction, method="primitive")
        for k in o:
            print(len(k[0]), k[0])
        t = e.parse(o, prediction[:,0,0])


    fakeTemporalPrediction()
