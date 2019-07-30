Statistics based methods
========================
The statistics-based methods are directly based on the statistics extracted from the temporal predictions
of each sample. The main advantage of these methods is that they are fast and often efficient.

Average
-------
By directly using the system temporal predictions, the mean is computed and used as a segmentation threshold.
Depending on the methods used, the calculation of the average can be done either at the file or dataset level.
Moreover, it can also be independent or dependent on the class.

- :math:`S` represent the dataset
- :math:`p(x)` the temporal prediction output by the system
- :math:`N` the number of file in the dataset.

Class-independent data-wise average
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The average is computed using the temporal prediction from the whole dataset and the classes.

.. math::
    \forall x \in S, Th=\frac{1}{N}\sum p(x)

Class-independent file-wise average
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:math:`n` is the size of the prediction from one file. It depends on the precision of the system.

.. math::
    \forall x \in S, Th_x= \frac{1}{n} \sum p(x)


Class-dependent data-wise average
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:math:`c` represent the class and :math:`C` the ensemble of class the system can target.

.. math::
    \forall x \in S, \forall c \in C, Th_c = \frac{1}{N} \sum p_c(x)


Class-dependent file-wise average
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:math:`c` represent the class and :math:`C` the ensemble of class the system can target.

:math:`n` is the size of the prediction from one file. It depends on the precision of the system.

.. math::
    \forall x \in S, \forall c \in C, Th_{xc} = \frac{1}{n} \sum p_c(x)


Median
------
.. warning:: The documentation here need to be change from here. Is it really necessary to write down the detail of the calculation of the median ?

By directly using the system temporal predictions, the median is computed and used as a segmentation threshold.
Depending on the methods used, the calculation of the average can be done either at the file or dataset level.
Moreover, it can also be independent or dependent on the class.

- :math:`S` represent the dataset
- :math:`p(x)` the temporal prediction output by the system
- :math:`N` the number of file in the dataset.

Class-dependent data-wise median
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The average is computed using the temporal prediction from the whole dataset and the classes.

.. math::
    \forall x \in S, Th=\frac{1}{N}\sum p(x)


Class-dependent file-wise median
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:math:`n` is the size of the prediction from one file. It depends on the precision of the system.

.. math::
    \forall x \in S, Th_x= \frac{1}{n} \sum p(x)

Class-independent data-wise median
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:math:`c` represent the class and :math:`C` the ensemble of class the system can target.

.. math::
    \forall x \in S, \forall c \in C, Th_c = \frac{1}{N} \sum p_c(x)


Class-independent file-wise median
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:math:`c` represent the class and :math:`C` the ensemble of class the system can target.

:math:`n` is the size of the prediction from one file. It depends on the precision of the system.

.. math::
    \forall x \in S, \forall c \in C, Th_{xc} = \frac{1}{n} \sum p_c(x)
