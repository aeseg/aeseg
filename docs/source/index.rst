.. aeseg documentation master file, created by
   sphinx-quickstart on Fri Jul 26 14:56:30 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*aeseg* is a python package for post-processing and optimization in Sound Event Detection task.

AESEG is a python library compiling an ensemble of post-processing and optimization methods for the
segmentation of audio event. I initially developed it for the `DCASE 2018 task four challenge <http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection-results>`_.

It is typically used for the post-processing step of a Sound Event Detection system that output
temporal probabilities that need to be cut into positive and negative segment.

Post processing methods
=======================

.. toctree::
   :maxdepth: 2

   statistics
   parametrics


Optimization algorithms
=======================
.. toctree::
   :maxdepth: 2

   optimization

Source
======
.. toctree::
   :maxdepth: 2

   source



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
