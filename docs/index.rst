##################
MLEM documentation
##################

| Paper: `Metric-Learning Encoding Models Identify Processing Profiles of Linguistic Features in BERT's Representations <https://arxiv.org/abs/2402.11608>`_
| Documentation for the github repository `MLEM <https://github.com/LouisJalouzot/MLEM>`_

Getting started
===============

Use a recent version of Python, figures in the paper were generated using Python 3.11. 

Dependencies
------------

.. code-block:: bash

   pip install -U -r requirements.txt
   git submodule update --init --recursive

Basic script
------------

Basic code block to run at the root of the repository to execute the pipeline with the default parameters:

.. code-block:: python

   from MLEM import *
   pipeline = MLEMPipeline()
   # Display the default parameters and saving directories
   print(pipeline)
   # Compute the features distance matrix
   pipeline.compute_features_distance()
   # Compute the Feature Importance for the 12 layers of `bert-base-uncased`
   for layer in tqdm(range(1, 13)):
      pipeline.layer = layer
      pipeline.compute_feature_importance()
   # Visualize the results
   ax = pipeline.plot_feature_importance()
   plt.show()

Walkthrough
-----------

A walkthrough notebook to use the MLEM pipeline and reproduce the figures from the paper is available at `walkthrough.ipynb <https://github.com/LouisJalouzot/MLEM/blob/master/walkthrough.ipynb>`_. Switch to the branch `precomputed <https://github.com/LouisJalouzot/MLEM/tree/precomputed>`_ if you want precomputed files.

.. toctree::
   :titlesonly:
   :caption: API

   MLEM/MLEM.rst
   MLEM/plotting_class.rst
   MLEM/regression_class.rst
   MLEM/distance_class.rst
   MLEM/recording_class.rst
   MLEM/base_class.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`