.. image:: https://travis-ci.org/MacHu-GWU/elementary_math-project.svg?branch=master

.. image:: https://img.shields.io/pypi/v/retrieveStack.svg

.. image:: https://img.shields.io/pypi/l/retrieveStack.svg

.. image:: https://img.shields.io/pypi/pyversions/retrieveStack.svg


Welcome to retrieveStack Documentation
===============================================================================
retrieveStack is an analytical framework by resembling a feedforward neural network and using stacked generalization in multiple levels to improve accuracy in classification (or regression) problems. 
In contrast to conventional feedforward neural-network-like stacked structures, retrieveStack gathers the information of models in all layers, choose good models from multiple layers by comparing to output layer, and combine them with output layer model to optimize final prediction. Below is the pictorial description of how retriveStack working:


.. image:: https://github.com/Robin888/retrieveStack-project/blob/master/retrieveStack.jpg

retrieveStack gives better prediction than the best single model contains in first layer. Please note its performance relies on a mix of strong and diverse single models in order to get the best out of this analytical framework


**Quick Links**
-------------------------------------------------------------------------------
- `GitHub Homepage <https://github.com/Robin888/retrieveStack-project>`_
- `Online Documentation <https://github.com/Robin888/retrieveStack-project/tree/master/documentation>`_
- `PyPI download <https://pypi.python.org/pypi/retrieveStack>`_
- `Install <install_>`_
- `Issue submit and feature request <https://github.com/Robin888/retrieveStack-project/issues>`_
- `API reference and source code <https://github.com/Robin888/retrieveStack-project/blob/master/documentation/Functions%20and%20Parameters.pdf>`_
- `Tutorial <https://github.com/Robin888/retrieveStack-project/blob/master/documentation/Tutorial.ipynb>`_


.. _install:

Install
-------------------------------------------------------------------------------

``retrieveStack`` is released on PyPI, so all you need is:

.. code-block:: console

	$ pip install retrieveStack

To upgrade to latest version:

.. code-block:: console

	$ pip install --upgrade retrieveStack
