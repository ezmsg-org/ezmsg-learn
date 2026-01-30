Real-Time Classification
========================

This guide shows how to use ezmsg-learn for real-time classification in streaming pipelines.

.. contents:: On this page
   :local:
   :depth: 2


Overview
--------

ezmsg-learn provides machine learning components that integrate with ezmsg pipelines.
Key features include:

- **Pre-trained models**: Load and apply existing classifiers
- **Online learning**: Update models incrementally with streaming data
- **Flexible backends**: Support for scikit-learn, PyTorch, and River models


Available Classifiers
---------------------

ezmsg-learn includes several classifier types:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Classifier
     - Description
     - Use Case
   * - ``SLDA``
     - Shrinkage Linear Discriminant Analysis
     - BCI, small datasets
   * - ``SklearnModelUnit``
     - Wrapper for any scikit-learn model
     - General ML tasks
   * - ``SGDClassifier``
     - Stochastic Gradient Descent
     - Online learning
   * - ``MLPUnit``
     - Multi-layer Perceptron (PyTorch)
     - Complex patterns


Using a Pre-Trained SLDA Classifier
-----------------------------------

The simplest approach is to use a pre-trained model:

.. code-block:: python

   from ezmsg.learn.process.slda import SLDA, SLDASettings

   classifier = SLDA(
       SLDASettings(
           settings_path="path/to/trained_model.pkl",
           axis="time",  # Axis containing samples
       )
   )

**Input format**: ``AxisArray[time, features]`` where features are flattened from your pipeline.

**Output format**: ``ClassifierMessage[time, classes]`` with class probabilities.

Training an SLDA model (offline):

.. code-block:: python

   import pickle
   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

   # Train offline with your data
   X_train = ...  # shape: (n_samples, n_features)
   y_train = ...  # shape: (n_samples,)

   lda = LDA(solver="lsqr", shrinkage="auto")
   lda.fit(X_train, y_train)

   # Save for use in ezmsg
   with open("trained_model.pkl", "wb") as f:
       pickle.dump(lda, f)


Using Scikit-Learn Models
-------------------------

``SklearnModelUnit`` wraps any scikit-learn compatible model:

.. code-block:: python

   from ezmsg.learn.process.sklearn import SklearnModelUnit, SklearnModelSettings
   import numpy as np

   classifier = SklearnModelUnit(
       SklearnModelSettings(
           model_class="sklearn.linear_model.SGDClassifier",
           model_kwargs={
               "loss": "log_loss",  # For probability outputs
               "warm_start": True,
           },
           partial_fit_classes=np.array([0, 1]),  # Required for online learning
       )
   )

Loading a pre-trained model:

.. code-block:: python

   classifier = SklearnModelUnit(
       SklearnModelSettings(
           model_class="sklearn.linear_model.SGDClassifier",
           checkpoint_path="path/to/saved_model.pkl",
       )
   )


Online Learning
---------------

For models that support ``partial_fit``, you can update them during streaming:

.. code-block:: python

   from ezmsg.learn.process.sklearn import SklearnModelProcessor, SklearnModelSettings
   from ezmsg.sigproc.sampler import SampleMessage

   # Create processor with online learning support
   processor = SklearnModelProcessor(
       settings=SklearnModelSettings(
           model_class="sklearn.linear_model.SGDClassifier",
           model_kwargs={"loss": "log_loss"},
           partial_fit_classes=np.array([0, 1]),
       )
   )

   # Training with labeled samples
   sample_msg = SampleMessage(
       sample=feature_array,  # AxisArray with features
       trigger=label_value,   # The class label
   )
   processor.partial_fit(sample_msg)

   # Prediction (after training)
   prediction = processor(input_features)


Complete Pipeline Example
-------------------------

Here's a complete BCI classification pipeline:

.. code-block:: python

   import ezmsg.core as ez
   from ezmsg.lsl.inlet import LSLInletUnit, LSLInletSettings, LSLInfo
   from ezmsg.lsl.outlet import LSLOutletUnit, LSLOutletSettings
   from ezmsg.sigproc.butterworthfilter import ButterworthFilter, ButterworthFilterSettings
   from ezmsg.sigproc.window import Window, WindowSettings
   from ezmsg.sigproc.spectrum import Spectrum, SpectrumSettings
   from ezmsg.sigproc.aggregate import RangedAggregate, RangedAggregateSettings, AggregationFunction
   from ezmsg.learn.process.slda import SLDA, SLDASettings

   components = {
       # Data acquisition
       "LSL_IN": LSLInletUnit(
           LSLInletSettings(info=LSLInfo(name="EEG", type="EEG"))
       ),

       # Signal processing
       "FILTER": ButterworthFilter(
           ButterworthFilterSettings(order=4, cuton=8.0, cutoff=30.0)
       ),
       "WINDOW": Window(
           WindowSettings(window_dur=1.0, window_shift=0.5)
       ),
       "SPECTRUM": Spectrum(SpectrumSettings(window="hann")),
       "BANDPOWER": RangedAggregate(
           RangedAggregateSettings(
               axis="freq",
               bands=[(8.0, 12.0), (18.0, 25.0)],
               operation=AggregationFunction.MEAN,
           )
       ),

       # Classification
       "CLASSIFIER": SLDA(
           SLDASettings(settings_path="model.pkl", axis="time")
       ),

       # Output
       "LSL_OUT": LSLOutletUnit(
           LSLOutletSettings(stream_name="Predictions", stream_type="Markers")
       ),
   }

   connections = (
       (components["LSL_IN"].OUTPUT_SIGNAL, components["FILTER"].INPUT_SIGNAL),
       (components["FILTER"].OUTPUT_SIGNAL, components["WINDOW"].INPUT_SIGNAL),
       (components["WINDOW"].OUTPUT_SIGNAL, components["SPECTRUM"].INPUT_SIGNAL),
       (components["SPECTRUM"].OUTPUT_SIGNAL, components["BANDPOWER"].INPUT_SIGNAL),
       (components["BANDPOWER"].OUTPUT_SIGNAL, components["CLASSIFIER"].INPUT_SIGNAL),
       (components["CLASSIFIER"].OUTPUT_SIGNAL, components["LSL_OUT"].INPUT_SIGNAL),
   )

   if __name__ == "__main__":
       ez.run(components=components, connections=connections)


Feature Preparation
-------------------

Classifiers expect flattened 2D input ``[samples, features]``. Multi-dimensional arrays
are automatically flattened along the channel dimension.

For example, if your bandpower output is ``[time=1, band=2, ch=8]``:

- The classifier receives shape ``[1, 16]`` (2 bands × 8 channels)
- Features are flattened in C-order (row-major)


Output Format
-------------

Classification outputs use ``ClassifierMessage``, which extends ``AxisArray`` with:

- **dims**: ``["time", "classes"]``
- **data**: Probability scores for each class
- **labels**: List of class names/identifiers

Example output shape: ``[time=1, classes=2]`` with probabilities for each class.


Tips for Better Performance
---------------------------

1. **Normalize features**: Use ``Scaler`` from ezmsg-sigproc before classification

   .. code-block:: python

      from ezmsg.sigproc.scaler import Scaler, ScalerSettings
      scaler = Scaler(ScalerSettings(mode="zscore"))

2. **Match training conditions**: Ensure online features match offline training preprocessing

3. **Window size**: Larger windows give more stable features but higher latency

4. **Feature selection**: Start with relevant frequency bands for your application


Troubleshooting
---------------

**"Model has not been fit yet"**:
   The model needs training data before prediction. Either:
   - Provide a ``checkpoint_path`` with a pre-trained model
   - Call ``fit()`` or ``partial_fit()`` before processing

**Shape mismatch errors**:
   - Verify input feature dimensions match trained model
   - Check ``n_features_in_`` attribute of loaded models

**NaN in predictions**:
   - Ensure input features don't contain NaN values
   - Check for numerical stability in preprocessing
