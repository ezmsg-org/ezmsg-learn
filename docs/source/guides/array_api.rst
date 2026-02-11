Array API Compatibility
=======================

ezmsg-learn uses the `Array API standard <https://data-apis.org/array-api/latest/>`_
to allow processors to operate on arrays from different backends — NumPy, CuPy,
PyTorch, and others — without code changes.

.. contents:: On this page
   :local:
   :depth: 2


How It Works
------------

Modules that support the Array API derive the array namespace from their input
data using ``array_api_compat.get_namespace()``:

.. code-block:: python

   from array_api_compat import get_namespace

   def process(self, data):
       xp = get_namespace(data)       # numpy, cupy, torch, etc.
       result = xp.linalg.inv(data)   # dispatches to the right backend
       return result

This means that if you pass a CuPy array, all computation stays on the GPU.
If you pass a NumPy array, it behaves exactly as before.

Helper utilities from ``ezmsg.sigproc.util.array`` handle device placement
and creation functions portably:

- ``array_device(x)`` — returns the device of an array, or ``None``
- ``xp_create(fn, *args, dtype=None, device=None)`` — calls creation
  functions (``zeros``, ``eye``) with optional device
- ``xp_asarray(xp, obj, dtype=None, device=None)`` — portable ``asarray``


Module Compatibility
--------------------

The table below summarises the Array API status of each module.

Fully compatible
^^^^^^^^^^^^^^^^

These modules perform all computation in the source array namespace.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Notes
   * - ``process.ssr``
     - LRR / self-supervised regression. Full Array API.
   * - ``model.cca``
     - Incremental CCA. Replaced ``scipy.linalg.sqrtm`` with an
       eigendecomposition-based inverse square root using only Array API ops.
   * - ``process.rnn``
     - PyTorch-native; operates on ``torch.Tensor`` throughout.

Mostly compatible (with NumPy boundaries)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These modules use the Array API for data manipulation but fall back to NumPy
at specific points where a dependency requires it.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Module
     - NumPy boundary
     - Reason
   * - ``model.refit_kalman``
     - ``_compute_gain()``
     - ``scipy.linalg.solve_discrete_are`` has no Array API equivalent.
       Matrices are converted to NumPy for the DARE solver, then converted back.
   * - ``model.refit_kalman``
     - ``refit()`` mutation loop
     - Per-sample velocity remapping uses ``np.linalg.norm`` on small vectors
       and scalar element assignment.
   * - ``process.refit_kalman``
     - Inherits boundaries from model
     - State init and output arrays use the source namespace.
   * - ``process.slda``
     - ``predict_proba``
     - sklearn ``LinearDiscriminantAnalysis`` requires NumPy input.
   * - ``process.adaptive_linear_regressor``
     - ``partial_fit`` / ``predict``
     - sklearn and river models require NumPy / pandas input.
   * - ``dim_reduce.adaptive_decomp``
     - ``partial_fit`` / ``transform``
     - sklearn ``IncrementalPCA`` and ``MiniBatchNMF`` require NumPy input.

Not converted
^^^^^^^^^^^^^

These modules use NumPy directly. Conversion would provide little benefit
because the underlying estimator is the bottleneck.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Reason
   * - ``process.linear_regressor``
     - Thin wrapper around sklearn ``LinearModel.predict``.
       Could be made compatible if sklearn's ``array_api_dispatch`` is enabled
       (see below).
   * - ``process.sgd``
     - sklearn ``SGDClassifier`` has no Array API support.
   * - ``process.sklearn``
     - Generic wrapper for arbitrary models; cannot assume Array API support.
   * - ``dim_reduce.incremental_decomp``
     - Delegates to ``adaptive_decomp``; trivial numpy usage (``np.prod`` on
       Python tuples).


sklearn Array API Dispatch
--------------------------

scikit-learn 1.8+ has experimental support for Array API dispatch on a subset
of estimators.  Two estimators used in ezmsg-learn are on the supported list:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Estimator
     - Used in
     - Constraint
   * - ``LinearDiscriminantAnalysis``
     - ``process.slda``
     - Requires ``solver="svd"`` (the ``"lsqr"`` solver with ``shrinkage``
       is not supported)
   * - ``Ridge``
     - ``process.linear_regressor``
     - Requires ``solver="svd"``

To use dispatch, enable it before creating the estimator:

.. code-block:: python

   from sklearn import set_config
   set_config(array_api_dispatch=True)

.. warning::

   - ``array_api_dispatch`` is marked **experimental** in sklearn.
   - Solver constraints (``solver="svd"``) may produce slightly different
     numerical results compared to other solvers.
   - Enabling dispatch globally may affect other sklearn estimators in the
     same process.
   - ezmsg-learn does **not** enable dispatch by default.

Estimators that do **not** support Array API dispatch:

- ``IncrementalPCA``, ``MiniBatchNMF`` — only batch ``PCA`` is supported
- ``SGDClassifier``, ``SGDRegressor``, ``PassiveAggressiveRegressor``
- All river models


Writing Array API Compatible Code
----------------------------------

When adding or modifying processors in ezmsg-learn, follow these patterns.

Deriving the namespace
^^^^^^^^^^^^^^^^^^^^^^

Always derive ``xp`` from the input data, not from a hardcoded ``numpy``:

.. code-block:: python

   from array_api_compat import get_namespace
   from ezmsg.sigproc.util.array import array_device, xp_create

   def _process(self, message):
       xp = get_namespace(message.data)
       dev = array_device(message.data)

Transposing matrices
^^^^^^^^^^^^^^^^^^^^

The Array API does not support ``.T``.  Use ``xp.linalg.matrix_transpose()``:

.. code-block:: python

   # Before (numpy-only)
   result = A.T @ B

   # After (Array API)
   _mT = xp.linalg.matrix_transpose
   result = _mT(A) @ B

Creating arrays
^^^^^^^^^^^^^^^

Use ``xp_create`` to handle device placement portably:

.. code-block:: python

   # Before
   I = np.eye(n)
   z = np.zeros((m, n), dtype=np.float64)

   # After
   I = xp_create(xp.eye, n, device=dev)
   z = xp_create(xp.zeros, (m, n), dtype=xp.float64, device=dev)

Handling sklearn boundaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When calling into sklearn (or other NumPy-only libraries), convert at the
boundary and convert back:

.. code-block:: python

   from array_api_compat import is_numpy_array

   # Convert to numpy for sklearn
   X_np = np.asarray(X) if not is_numpy_array(X) else X
   result_np = estimator.predict(X_np)

   # Convert back to source namespace
   result = xp.asarray(result_np) if not is_numpy_array(X) else result_np

Checking for NaN
^^^^^^^^^^^^^^^^

Use ``xp.isnan`` instead of ``np.isnan``:

.. code-block:: python

   if xp.any(xp.isnan(message.data)):
       return

Norms
^^^^^

Use ``xp.linalg.matrix_norm`` (Frobenius by default) instead of
``np.linalg.norm`` for matrices.  For vectors, use ``xp.linalg.vector_norm``.
