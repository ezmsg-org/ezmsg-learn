import numpy as np
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.sigproc.sampler import SampleMessage, SampleTriggerMessage

from ezmsg.learn.process.sgd import sgd_decoder


def test_sgd():
    # Prepare training samples
    samples = []
    time_idx = {"A": 0, "B": 1, "C": 2}
    for label in ["A", "B", "C"] * 5:
        data = np.random.normal(scale=0.05, size=(3, 2, 1))
        data[time_idx[label] : time_idx[label] + 1, 0, 0] += 1.0
        samples.append(
            SampleMessage(
                trigger=SampleTriggerMessage(
                    timestamp=len(samples), period=None, value=label
                ),
                sample=AxisArray(data=data, dims=["time", "ch", "freq"]),
            )
        )

    # Prepare AxisArray messages for inference. Each message has 2 windows.
    windows = []
    for win_ix in range(30):
        label = ["A", "B", "C"][win_ix % 3]
        data = np.random.normal(scale=0.05, size=(2, 3, 2, 1))
        data[:, time_idx[label] : time_idx[label] + 1, 0, 0] += 1.0
        windows.append(AxisArray(data=data, dims=["win", "time", "ch", "freq"]))

    """
    train_X = np.stack([_.sample.data for _ in samples], axis=0)
    train_Y = np.stack([_.trigger.value for _ in samples])
    test_X = AxisArray.concatenate(*windows, dim="win")
    model = SGDClassifier(
        loss="squared_hinge",
        alpha=1e-3,
        penalty="elasticnet",
        learning_rate="adaptive",
        eta0=0.01,
        early_stopping=False,
    )
    model.partial_fit(
        train_X.reshape(train_X.shape[0], -1), train_Y, classes=["A", "B", "C"]
    )
    model.predict(test_X.data.reshape(test_X.data.shape[0], -1))
    """
    label_weights = {k: 1.0 for k in time_idx.keys()}
    # Sending an axis array before it has seen any training samples should yield None
    gen = sgd_decoder(alpha=1e-3, loss="squared_hinge", label_weights=label_weights)
    assert gen.send(windows[0]) is None

    # Now let's try training on all samples
    for sample in samples:
        gen.send(sample)
    # Then doing inference on all multi-wins
    probas = [gen.send(win) for win in windows]

    # With this easy-to-classify data, accuracy should be 100%
    # when we fit all training before predicting any test.
    class_ids = []
    for cm in probas:
        class_ids.extend(np.argmax(cm.data, axis=1).tolist())
    expected_ids = [0, 0, 1, 1, 2, 2] * 10
    assert np.array_equal(class_ids, expected_ids)

    # Try again (new model) but alternate 1 train, 2 test.
    gen = sgd_decoder(alpha=1e-3, loss="squared_hinge", label_weights=label_weights)
    probas = []
    for samp_ix, samp in enumerate(samples):
        gen.send(samp)
        probas.append(gen.send(windows[samp_ix * 2]))
        probas.append(gen.send(windows[samp_ix * 2 + 1]))
    class_ids = []
    for cm in probas:
        class_ids.extend(np.argmax(cm.data, axis=1).tolist())
    # We expect early predictions to be bad but it should improve at the end.
    b_correct = [x == p for x, p in zip(expected_ids, class_ids)]
    assert not np.all(b_correct[:30])
    assert np.all(b_correct[30:])
