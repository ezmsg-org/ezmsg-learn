from ezmsg.learn.collection.sample_adapt_regressor import (
    SampleAdaptRegressor,
    SampleAdaptRegressorSettings,
)


def test_sample_adapt_regressor_uses_windowed_decode_branch_when_configured():
    collection = SampleAdaptRegressor(
        settings=SampleAdaptRegressorSettings(
            decode_window_dur=0.2,
            decode_window_shift=0.01,
        )
    )
    collection.configure()

    network = collection.network()

    assert (collection.INPUT_SIGNAL, collection.WINDOW.INPUT_SIGNAL) in network
    assert (collection.WINDOW.OUTPUT_SIGNAL, collection.FLATTEN.INPUT_SIGNAL) in network
    assert (collection.FLATTEN.OUTPUT_SIGNAL, collection.REGRESSOR.INPUT_SIGNAL) in network
    assert (collection.INPUT_SIGNAL, collection.REGRESSOR.INPUT_SIGNAL) not in network
