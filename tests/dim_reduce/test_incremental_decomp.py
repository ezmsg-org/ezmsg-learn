# from dataclasses import replace
#
# import pytest
# import numpy as np
# from sklearn.decomposition import IncrementalPCA, MiniBatchNMF
# from ezmsg.util.messages.axisarray import AxisArray
#
# from ezmsg.learn.dim_reduce.incremental_decomp import IncrementalDecompTransformer
#
#
# @pytest.fixture
# def sample_data():
#     n_times = 1000
#     n_ch = 32
#     data = np.arange(n_times * n_ch).reshape(n_times, n_ch)
#     return data
#
#
# @pytest.mark.parametrize("update_interval", [0.0, 1.0, 0.5])
# @pytest.mark.parametrize("method", ["nmf", "pca"])
# def test_incremental_decomp(sample_data, update_interval: float, method: str):
#     n_components = 3
#     fs = 128.0
#     chunk_dur = 0.125
#     n_times, n_ch = sample_data.shape
#
#     # Calculate using the transformer
#     template = AxisArray(
#         data=np.array([[]]),
#         dims=["time", "ch"],
#         axes={
#             "time": AxisArray.TimeAxis(fs=fs, offset=0),
#             "ch": AxisArray.CoordinateAxis(
#                 data=np.arange(n_ch).astype(str), dims=["ch"]
#             ),
#         },
#         key="test_incremental_pca",
#     )
#     ezdecomp = IncrementalDecompTransformer(
#         axis="!time",
#         n_components=n_components,
#         update_interval=update_interval,
#         method=method,
#     )
#     chunk_size = int(chunk_dur * fs)
#     n_chunks = int(n_times // chunk_size)
#     if n_times % chunk_size:
#         n_chunks += 1
#     res = []
#     for msg_ix in range(n_chunks):
#         msg_in = replace(
#             template,
#             data=sample_data[msg_ix * chunk_size : (msg_ix + 1) * chunk_size],
#             axes={
#                 **template.axes,
#                 "time": replace(template.axes["time"], offset=msg_ix * chunk_size / fs),
#             },
#         )
#         res.append(ezdecomp(msg_in))
#     assert len(res) == n_chunks
#     cat = AxisArray.concatenate(*res, dim="time")
#
#     # Calculate Expected
#     if method == "nmf":
#         decomp = MiniBatchNMF(n_components=n_components)
#     else:
#         decomp = IncrementalPCA(n_components=n_components)
#
#     # Prepare data chunks
#     chunks = [
#         _
#         for _ in sample_data[: -(sample_data.shape[0] % chunk_size)].reshape(
#             -1, chunk_size, n_ch
#         )
#     ]
#     if sample_data.shape[0] % chunk_size:
#         chunks.append(sample_data[-(sample_data.shape[0] % chunk_size) :])
#
#     # The first iteration always uses chunk_size data.
#     decomp.partial_fit(chunks[0])
#     expected = [decomp.transform(chunks[0])]
#
#     # If update_interval is 0, then each chunk trains the model.
#     # If update_interval is not 0, then each chunk is buffered, and the model is updated if the buffer passes update_interval.
#     buff_size = int(update_interval * fs) if update_interval > 0 else chunk_size
#     buffer = None
#     for chunk_ix, chunk in enumerate(chunks[1:]):
#         if buffer is None:
#             buffer = chunk
#         else:
#             buffer = np.concatenate((buffer, chunk), axis=0)
#         if buffer.shape[0] >= buff_size or (
#             update_interval == 0 and chunk_ix == len(chunks) - 2
#         ):
#             decomp.partial_fit(buffer[:buff_size])
#             buffer = buffer[buff_size:]
#         expected.append(decomp.transform(chunk))
#     expected = np.concatenate(expected, axis=0)
#
#     assert cat.shape == expected.shape
#     assert np.allclose(cat.data, expected)
