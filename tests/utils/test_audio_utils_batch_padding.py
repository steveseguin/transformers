# coding=utf-8
# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

from transformers.audio_utils import mel_filter_bank, spectrogram, spectrogram_batch, window_function


class SpectrogramBatchPaddingTest(unittest.TestCase):
    def test_db_range_ignores_batch_padding(self):
        # The valid frames are silent after DC removal, but the zero-padded tail
        # contains a discontinuity. It must not change any retained frame.
        waveforms = [np.ones(800, dtype=np.float32), np.ones(3200, dtype=np.float32)]
        window = window_function(400)
        filters = mel_filter_bank(201, 20, 0.0, 8000.0, 16000)

        for power in (1.0, 2.0):
            for center in (False, True):
                for mel_filters in (None, filters):
                    with self.subTest(power=power, center=center, mel=mel_filters is not None):
                        kwargs = dict(
                            window=window,
                            frame_length=400,
                            hop_length=160,
                            power=power,
                            center=center,
                            remove_dc_offset=True,
                            mel_filters=mel_filters,
                            log_mel="dB",
                            db_range=80.0,
                        )
                        expected = [spectrogram(waveform, **kwargs) for waveform in waveforms]
                        for single in expected:
                            np.testing.assert_array_equal(single, -200.0 if power == 1.0 else -100.0)
                        actual = spectrogram_batch(waveforms, **kwargs)
                        reversed_actual = spectrogram_batch(waveforms[::-1], **kwargs)[::-1]
                        for single, batched, reordered in zip(expected, actual, reversed_actual):
                            self.assertEqual(batched.shape, single.shape)
                            self.assertEqual(batched.dtype, single.dtype)
                            np.testing.assert_array_equal(batched, single)
                            np.testing.assert_array_equal(reordered, single)

    def test_db_range_preserves_real_peaks(self):
        waveforms = [np.zeros(800, dtype=np.float32), np.zeros(3200, dtype=np.float32)]
        waveforms[0][200] = 1.0
        waveforms[1][1600] = 100.0

        for power in (1.0, 2.0):
            with self.subTest(power=power):
                kwargs = dict(
                    window=window_function(400),
                    frame_length=400,
                    hop_length=160,
                    power=power,
                    log_mel="dB",
                    reference=2.0,
                    min_value=1e-8,
                    db_range=20.0,
                )
                actual = spectrogram_batch(waveforms, **kwargs)
                for waveform, batched in zip(waveforms, actual):
                    single = spectrogram(waveform, **kwargs)
                    np.testing.assert_array_equal(batched, single)
                    self.assertAlmostEqual(float(batched.max() - batched.min()), 20.0, places=5)

    def test_unclipped_and_non_db_outputs_are_unchanged(self):
        waveforms = [np.ones(800, dtype=np.float32), np.ones(1600, dtype=np.float32)]

        for log_mel in (None, "log", "log10", "dB"):
            with self.subTest(log_mel=log_mel):
                kwargs = dict(
                    window=window_function(400),
                    frame_length=400,
                    hop_length=160,
                    power=2.0,
                    mel_filters=mel_filter_bank(201, 20, 0.0, 8000.0, 16000),
                    log_mel=log_mel,
                )
                actual = spectrogram_batch(waveforms, **kwargs)
                for waveform, batched in zip(waveforms, actual):
                    np.testing.assert_allclose(batched, spectrogram(waveform, **kwargs), rtol=1e-6, atol=1e-6)
