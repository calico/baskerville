# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#         https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function
import pdb

import numpy as np
import tensorflow as tf

from baskerville import dna


class PredStreamGen:
    """Interface to acquire predictions via a buffered stream mechanism
    rather than getting them all at once and using excessive memory.
    Accepts generator and constructs stream batches from it."""

    def __init__(self, model, seqs_gen, batch_size, stream_seqs=32, verbose=False):
        self.model = model
        self.seqs_gen = seqs_gen
        self.stream_seqs = stream_seqs
        self.batch_size = batch_size
        self.verbose = verbose

        self.stream_start = 0
        self.stream_end = 0
        self.stream_preds = []

    def __getitem__(self, i):
        # acquire predictions, if needed
        if i >= self.stream_end:
            # update start
            self.stream_start = self.stream_end

            # predict
            del self.stream_preds
            self.stream_preds = self.model.predict(self.make_dataset())

            # update end
            self.stream_end = self.stream_start + self.stream_preds.shape[0]

            if self.verbose:
                print(
                    "Predicting %d-%d" % (self.stream_start, self.stream_end),
                    flush=True,
                )

        return self.stream_preds[i - self.stream_start]

    def make_dataset(self):
        """Construct Dataset object for this stream chunk."""
        seqs_1hot = []
        stream_end = self.stream_start + self.stream_seqs
        for si in range(self.stream_start, stream_end):
            try:
                seqs_1hot.append(self.seqs_gen.__next__())
            except StopIteration:
                continue

        seqs_1hot = np.array(seqs_1hot)

        dataset = tf.data.Dataset.from_tensor_slices((seqs_1hot,))
        dataset = dataset.batch(self.batch_size)
        return dataset
