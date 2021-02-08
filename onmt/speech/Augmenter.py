import math
import torch
from collections import defaultdict
import onmt
import random
import numpy as np

class Augmenter(object):
    """
    Implementation of the "Spec Augmentation" method
    (Only vertical and horizontal masking)
    """

    def __init__(self, F=12, mf=2, T=64, max_t=0.25, mt=2,
                 input_size=40, concat=4, time_stretch=True):

        self.F = F
        self.mf = mf
        self.T = T
        self.max_t = max_t
        self.mt = mt
        self.input_size = input_size
        self.concat = concat
        self.time_stretch = time_stretch

    def augment(self, tensor):

        tensor = tensor.float()
        #return tensor
        if self.time_stretch:

            time_len = tensor.size(0)
            ids = None

            low = 0.8
            high = 1.25
            win = 10000

            for i in range((time_len // win) + 1):

                s = random.uniform(low, high)
                e = min(time_len, win * (i + 1))
                r = np.arange(win * i, e - 1, s, dtype=np.float32)
                r = np.round(r).astype(np.int32)
                ids = r if ids is None else np.concatenate((ids, r))
                # print(ids.shape[0])
            tensor = tensor[ids]


        feat_size = tensor.size(1)
        original_len = tensor.size(0)
        reshape_size = feat_size / self.input_size


        # First we have to upsample the tensor (if it was downsampled during preprocessing)
        #         # Copy to a new storage because otherwise it is zeroed permanently`
        tensor_ = tensor.view(-1, self.input_size).new(*tensor.size()).copy_(tensor)



        for _ in range(self.mf):

            # frequency masking (second dimension)
            # 40 is the number of features (logmel)
            f = int(random.uniform(0.0, self.F))
            f_0 = int(random.uniform(0.0, 40 - f))

            tensor_[:, f_0:f_0 + f].zero_()

        for _ in range(self.mt):
            # time masking (first dimension)
            t = int(random.uniform(0.0, self.T))

            t = min(t, int(self.max_t * original_len))

            t_0 = int(random.uniform(0.0, original_len - t - 1))

            tensor_[t_0: t_0 + t].zero_()

        # reshaping back to downsampling
        tensor__ = tensor_.view(original_len, feat_size)

 #       print("spec_aug  "+ str(time.time() -start))

        return tensor__




