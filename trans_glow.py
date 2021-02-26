#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import struct
import onmt
import onmt.markdown
import torch
import argparse
import math
import numpy
import sys
import h5py as h5
import numpy as np
import apex
from scipy.io.wavfile import write
from onmt.model_factory import build_model, build_language_model, optimize_model
from onmt.inference.fast_translator import FastTranslator
from onmt.inference.stream_translator import StreamTranslator
import os

parser = argparse.ArgumentParser(description='translate.py')
onmt.markdown.add_md_help_argument(parser)

parser.add_argument('-src', required=True,
                    help='Source sequence to decode (one line per sequence)')

parser.add_argument('-encoder_type', default='audio',
                    help="Type of encoder to use. Options are [audio].")

parser.add_argument('-output',
                    help="Path to output")

parser.add_argument('-fp16', action='store_true',
                    help='To use floating point 16 in decoding')

parser.add_argument('-asr_format', default="scp", required=False,
                    help="Format of asr data h5 or scp")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")

parser.add_argument('-batch_size', type=int, default=1,
                    help='Batch size, currently support batch size 1')


def main():
    opt = parser.parse_args()

    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()

    if opt.encoder_type == "audio" and opt.asr_format == "scp":
        import kaldiio
        from kaldiio import ReadHelper
        audio_data = iter(ReadHelper('scp:' + opt.src))

    in_file = None

    output_dir = opt.output

    if opt.encoder_type == "audio":

        i = 0
        while True:
            if opt.asr_format == "h5":
                if i == len(in_file):
                    break
                line = np.array(in_file[str(i)])
                i += 1
            elif opt.asr_format == "scp":
                try:
                    seg_name, line = next(audio_data)
                    i += 1
                except StopIteration:
                    break

            print(line.shape)
            line = torch.from_numpy(line)

            mel = line[:500, :].transpose(0, 1).unsqueeze(0).cuda()
            # print(outputs[0].shape)
            audio = waveglow.infer(mel)
            audio_numpy = audio[0].data.cpu().numpy()
            rate = 22050
            write("%s/audio_%s.wav" % (output_dir, seg_name), rate, audio_numpy)
            if i > 10:
                break

    #       dic = {seg_name: mel}
    #        write_ark(output_ark, dic, output_scp, append=True)


if __name__ == "__main__":
    main()
