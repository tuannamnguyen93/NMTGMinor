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

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')

parser.add_argument('-src', required=True,
                    help='Source sequence to decode (one line per sequence)')


parser.add_argument('-concat', type=int, default=1,
                    help="Concate sequential audio features to decrease sequence length")

parser.add_argument('-stride', type=int, default=1,
                    help="Stride on input features")

parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img|audio].")

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

def write_array(fd, array):
    size = 0
    assert isinstance(array, np.ndarray), type(array)
    fd.write(b'\0B')
    size += 2
    dt = array.dtype
    if dt == np.float32 or dt == np.float16:
        atype = b'FM ' if dt == np.float32 else b'HM '
        if len(array.shape) == 2:
            fd.write(atype)
            size += 3
            fd.write(b'\4')
            size += 1
            fd.write(struct.pack('<i', len(array)))  # Rows
            size += 4

            fd.write(b'\4')
            size += 1
            fd.write(struct.pack('<i', array.shape[1]))  # Cols
            size += 4
        fd.write(array.tobytes())
        size += array.nbytes
    else:
        raise ValueError('Unsupported array type: {}'.format(dt))
    return size

def write_ark(ark, dic, scp=None, append=False):
    # Write ark
    mode = 'ab' if append else 'wb'
    pos_list = []
    with open(ark, mode) as fd:
        pos = fd.tell() if append else 0
        for key in dic:
            encode_key = (key + ' ').encode()
            fd.write(encode_key)
            pos += len(encode_key)
            pos_list.append(pos)
            data = dic[key]
            pos += write_array(fd, data)

    # Write scp
    if scp is not None:
        mode = 'a' if append else 'w'
        with open(scp, mode) as fd:
            for key, position in zip(dic, pos_list):
                fd.write(key + u' ' + ark + ':' + str(position) + os.linesep)
def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    # waveglow = waveglow.remove_weightnorm(waveglow)
    # waveglow.eval()

    model = opt.model
    checkpoint = torch.load(model,
                            map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']

    model = build_model(model_opt, checkpoint['dicts'])
    optimize_model(model)
    model.load_state_dict(checkpoint['model'])

    if opt.fp16:
        model = model.half()

    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    model.eval()
    if opt.encoder_type == "audio" and opt.asr_format == "scp":
        import kaldiio
        from kaldiio import ReadHelper
        audio_data = iter(ReadHelper('scp:' + opt.src))

    in_file = None

    output_ark = opt.output + "/output.ark"
    output_scp =  opt.output + "/output.scp"

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
                except StopIteration:
                    break

            if opt.stride != 1:
                line = line[0::opt.stride]
            line = torch.from_numpy(line)
            if opt.concat != 1:
                add = (opt.concat - line.size()[0] % opt.concat) % opt.concat
                z = torch.FloatTensor(add, line.size()[1]).zero_()
                line = torch.cat((line, z), 0)
                line = line.reshape((line.size()[0] // opt.concat, line.size()[1] * opt.concat))

            length, feature_size = line.size(0), line.size(1)

            tensor = torch.cat([torch.ones(length).unsqueeze(1),line],dim=-1)

            outputs = model.inference(tensor.unsqueeze(0).half().cuda())
            mel = outputs[0].squeeze(0).transpose(0,1).float().cpu().detach().numpy()
            print(mel.shape)
            dic = {seg_name: mel}
            write_ark(output_ark, dic, output_scp, append=True)


if __name__ == "__main__":
    main()