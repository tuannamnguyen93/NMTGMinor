from __future__ import division

import datetime
import gc
import inspect
import math
import os
import re
import time
import torch
import sys
from apex import amp
import random

import onmt
import onmt.markdown
import onmt.modules
from onmt.data.data_iterator import DataIterator
from onmt.data.multidata_iterator import MultiDataIterator
from onmt.data.dataset import rewrap
from onmt.model_factory import build_model, build_language_model, optimize_model
from onmt.model_factory import init_model_parameters
from onmt.train_utils.stats import Logger
from onmt.utils import checkpoint_paths, normalize_gradients


def flip_attributes(attributes, n_cat):
    if n_cat == 2:
        attributes = 1 - attributes

    else:

        shift = torch.LongTensor(attributes.size()).random_(n_cat - 1) + 1
        attributes = (attributes + shift.to(attributes.device)) % n_cat

    return attributes


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def get_lambda(lambda_lat_dis, step):
    """
    Compute discriminators' lambdas.
    """
    s = 50000

    return lambda_lat_dis * float(min(step, s)) / s


def generate_data_iterator(dataset, seed, num_workers=1, epoch=1., buffer_size=0):
    # check if dataset is a list:
    if isinstance(dataset, list):
        # this is a multidataset

        data_iterator = MultiDataIterator(dataset, seed=seed, num_workers=num_workers,
                                          epoch=epoch, buffer_size=buffer_size)
    else:

        data_iterator = DataIterator(dataset, dataset.collater, dataset.batches, seed=seed,
                                     num_workers=num_workers, epoch=epoch, buffer_size=buffer_size)

    return data_iterator


class BaseTrainer(object):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt):

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data

        self.dicts = dicts
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1 and opt.gpus[0] >= 0)

        self.loss_function = loss_function
        self.start_time = 0

    def run(self, *args, **kwargs):

        raise NotImplementedError

    def eval(self, data):

        raise NotImplementedError

    def load_encoder_weight(self, checkpoint_file):

        print("Loading pretrained models from %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        pretrained_model = build_model(checkpoint['opt'], checkpoint['dicts'])
        pretrained_model.load_state_dict(checkpoint['model'])

        print("Loading pretrained encoder weights ...")
        pretrained_model.encoder.language_embedding = None
        enc_language_embedding = self.model.encoder.language_embedding
        self.model.encoder.language_embedding = None
        encoder_state_dict = pretrained_model.encoder.state_dict()

        self.model.encoder.load_state_dict(encoder_state_dict)
        self.model.encoder.language_embedding = enc_language_embedding
        return

    def load_decoder_weight(self, checkpoint_file):

        print("Loading pretrained models from %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        chkpoint_dict = checkpoint['dicts']

        pretrained_model = build_model(checkpoint['opt'], chkpoint_dict)
        pretrained_model.load_state_dict(checkpoint['model'])

        print("Loading pretrained decoder weights ...")
        # first we have to remove the embeddings which probably have difference size ...
        pretrained_word_emb = pretrained_model.decoder.word_lut
        pretrained_model.decoder.word_lut = None
        pretrained_lang_emb = pretrained_model.decoder.language_embeddings
        pretrained_model.decoder.language_embeddings = None

        # actually we assume that two decoders have the same language embeddings...
        untrained_word_emb = self.model.decoder.word_lut
        self.model.decoder.word_lut = None
        untrained_lang_emb = self.model.decoder.language_embeddings
        self.model.decoder.language_embeddings = None

        decoder_state_dict = pretrained_model.decoder.state_dict()
        self.model.decoder.load_state_dict(decoder_state_dict)

        # now we load the embeddings ....
        n_copies = 0
        for token in self.dicts['tgt'].labelToIdx:

            untrained_id = self.dicts['tgt'].labelToIdx[token]

            if token in chkpoint_dict['tgt'].labelToIdx:
                pretrained_id = chkpoint_dict['tgt'].labelToIdx[token]
                untrained_word_emb.weight.data[untrained_id].copy_(pretrained_word_emb.weight.data[pretrained_id])

                self.model.generator[0].linear.bias.data[untrained_id].copy_(pretrained_model
                                                                             .generator[0].linear.bias.data[
                                                                                 pretrained_id])
                n_copies += 1

        print("Copied embedding for %d words" % n_copies)
        self.model.decoder.word_lut = untrained_word_emb

        # now we load the language embeddings ...
        if pretrained_lang_emb and untrained_lang_emb and 'langs' in chkpoint_dict:
            for lang in self.dicts['langs']:

                untrained_id = self.dicts['langs'][lang]
                if lang in chkpoint_dict['langs']:
                    pretrained_id = chkpoint_dict['langs'][lang]
                    untrained_lang_emb.weight.data[untrained_id].copy_(pretrained_lang_emb.weight.data[pretrained_id])

        self.model.decoder.language_embeddings = untrained_lang_emb

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                                                                         'Use the param in the forward pass or set requires_grad=False.' +
                                   ' If you are using Stochastic model + fp16 - '
                                   'try to increase the number of minibatches' +
                                   ' each update to avoid uninitialized gradients.')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(
                grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset + numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]

    def warm_up(self):
        """
        Warmup the memory allocator, by attempting to fit the largest batch
        :return:
        """
        if self.opt.memory_profiling:
            from pytorch_memlab import MemReporter
            reporter = MemReporter()

        batch = self.train_data[0].get_largest_batch() if isinstance(self.train_data, list) \
            else self.train_data.get_largest_batch()
        opt = self.opt

        if self.cuda:
            batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

        self.model.train()
        self.model.zero_grad()
        oom = False

        if self.opt.memory_profiling:
            print("Input size: ")
            print(batch.size, batch.src_size, batch.tgt_size)

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        try:
            targets = batch.get('target_output')
            tgt_mask = None
            outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                 zero_encoder=opt.zero_encoder,
                                 mirror=opt.mirror_loss, streaming_state=streaming_state,
                                 nce=opt.nce)

            outputs['tgt_mask'] = tgt_mask

            loss_dict = self.loss_function(outputs, targets, model=self.model)
            loss = loss_dict['loss']  # a little trick to avoid gradient overflow with fp16
            full_loss = loss

            if opt.mirror_loss:
                rev_loss = loss_dict['rev_loss']
                mirror_loss = loss_dict['mirror_loss']
                full_loss = full_loss + rev_loss + mirror_loss

            # reconstruction loss
            if opt.reconstruct:
                rec_loss = loss_dict['rec_loss']
                rec_loss = rec_loss
                full_loss = full_loss + rec_loss

            if opt.lfv_multilingual or opt.lid_loss:
                lid_logits = outputs['lid_logits']
                lid_labels = batch.get('target_lang')
                lid_loss_function = self.loss_function.get_loss_function('lid_loss')
                lid_loss = lid_loss_function([lid_logits.unsqueeze(0)], lid_labels)
                full_loss = full_loss + lid_loss

            optimizer = self.optim.optimizer

            if self.opt.memory_profiling:
                reporter.report(verbose=True)

                # for obj in gc.get_objects():
                #     try:
                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #             # print(varname(obj))
                #             # we can rule out parameter cost later
                #             # if 'parameter' not in type(obj):
                #             # if len(obj.shape) == 3:
                #             # if not isinstance(obj, torch.nn.parameter.Parameter):
                #             #     tensor = obj
                #             #     numel = tensor.
                #             print(type(obj), obj.type(), obj.size())
                #     except:
                #         pass

                # print("Memory profiling complete.")
                # print(torch.cuda.memory_summary())
                # exit()

            if self.cuda:
                with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.div_(batch.tgt_size).backward()

            if self.opt.memory_profiling:
                print('========= after backward =========')
                reporter.report(verbose=True)

            self.model.zero_grad()
            self.optim.zero_grad()
            # self.optim.step()
            # self.optim.reset()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
            else:
                raise e

        if oom:
            print("* Warning: out-of-memory in warming up. This is due to the largest batch is too big for the GPU.")
        else:
            print("* Warming up successuflly.")

        if self.opt.memory_profiling:
            if hasattr(torch.cuda, 'memory_summary'):
                print(torch.cuda.memory_summary())
            exit()


class SpeechFNTrainer(object):
    def __init__(self, model, lat_dis, loss_function, train_data, valid_data, dicts, opt, clf=None,
                 setup_optimizer=True):

        self.train_data = train_data
        self.valid_data = valid_data
        self.n_cat = len(dicts['langs'])
        self.dicts = dicts
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1 and opt.gpus[0] >= 0)

        self.start_time = 0
        self.n_gpus = len(self.opt.gpus)

        self.loss_function_ae, self.loss_lat_dis = loss_function
        self.model_ae = model
        self.lat_dis = lat_dis
        self.clf = clf

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            if self.opt.seed >= 0:
                torch.manual_seed(self.opt.seed)

            self.loss_function_ae = self.loss_function_ae.cuda()
            self.model_ae = self.model_ae.cuda()

            self.lat_dis = self.lat_dis.cuda()
            self.loss_lat_dis = self.loss_lat_dis.cuda()

            if self.clf is not None:
                self.clf = self.clf.cuda()

        if setup_optimizer:

            self.optim_ae = onmt.Optim(opt)
            self.optim_ae.set_parameters(self.model_ae.parameters())

            self.optim_lat_dis = onmt.Optim(opt)
            self.optim_lat_dis.set_parameters(self.lat_dis.parameters())

            if self.clf is not None:
                self.optim_clf = onmt.Optim(opt)
                self.optim_clf.set_parameters(self.clf.parameters())

            if not self.opt.fp16:
                opt_level = "O0"
                keep_batchnorm_fp32 = False
            elif self.opt.fp16_mixed:
                opt_level = "O1"
                keep_batchnorm_fp32 = None
            else:
                opt_level = "O2"
                keep_batchnorm_fp32 = False

            if self.cuda:
                # print(234)
                self.model_ae, self.optim_ae.optimizer = amp.initialize(self.model_ae,
                                                                        self.optim_ae.optimizer,
                                                                        opt_level=opt_level,
                                                                        keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                        loss_scale="dynamic",
                                                                        verbosity=1 if self.opt.verbose else 0)

                self.lat_dis, self.optim_lat_dis.optimizer = amp.initialize(self.lat_dis,
                                                                            self.optim_lat_dis.optimizer,
                                                                            opt_level=opt_level,
                                                                            keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                            loss_scale="dynamic",
                                                                            verbosity=1 if self.opt.verbose else 0)

                if self.clf is not None:
                    self.clf, self.optim_clf.optimizer = amp.initialize(self.clf,
                                                                        self.optim_clf.optimizer,
                                                                        opt_level=opt_level,
                                                                        keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                        loss_scale="dynamic",
                                                                        verbosity=1 if self.opt.verbose else 0)

    def warm_up(self):
        """
        Warmup the memory allocator, by attempting to fit the largest batch
        :return:
        """
        print("Tacotron_warmup")
        if self.opt.memory_profiling:
            from pytorch_memlab import MemReporter
            reporter = MemReporter()

        batch = self.train_data[0].get_largest_batch() if isinstance(self.train_data, list) \
            else self.train_data.get_largest_batch()
        opt = self.opt

        if self.cuda:
            batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

        self.model_ae.train()
        self.model_ae.zero_grad()
        oom = False

        if self.opt.memory_profiling:
            print("Input size: ")
            print(batch.size, batch.src_size, batch.tgt_size)

        try:

            encoder_outputs, decoder_outputs = self.model_ae(batch)

            gate_padded = batch.get('gate_padded')

            if self.opt.n_frames_per_step > 1:
                slice = torch.arange(self.opt.n_frames_per_step - 1, gate_padded.size(1), self.opt.n_frames_per_step)
                gate_padded = gate_padded[:, slice]

            src_org = batch.get('source_org')
            src_org = src_org.narrow(2, 1, src_org.size(2) - 1)
            target = [src_org.permute(1, 2, 0).contiguous(), gate_padded]
            loss = self.loss_function_ae(decoder_outputs, target)
            full_loss = loss

            optimizer = self.optim_ae.optimizer

            if self.opt.memory_profiling:
                reporter.report(verbose=True)

            if self.cuda:
                with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.div_(batch.tgt_size).backward()

            if self.opt.memory_profiling:
                print('========= after backward =========')
                reporter.report(verbose=True)

            self.model_ae.zero_grad()
            self.optim_ae.zero_grad()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
            else:
                raise e

        if oom:
            print("* Warning: out-of-memory in warming up. This is due to the largest batch is too big for the GPU.")
        else:
            print("* Warming up successuflly.")

        if self.opt.memory_profiling:
            if hasattr(torch.cuda, 'memory_summary'):
                print(torch.cuda.memory_summary())
            exit()

    def clf_backward(self, batch):
        oom = False
        self.clf.train()

        try:
            src_org = batch.get('source')
            src_lang = batch.get('source_lang')
            preds, src_mask = self.clf(src_org.transpose(0, 1))
            loss = self.loss_lat_dis(preds, src_lang, mask=src_mask, adversarial=False)

            loss_data = loss.data.item()
            # a little trick to avoid gradient overflow with fp16
            full_loss = loss

            optimizer = self.optim_clf.optimizer

            # When the batch size is large, each gradient step is very easy to explode on fp16
            # Normalizing the loss to grad scaler ensures this will not happen
            full_loss.div_(1.0)

            if self.cuda:
                with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                full_loss.backward()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on GPU , skipping batch')
                oom = True
                torch.cuda.empty_cache()
                loss = 0
            else:
                raise e

        if loss != loss:

            # catching NAN problem
            oom = True

            self.model_ae.zero_grad()
            self.loss_lat_dis.zero_grad()
            self.clf.zero_grad()
            self.optim_ae.zero_grad()
            self.optim_clf.zero_grad()
            self.optim_lat_dis.zero_grad()

            self.nan_counter = self.nan_counter + 1
            print("Warning!!! Loss is Nan")
            if self.nan_counter >= 15:
                raise ValueError("Training stopped because of multiple NaN occurence. "
                                 "For ASR, using the Relative Transformer is more stable and recommended.")
        else:
            self.nan_counter = 0

        if not oom:
            self.optim_clf.step()

            self.model_ae.zero_grad()
            self.loss_lat_dis.zero_grad()
            self.clf.zero_grad()
            self.optim_ae.zero_grad()
            self.optim_clf.zero_grad()
            self.optim_lat_dis.zero_grad()

        return loss_data

    def lat_dis_backward(self, batch):
        oom = False
        self.model_ae.eval()
        self.lat_dis.train()
        try:
            encoder_outputs = self.model_ae.encode(batch)
            preds = self.lat_dis(encoder_outputs['context'].detach())

            loss = self.loss_lat_dis(preds, batch.get('source_lang'), mask=encoder_outputs['src_mask'],
                                     adversarial=False)

            loss_data = loss.data.item()
            # a little trick to avoid gradient overflow with fp16
            full_loss = loss

            optimizer = self.optim_lat_dis.optimizer

            # When the batch size is large, each gradient step is very easy to explode on fp16
            # Normalizing the loss to grad scaler ensures this will not happen
            full_loss.div_(1.0)

            if self.cuda:
                with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                full_loss.backward()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on GPU , skipping batch')
                oom = True
                torch.cuda.empty_cache()
                loss = 0
            else:
                raise e

        if loss != loss:

            # catching NAN problem
            oom = True
            self.model_ae.zero_grad()
            self.loss_lat_dis.zero_grad()
            self.clf.zero_grad()
            self.optim_ae.zero_grad()
            self.optim_clf.zero_grad()
            self.optim_lat_dis.zero_grad()

            self.nan_counter = self.nan_counter + 1
            print("Warning!!! Loss is Nan")
            if self.nan_counter >= 15:
                raise ValueError("Training stopped because of multiple NaN occurence. "
                                 "For ASR, using the Relative Transformer is more stable and recommended.")
        else:
            self.nan_counter = 0

        if not oom:
            self.optim_lat_dis.step()

            self.model_ae.zero_grad()
            self.loss_lat_dis.zero_grad()
            self.clf.zero_grad()
            self.optim_ae.zero_grad()
            self.optim_clf.zero_grad()
            self.optim_lat_dis.zero_grad()

        return loss_data

    def autoencoder_backward(self, batch, step=0):

        self.model_ae.train()
        self.lat_dis.eval()
        oom = False
        try:
            encoder_outputs, decoder_outputs = self.model_ae(batch)

            gate_padded = batch.get('gate_padded')

            if self.opt.n_frames_per_step > 1:
                slice = torch.arange(self.opt.n_frames_per_step - 1, gate_padded.size(1), self.opt.n_frames_per_step)
                gate_padded = gate_padded[:, slice]

            src_org = batch.get('source_org')
            src_org = src_org.narrow(2, 1, src_org.size(2) - 1)
            target = [src_org.permute(1, 2, 0).contiguous(), gate_padded]
            loss = self.loss_function_ae(decoder_outputs, target)
            # loss_data = loss.data.item()
            # if self.opt.lambda_lat_dis:

            if self.opt.lambda_lat_dis > 0:
                latern = self.model_ae.encode(batch)
                lat_dis_preds = self.lat_dis(latern['context'])
                adversarial_loss = self.loss_lat_dis(lat_dis_preds, batch.get('source_lang'),
                                                     mask=latern['src_mask'], adversarial=True)
                loss = loss + get_lambda(self.opt.lambda_lat_dis, step) * adversarial_loss  # lambda

            swap_classifier_loss = 0.0

            if self.opt.lambda_clf > 0:
                self.clf.train()
                fake_attributes = flip_attributes(batch.get('source_lang'), self.n_cat)
                outputs_flipped = self.model_ae.decode_batch(latern, fake_attributes, batch.get('source_org').size(0))[0]
                outputs_flipped = outputs_flipped.transpose(1, 2)
                outputs_flipped_mask = batch.get('source_org').transpose(0, 1).narrow(2, 0, 1)
                outputs_flipped = torch.cat([outputs_flipped_mask, outputs_flipped], dim=-1)

                flip_pred, src_mask = self.clf(outputs_flipped)

                swap_classifier_loss = self.loss_lat_dis(flip_pred, fake_attributes, mask=latern['src_mask'],
                                                         adversarial=False)

                loss = loss + get_lambda(self.opt.lambda_clf, step) * swap_classifier_loss

            loss_data = loss.data.item()
            adversarial_loss_data = adversarial_loss.data.item()
            swap_classifier_loss_data = swap_classifier_loss.data.item()
            # a little trick to avoid gradient overflow with fp16
            full_loss = loss

            optimizer = self.optim_ae.optimizer

            # When the batch size is large, each gradient step is very easy to explode on fp16
            # Normalizing the loss to grad scaler ensures this will not happen
            full_loss.div_(1.0)

            if self.cuda:
                with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                full_loss.backward()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on GPU , skipping batch')
                oom = True
                torch.cuda.empty_cache()
                loss = 0
            else:
                raise e

        if loss != loss:
            oom = True
            self.model_ae.zero_grad()
            self.loss_lat_dis.zero_grad()
            self.clf.zero_grad()
            self.optim_ae.zero_grad()
            self.optim_clf.zero_grad()
            self.optim_lat_dis.zero_grad()

            self.nan_counter = self.nan_counter + 1
            print("Warning!!! Loss is Nan")
            if self.nan_counter >= 15:
                raise ValueError("Training stopped because of multiple NaN occurence. "
                                 "For ASR, using the Relative Transformer is more stable and recommended.")
        else:

            self.nan_counter = 0

        self.optim_ae.step()

        self.model_ae.zero_grad()
        self.loss_lat_dis.zero_grad()
        self.clf.zero_grad()
        self.optim_ae.zero_grad()
        self.optim_clf.zero_grad()
        self.optim_lat_dis.zero_grad()

        return loss_data, adversarial_loss_data, swap_classifier_loss_data

    def save(self, epoch, valid_loss, itr=None):

        opt = self.opt
        model = self.model_ae
        dicts = self.dicts

        model_state_dict = self.model_ae.state_dict()
        optim_state_dict = self.optim_ae.state_dict()

        if itr:
            itr_state_dict = itr.state_dict()
        else:
            itr_state_dict = None

        #  drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dicts,
            'opt': opt,
            'epoch': epoch,
            'itr': itr_state_dict,
            'optim': optim_state_dict,
            'amp': amp.state_dict()
        }

        file_name = '%s_ppl_%.6f_e%.2f.pt' % (opt.save_model, valid_loss, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

        # check the save directory here
        checkpoint_dir = os.path.dirname(opt.save_model)
        existed_save_files = checkpoint_paths(checkpoint_dir)
        for save_file in existed_save_files[opt.keep_save_files:]:
            print(" * Deleting old save file %s ...." % save_file)
            os.remove(save_file)

    def run(self, checkpoint=None):

        opt = self.opt
        model_ae = self.model_ae
        optim_ae = self.optim_ae

        if checkpoint is not None:
            self.model_ae.load_state_dict(checkpoint['model'])
            prec_opt = checkpoint['opt'] if 'opt' in checkpoint else None

            if not opt.reset_optim:
                print("* Loading optimizer states ... ")
                self.optim_ae.load_state_dict(checkpoint['optim'])
                if prec_opt is not None and hasattr(prec_opt, "fp16_mixed"):
                    # Only load amp information if the mode is the same
                    # Maybe its better to change between optimization mode?
                    if opt.fp16_mixed == prec_opt.fp16_mixed and opt.fp16 == prec_opt.fp16:
                        if 'amp' in checkpoint:
                            amp.load_state_dict(checkpoint['amp'])

                # Only load the progress when we use the same optimizer
                if 'itr' in checkpoint:
                    itr_progress = checkpoint['itr']
                else:
                    itr_progress = None

                resume = True
                start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 1
                if start_epoch is None:
                    start_epoch = 1
            else:
                itr_progress = None
                resume = False
                start_epoch = 1

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            itr_progress = None
            print('Initializing model parameters')
            init_model_parameters(model_ae, opt)
            resume = False
            start_epoch = 1

        # if we are on a GPU: warm up the memory allocator
        if self.cuda:
            self.warm_up()
            #
            valid_loss_ae, valid_loss_lat_dis, valid_loss_clf = self.eval(self.valid_data)
            #
            print('Validation loss ae: %g' % valid_loss_ae)
            print('Validation loss latent discriminator: %g' % valid_loss_lat_dis)
            print('Validation loss classifier: %g' % valid_loss_clf)
        #
        self.start_time = time.time()

        for epoch in range(start_epoch, start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss_ae, train_loss_lat_dis, train_loss_adv, train_loss_clf_swap, train_loss_clf = self.train_epoch(
                epoch, resume=resume,
                itr_progress=itr_progress)

            print('Train loss ae: %g' % train_loss_ae)
            print('Train loss latent discriminator: %g' % train_loss_lat_dis)
            print('Train loss adversarial : %g' % train_loss_adv)
            print('Train loss classifier  : %g' % train_loss_clf)
            print('Train loss classifier swap : %g' % train_loss_clf_swap)

            # #  (2) evaluate on the validation set
            valid_loss_ae, valid_loss_lat_dis, val_loss_clf = self.eval(self.valid_data)
            print('Validation loss ae: %g' % valid_loss_ae)
            print('Validation loss latent discriminator: %g' % valid_loss_lat_dis)
            print('Validation loss classifier: %g' % val_loss_clf)
            #
            self.save(epoch, valid_loss_ae)
            itr_progress = None
            resume = False

    def eval(self, data):
        total_loss_ae = 0
        total_loss_lat_dis = 0
        total_loss_clf = 0
        total_tgt_frames = 0
        total_sent = 0
        opt = self.opt

        self.model_ae.eval()
        self.loss_function_ae.eval()
        self.lat_dis.eval()
        self.loss_lat_dis.eval()
        # self.model.reset_states()

        # the data iterator creates an epoch iterator
        data_iterator = generate_data_iterator(data, seed=self.opt.seed,
                                               num_workers=opt.num_workers, epoch=1, buffer_size=opt.buffer_size)
        epoch_iterator = data_iterator.next_epoch_itr(False, pin_memory=False)

        """ PyTorch semantics: save space by not creating gradients """

        data_size = len(epoch_iterator)
        i = 0

        with torch.no_grad():
            # for i in range(len()):
            while not data_iterator.end_of_epoch():
                # batch = data.next()[0]
                batch = next(epoch_iterator)
                if isinstance(batch, list):
                    batch = batch[0]
                batch = rewrap(batch)

                if self.cuda:
                    batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                encoder_outputs, decoder_outputs = self.model_ae(batch)

                gate_padded = batch.get('gate_padded')

                if self.opt.n_frames_per_step > 1:
                    slice = torch.arange(self.opt.n_frames_per_step - 1, gate_padded.size(1),
                                         self.opt.n_frames_per_step)
                    gate_padded = gate_padded[:, slice]

                src_org = batch.get('source_org')
                src_org = src_org.narrow(2, 1, src_org.size(2) - 1)
                target = [src_org.permute(1, 2, 0).contiguous(), gate_padded]
                loss_ae = self.loss_function_ae(decoder_outputs, target)
                loss_ae_data = loss_ae.data.item()

                preds = self.lat_dis(encoder_outputs['context'])

                loss_lat_dis = self.loss_lat_dis(preds, batch.get('source_lang'), mask=encoder_outputs['src_mask'],
                                                 adversarial=False)
                loss_lat_dis_data = loss_lat_dis.data.item()

                src_org = batch.get('source_org')
                src_lang = batch.get('source_lang')
                preds, src_mask = self.clf(src_org.transpose(0, 1))
                loss_clf = self.loss_lat_dis(preds, src_lang, mask=src_mask, adversarial=False)
                loss_clf_data = loss_clf.data.item()

                total_loss_ae += loss_ae_data
                total_loss_lat_dis += loss_lat_dis_data
                total_loss_clf += loss_clf_data
                total_tgt_frames += batch.src_size
                total_sent += batch.size
                i = i + 1

        return total_loss_ae / data_size * 100, total_loss_lat_dis / data_size * 100, total_loss_clf / data_size * 100

    def train_epoch(self, epoch, resume=False, itr_progress=None):

        global rec_ppl
        opt = self.opt
        train_data = self.train_data
        streaming = opt.streaming

        self.model_ae.train()
        self.loss_function_ae.train()
        self.lat_dis.train()
        self.loss_lat_dis.train()

        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model_ae.zero_grad()
        self.lat_dis.zero_grad()

        dataset = train_data
        data_iterator = generate_data_iterator(dataset, seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=epoch, buffer_size=opt.buffer_size)

        if resume:
            data_iterator.load_state_dict(itr_progress)

        epoch_iterator = data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

        total_loss_ae, total_loss_lat_dis, total_frames, total_loss_adv, total_loss_clf_swap, total_loss_clf = 0, 0, 0, 0, 0, 0

        report_loss_ae, report_loss_lat_dis, report_loss_adv, report_loss_clf_swap, report_loss_clf, report_tgt_frames, report_sent = 0, 0, 0, 0, 0, 0, 0

        start = time.time()
        n_samples = len(epoch_iterator)

        counter = 0
        step = 0

        num_accumulated_sents = 0
        grad_scaler = -1
        self.nan_counter = 0
        nan = False
        nan_counter = 0
        n_step_ae = opt.update_frequency
        n_step_lat_dis = opt.update_frequency

        mode = ["ae"] * self.opt.n_step_ae + ["lat_dis"] * self.opt.n_step_lat_dis + ["clf"] * self.opt.n_clf
        random.shuffle(mode)

        mode_i = 0

        loss_lat_dis = 0.0
        loss_swap_clf = 0.0
        loss_clf = 0.0
        loss_adv = 0.0
        loss_ae = 0.0

        i = data_iterator.iterations_in_epoch if not isinstance(train_data, list) else epoch_iterator.n_yielded

        while not data_iterator.end_of_epoch():

            curriculum = (epoch < opt.curriculum)

            # this batch generator is not very clean atm
            batch = next(epoch_iterator)
            if isinstance(batch, list) and self.n_gpus == 1:
                batch = batch[0]
            batch = rewrap(batch)

            batch_size = batch.size
            if grad_scaler == -1:
                grad_scaler = 1  # if self.opt.update_frequency > 1 else batch.tgt_size

            if self.cuda:
                batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)



                # outputs is a dictionary containing keys/values necessary for loss function
                # can be flexibly controlled within models for easier extensibility
                #    targets = batch.get('target_output')
                #  tgt_mask = targets.ne(onmt.constants.PAD)

            oom = False

            if mode[mode_i] == "ae":

                loss_ae, loss_adv, loss_swap_clf = self.autoencoder_backward(batch, step)

            elif mode[mode_i] == "lat_dis":

                loss_lat_dis = self.lat_dis_backward(batch)

            else:

                loss_clf = self.clf_backward(batch)




            if not oom:
                src_size = batch.src_size

                counter = counter + 1
                mode_i = mode_i + 1
                #   We only update the parameters after getting gradients from n mini-batches
                update_flag = False
                if mode_i >= len(mode) or i == n_samples:
                    update_flag = True

                if update_flag:
                    # # accumulated gradient case, in this case the update frequency
                    # if (counter == 1 and self.opt.update_frequency != 1) or counter > 1:
                    #     grad_denom = 1 / grad_scaler
                    #     # if self.opt.normalize_gradient:
                    #     #     grad_denom = num_accumulated_words * grad_denom
                    # else:
                    #     grad_denom = 1.0
                    # # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                    # normalize_gradients(amp.master_params(self.optim_ae.optimizer), grad_denom)
                    # normalize_gradients(amp.master_params(self.optim_lat_dis.optimizer), grad_denom)
                    # # Update the parameters.
                    # if self.opt.max_grad_norm > 0:
                    #     torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim_ae.optimizer),
                    #                                    self.opt.max_grad_norm)
                    #     torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim_lat_dis.optimizer),
                    #                                    self.opt.max_grad_norm)
                    self.nan_counter = 0
                    mode_i = 0
                    random.shuffle(mode)
                    counter = 0


                    grad_scaler = -1

                report_loss_ae += loss_ae
                report_loss_lat_dis += loss_lat_dis
                report_loss_adv += loss_adv
                report_loss_clf += loss_clf
                report_loss_clf_swap += loss_swap_clf

                # report_tgt_words += num_words
                num_accumulated_sents += batch_size
                report_sent += batch_size
                total_frames += src_size
                report_tgt_frames += src_size
                total_loss_ae += loss_ae
                total_loss_lat_dis += loss_lat_dis
                total_loss_adv += loss_adv
                total_loss_clf_swap += loss_swap_clf
                total_loss_clf += loss_clf

                optim_ae = self.optim_ae
                optim_lat_dis = self.optim_lat_dis
                optim_clf = self.optim_clf
                # batch_efficiency = total_non_pads / total_tokens

                step = optim_lat_dis._step

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    log_string = (
                            "Epoch %2d, %5d/%5d; ; loss_ae : %6.2f ;  loss_lat_dis : %6.2f, loss_adv : %6.2f, loss_clf : %6.2f, loss_clf_swap : %6.2f " %
                            (epoch, i + 1, len(data_iterator),
                             report_loss_ae, report_loss_lat_dis, report_loss_adv, report_loss_clf,
                             report_loss_clf_swap))

                    log_string += ("lr_ae: %.7f ; updates: %7d; " %
                                   (optim_ae.getLearningRate(),
                                    optim_ae._step))

                    log_string += ("lr_lat_dis: %.7f ; updates: %7d; " %
                                   (optim_lat_dis.getLearningRate(),
                                    optim_lat_dis._step))

                    log_string += ("lr_clf: %.7f ; updates: %7d; " %
                                   (optim_clf.getLearningRate(),
                                    optim_clf._step))
                    #
                    log_string += ("%5.0f src tok/s " %
                                   (report_tgt_frames / (time.time() - start)))

                    log_string += ("%s elapsed" %
                                   str(datetime.timedelta(seconds=int(time.time() - self.start_time))))

                    print(log_string)

                    report_loss_ae = 0
                    report_loss_lat_dis = 0
                    report_loss_adv = 0
                    report_loss_clf_swap = 0
                    report_loss_clf = 0
                    report_tgt_frames = 0
                    report_sent = 0
                    start = time.time()

                i = i + 1

        return total_loss_ae / n_samples * 100, total_loss_lat_dis / n_samples * 100, total_loss_adv / n_samples * 100, total_loss_clf / n_samples * 100, total_loss_clf_swap / n_samples * 100


class SpeechAETrainer(BaseTrainer):
    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt, setup_optimizer=True):
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt)

        self.n_gpus = len(self.opt.gpus)

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            if self.opt.seed >= 0:
                torch.manual_seed(self.opt.seed)
            self.loss_function = self.loss_function.cuda()
            self.model = self.model.cuda()

        if setup_optimizer:

            self.optim = onmt.Optim(opt)
            self.optim.set_parameters(self.model.parameters())

            if not self.opt.fp16:
                opt_level = "O0"
                keep_batchnorm_fp32 = False
            elif self.opt.fp16_mixed:
                opt_level = "O1"
                keep_batchnorm_fp32 = None
            else:
                opt_level = "O2"
                keep_batchnorm_fp32 = False

            if self.cuda:
                # print(234)
                self.model, self.optim.optimizer = amp.initialize(self.model,
                                                                  self.optim.optimizer,
                                                                  opt_level=opt_level,
                                                                  keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                  loss_scale="dynamic",
                                                                  verbosity=1 if self.opt.verbose else 0)

    def warm_up(self):
        """
        Warmup the memory allocator, by attempting to fit the largest batch
        :return:
        """
        print("Tacotron_warmup")
        if self.opt.memory_profiling:
            from pytorch_memlab import MemReporter
            reporter = MemReporter()

        batch = self.train_data[0].get_largest_batch() if isinstance(self.train_data, list) \
            else self.train_data.get_largest_batch()
        opt = self.opt

        if self.cuda:
            batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

        self.model.train()
        self.model.zero_grad()
        oom = False

        if self.opt.memory_profiling:
            print("Input size: ")
            print(batch.size, batch.src_size, batch.tgt_size)

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        try:
            targets = batch.get('target_output')
            tgt_mask = None
            outputs = self.model(batch)

            gate_padded = batch.get('gate_padded')

            if self.opt.n_frames_per_step > 1:
                slice = torch.arange(self.opt.n_frames_per_step - 1, gate_padded.size(1), self.opt.n_frames_per_step)
                gate_padded = gate_padded[:, slice]

            src_org = batch.get('source_org')
            src_org = src_org.narrow(2, 1, src_org.size(2) - 1)
            target = [src_org.permute(1, 2, 0).contiguous(), gate_padded]
            loss = self.loss_function(outputs, target)
            # loss_dict = self.loss_function(outputs, targets, model=self.model)
            loss = loss  # a little trick to avoid gradient overflow with fp16
            full_loss = loss

            optimizer = self.optim.optimizer

            if self.opt.memory_profiling:
                reporter.report(verbose=True)

            if self.cuda:
                with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.div_(batch.tgt_size).backward()

            if self.opt.memory_profiling:
                print('========= after backward =========')
                reporter.report(verbose=True)

            self.model.zero_grad()
            self.optim.zero_grad()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                oom = True
            else:
                raise e

        if oom:
            print("* Warning: out-of-memory in warming up. This is due to the largest batch is too big for the GPU.")
        else:
            print("* Warming up successuflly.")

        if self.opt.memory_profiling:
            if hasattr(torch.cuda, 'memory_summary'):
                print(torch.cuda.memory_summary())
            exit()

    def save(self, epoch, valid_ppl, itr=None):

        opt = self.opt
        model = self.model
        dicts = self.dicts

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()

        if itr:
            itr_state_dict = itr.state_dict()
        else:
            itr_state_dict = None

        #  drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dicts,
            'opt': opt,
            'epoch': epoch,
            'itr': itr_state_dict,
            'optim': optim_state_dict,
            'amp': amp.state_dict()
        }

        file_name = '%s_ppl_%.6f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

        # check the save directory here
        checkpoint_dir = os.path.dirname(opt.save_model)
        existed_save_files = checkpoint_paths(checkpoint_dir)
        for save_file in existed_save_files[opt.keep_save_files:]:
            print(" * Deleting old save file %s ...." % save_file)
            os.remove(save_file)

    def run(self, checkpoint=None):

        opt = self.opt
        model = self.model
        optim = self.optim

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model'])
            prec_opt = checkpoint['opt'] if 'opt' in checkpoint else None

            if not opt.reset_optim:
                print("* Loading optimizer states ... ")
                self.optim.load_state_dict(checkpoint['optim'])
                if prec_opt is not None and hasattr(prec_opt, "fp16_mixed"):
                    # Only load amp information if the mode is the same
                    # Maybe its better to change between optimization mode?
                    if opt.fp16_mixed == prec_opt.fp16_mixed and opt.fp16 == prec_opt.fp16:
                        if 'amp' in checkpoint:
                            amp.load_state_dict(checkpoint['amp'])

                # Only load the progress when we use the same optimizer
                if 'itr' in checkpoint:
                    itr_progress = checkpoint['itr']
                else:
                    itr_progress = None

                resume = True
                start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 1
                if start_epoch is None:
                    start_epoch = 1
            else:
                itr_progress = None
                resume = False
                start_epoch = 1

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            itr_progress = None
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume = False
            start_epoch = 1

        if opt.load_encoder_from:
            self.load_encoder_weight(opt.load_encoder_from)

        if opt.load_decoder_from:
            self.load_decoder_weight(opt.load_decoder_from)

        # if we are on a GPU: warm up the memory allocator
        if self.cuda:
            self.warm_up()

            valid_loss = self.eval(self.valid_data)

            print('Validation loss: %g' % valid_loss)

        self.start_time = time.time()

        for epoch in range(start_epoch, start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, resume=resume, itr_progress=itr_progress)

            print('Train loss: %g' % train_loss)

            #  (2) evaluate on the validation set
            valid_loss = self.eval(self.valid_data)
            print('Validation loss: %g' % valid_loss)

            self.save(epoch, valid_loss)
            itr_progress = None
            resume = False

    def eval(self, data):
        total_loss = 0
        total_tgt_frames = 0
        total_sent = 0
        opt = self.opt

        self.model.eval()
        self.loss_function.eval()
        # self.model.reset_states()

        # the data iterator creates an epoch iterator
        data_iterator = generate_data_iterator(data, seed=self.opt.seed,
                                               num_workers=opt.num_workers, epoch=1, buffer_size=opt.buffer_size)
        epoch_iterator = data_iterator.next_epoch_itr(False, pin_memory=False)

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        """ PyTorch semantics: save space by not creating gradients """

        data_size = len(epoch_iterator)
        i = 0

        with torch.no_grad():
            # for i in range(len()):
            while not data_iterator.end_of_epoch():
                # batch = data.next()[0]
                batch = next(epoch_iterator)
                if isinstance(batch, list):
                    batch = batch[0]
                batch = rewrap(batch)

                if self.cuda:
                    batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """

                encoder_outputs, outputs = self.model(batch)

                gate_padded = batch.get('gate_padded')

                if self.opt.n_frames_per_step > 1:
                    slice = torch.arange(self.opt.n_frames_per_step - 1, gate_padded.size(1),
                                         self.opt.n_frames_per_step)
                    gate_padded = gate_padded[:, slice]

                src_org = batch.get('source_org')
                src_org = src_org.narrow(2, 1, src_org.size(2) - 1)
                target = [src_org.permute(1, 2, 0).contiguous(), gate_padded]
                loss = self.loss_function(outputs, target)
                loss_data = loss.data.item()

                total_loss += loss_data
                total_tgt_frames += batch.src_size
                total_sent += batch.size
                i = i + 1

        self.model.train()
        self.loss_function.train()
        return total_loss / data_size * 100

    def train_epoch(self, epoch, resume=False, itr_progress=None):

        global rec_ppl
        opt = self.opt
        train_data = self.train_data
        streaming = opt.streaming

        self.model.train()
        self.loss_function.train()
        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model.zero_grad()

        dataset = train_data
        data_iterator = generate_data_iterator(dataset, seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=epoch, buffer_size=opt.buffer_size)

        if resume:
            data_iterator.load_state_dict(itr_progress)

        epoch_iterator = data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

        total_loss, total_frames = 0, 0

        report_loss, report_tgt_frames, report_sent = 0, 0, 0

        start = time.time()
        n_samples = len(epoch_iterator)

        counter = 0

        num_accumulated_sents = 0
        grad_scaler = -1

        nan = False
        nan_counter = 0

        i = data_iterator.iterations_in_epoch if not isinstance(train_data, list) else epoch_iterator.n_yielded

        while not data_iterator.end_of_epoch():

            curriculum = (epoch < opt.curriculum)

            # this batch generator is not very clean atm
            batch = next(epoch_iterator)
            if isinstance(batch, list) and self.n_gpus == 1:
                batch = batch[0]
            batch = rewrap(batch)
            if grad_scaler == -1:
                grad_scaler = 1  # if self.opt.update_frequency > 1 else batch.tgt_size

            if self.cuda:
                batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

            oom = False
            try:
                # outputs is a dictionary containing keys/values necessary for loss function
                # can be flexibly controlled within models for easier extensibility
                #    targets = batch.get('target_output')
                #  tgt_mask = targets.ne(onmt.constants.PAD)
                outputs = self.model(batch)

                gate_padded = batch.get('gate_padded')

                if self.opt.n_frames_per_step > 1:
                    slice = torch.arange(self.opt.n_frames_per_step - 1, gate_padded.size(1),
                                         self.opt.n_frames_per_step)
                    gate_padded = gate_padded[:, slice]

                src_org = batch.get('source_org')
                src_org = src_org.narrow(2, 1, src_org.size(2) - 1)

                target = [src_org.permute(1, 2, 0).contiguous(), gate_padded]
                loss = self.loss_function(outputs, target)

                batch_size = batch.size
                loss_data = loss.data.item()
                # a little trick to avoid gradient overflow with fp16
                full_loss = loss

                optimizer = self.optim.optimizer

                # When the batch size is large, each gradient step is very easy to explode on fp16
                # Normalizing the loss to grad scaler ensures this will not happen
                full_loss.div_(grad_scaler)

                if self.cuda:
                    with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    full_loss.backward()

                del outputs

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                    loss = 0
                    if opt.streaming:  # reset stream in this case ...
                        streaming_state = self.model.init_stream()
                else:
                    raise e

            if loss != loss:
                # catching NAN problem
                oom = True
                self.model.zero_grad()
                self.optim.zero_grad()
                nan_counter = nan_counter + 1
                print("Warning!!! Loss is Nan")
                if nan_counter >= 15:
                    raise ValueError("Training stopped because of multiple NaN occurence. "
                                     "For ASR, using the Relative Transformer is more stable and recommended.")
            else:
                nan_counter = 0

            if not oom:
                src_size = batch.src_size

                counter = counter + 1

                #   We only update the parameters after getting gradients from n mini-batches
                update_flag = False
                if counter >= opt.update_frequency > 0:
                    update_flag = True
                elif i == n_samples:  # update for the last minibatch
                    update_flag = True

                if update_flag:
                    # accumulated gradient case, in this case the update frequency
                    if (counter == 1 and self.opt.update_frequency != 1) or counter > 1:
                        grad_denom = 1 / grad_scaler
                        # if self.opt.normalize_gradient:
                        #     grad_denom = num_accumulated_words * grad_denom
                    else:
                        grad_denom = 1.0
                    # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                    normalize_gradients(amp.master_params(optimizer), grad_denom)
                    # Update the parameters.
                    if self.opt.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.opt.max_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()
                    self.model.zero_grad()
                    counter = 0
                    # num_accumulated_words = 0

                    grad_scaler = -1
                    num_updates = self.optim._step
                    if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
                        valid_loss = self.eval(self.valid_data)
                        valid_ppl = math.exp(min(valid_loss, 100))
                        print('Validation perplexity: %g' % valid_ppl)

                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)

                        self.save(ep, valid_ppl, itr=data_iterator)

                report_loss += loss_data
                # report_tgt_words += num_words
                num_accumulated_sents += batch_size
                report_sent += batch_size
                total_frames += src_size
                report_tgt_frames += src_size
                total_loss += loss_data

                optim = self.optim
                # batch_efficiency = total_non_pads / total_tokens

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    log_string = ("Epoch %2d, %5d/%5d; ; loss : %6.2f ; " %
                                  (epoch, i + 1, len(data_iterator),
                                   report_loss))

                    log_string += ("lr: %.7f ; updates: %7d; " %
                                   (optim.getLearningRate(),
                                    optim._step))
                    #
                    log_string += ("%5.0f src tok/s " %
                                   (report_tgt_frames / (time.time() - start)))

                    log_string += ("%s elapsed" %
                                   str(datetime.timedelta(seconds=int(time.time() - self.start_time))))

                    print(log_string)

                    report_loss = 0
                    report_tgt_frames = 0
                    report_sent = 0
                    start = time.time()

                i = i + 1

        return total_loss / n_samples * 100


class XETrainer(BaseTrainer):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt, setup_optimizer=True):
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt)

        if opt.lfv_multilingual or opt.lid_loss:
            from onmt.models.speech_recognizer.lid_loss import CrossEntropyLIDLoss
            lid_loss = CrossEntropyLIDLoss(opt.n_languages, opt.label_smoothing, opt.fast_xentropy)
            self.loss_function.add_loss_function(lid_loss, 'lid_loss')

        self.n_gpus = len(self.opt.gpus)

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            if self.opt.seed >= 0:
                torch.manual_seed(self.opt.seed)
            self.loss_function = self.loss_function.cuda()
            self.model = self.model.cuda()

        if setup_optimizer:

            self.optim = onmt.Optim(opt)
            self.optim.set_parameters(self.model.parameters())

            if not self.opt.fp16:
                opt_level = "O0"
                keep_batchnorm_fp32 = False
            elif self.opt.fp16_mixed:
                opt_level = "O1"
                keep_batchnorm_fp32 = None
            else:
                opt_level = "O2"
                keep_batchnorm_fp32 = False

            if self.cuda:
                # print(234)
                self.model, self.optim.optimizer = amp.initialize(self.model,
                                                                  self.optim.optimizer,
                                                                  opt_level=opt_level,
                                                                  keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                  loss_scale="dynamic",
                                                                  verbosity=1 if self.opt.verbose else 0)
        # An ugly hack to switch between align right and align left
        if hasattr(self.model, 'relative'):
            if self.model.relative:
                self.train_data.src_align_right = True
                self.train_data.tgt_align_right = False
                self.valid_data.src_align_right = True
                self.valid_data.tgt_align_right = False
                self.valid_data.tgt_align_right = False

    def save(self, epoch, valid_ppl, itr=None):

        opt = self.opt
        model = self.model
        dicts = self.dicts

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()

        if itr:
            itr_state_dict = itr.state_dict()
        else:
            itr_state_dict = None

        #  drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dicts,
            'opt': opt,
            'epoch': epoch,
            'itr': itr_state_dict,
            'optim': optim_state_dict,
            'amp': amp.state_dict()
        }

        file_name = '%s_ppl_%.6f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

        # check the save directory here
        checkpoint_dir = os.path.dirname(opt.save_model)
        existed_save_files = checkpoint_paths(checkpoint_dir)
        for save_file in existed_save_files[opt.keep_save_files:]:
            print(" * Deleting old save file %s ...." % save_file)
            os.remove(save_file)

    def eval(self, data):
        total_loss = 0
        total_words = 0
        opt = self.opt

        self.model.eval()
        self.loss_function.eval()
        self.model.reset_states()

        # the data iterator creates an epoch iterator
        data_iterator = generate_data_iterator(data, seed=self.opt.seed,
                                               num_workers=opt.num_workers, epoch=1, buffer_size=opt.buffer_size)
        epoch_iterator = data_iterator.next_epoch_itr(False, pin_memory=False)

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        """ PyTorch semantics: save space by not creating gradients """

        data_size = len(epoch_iterator)
        i = 0

        with torch.no_grad():
            # for i in range(len()):
            while not data_iterator.end_of_epoch():
                # batch = data.next()[0]
                batch = next(epoch_iterator)
                if isinstance(batch, list):
                    batch = batch[0]
                batch = rewrap(batch)

                if self.cuda:
                    batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                targets = batch.get('target_output')
                tgt_mask = targets.ne(onmt.constants.PAD)
                outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                     mirror=opt.mirror_loss, streaming_state=streaming_state, nce=opt.nce)

                if opt.streaming:
                    streaming_state = outputs['streaming_state']

                outputs['tgt_mask'] = tgt_mask

                loss_dict = self.loss_function(outputs, targets, model=self.model, eval=True)

                loss_data = loss_dict['data']

                total_loss += loss_data
                total_words += batch.tgt_size
                i = i + 1

        self.model.train()
        self.loss_function.train()
        return total_loss / total_words

    def train_epoch(self, epoch, resume=False, itr_progress=None):

        global rec_ppl
        opt = self.opt
        train_data = self.train_data
        streaming = opt.streaming

        self.model.train()
        self.loss_function.train()
        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model.zero_grad()
        self.model.reset_states()

        dataset = train_data
        data_iterator = generate_data_iterator(dataset, seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=epoch, buffer_size=opt.buffer_size)

        if resume:
            data_iterator.load_state_dict(itr_progress)

        epoch_iterator = data_iterator.next_epoch_itr(not streaming, pin_memory=opt.pin_memory)

        total_tokens, total_loss, total_words = 0, 0, 0
        total_non_pads = 0
        report_loss, report_tgt_words = 0, 0
        report_src_words = 0
        report_rec_loss, report_rev_loss, report_mirror_loss = 0, 0, 0
        start = time.time()
        n_samples = len(epoch_iterator)

        counter = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0
        grad_scaler = -1

        nan = False
        nan_counter = 0

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        i = data_iterator.iterations_in_epoch if not isinstance(train_data, list) else epoch_iterator.n_yielded

        while not data_iterator.end_of_epoch():

            curriculum = (epoch < opt.curriculum)

            # this batch generator is not very clean atm
            batch = next(epoch_iterator)
            if isinstance(batch, list) and self.n_gpus == 1:
                batch = batch[0]
            batch = rewrap(batch)
            if grad_scaler == -1:
                grad_scaler = 1  # if self.opt.update_frequency > 1 else batch.tgt_size

            if self.cuda:
                batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

            # if opt.streaming:
            #     if train_data.is_new_stream():
            #         streaming_state = self.model.init_stream()
            # else:
            #     streaming_state = None

            oom = False
            try:
                # outputs is a dictionary containing keys/values necessary for loss function
                # can be flexibly controlled within models for easier extensibility
                targets = batch.get('target_output')
                tgt_mask = targets.ne(onmt.constants.PAD)

                outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                     zero_encoder=opt.zero_encoder,
                                     mirror=opt.mirror_loss, streaming_state=streaming_state,
                                     nce=opt.nce)
                # print("time " + str(time.time() - start_time_t))
                batch_size = batch.size

                outputs['tgt_mask'] = tgt_mask

                loss_dict = self.loss_function(outputs, targets, model=self.model)
                loss_data = loss_dict['data']
                loss = loss_dict['loss']  # a little trick to avoid gradient overflow with fp16
                full_loss = loss

                if opt.mirror_loss:
                    rev_loss = loss_dict['rev_loss']
                    rev_loss_data = loss_dict['rev_loss_data']
                    mirror_loss = loss_dict['mirror_loss']
                    full_loss = full_loss + rev_loss + mirror_loss
                    mirror_loss_data = loss_dict['mirror_loss'].item()
                else:
                    rev_loss_data = None
                    mirror_loss_data = 0

                # reconstruction loss
                if opt.reconstruct:
                    rec_loss = loss_dict['rec_loss']
                    rec_loss = rec_loss
                    full_loss = full_loss + rec_loss
                    rec_loss_data = loss_dict['rec_loss_data']
                else:
                    rec_loss_data = None

                if opt.lfv_multilingual or opt.lid_loss:
                    lid_logits = outputs['lid_logits']
                    lid_labels = batch.get('target_lang')
                    lid_loss_function = self.loss_function.get_loss_function('lid_loss')
                    lid_loss = lid_loss_function([lid_logits.unsqueeze(0)], lid_labels)
                    full_loss = full_loss + lid_loss

                optimizer = self.optim.optimizer

                # When the batch size is large, each gradient step is very easy to explode on fp16
                # Normalizing the loss to grad scaler ensures this will not happen
                full_loss.div_(grad_scaler)

                if self.cuda:
                    with amp.scale_loss(full_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    full_loss.backward()

                del outputs

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory on GPU , skipping batch')
                    oom = True
                    torch.cuda.empty_cache()
                    loss = 0
                    if opt.streaming:  # reset stream in this case ...
                        streaming_state = self.model.init_stream()
                else:
                    raise e

            if loss != loss:
                # catching NAN problem
                oom = True
                self.model.zero_grad()
                self.optim.zero_grad()
                num_accumulated_words = 0
                num_accumulated_sents = 0
                nan_counter = nan_counter + 1
                print("Warning!!! Loss is Nan")
                if nan_counter >= 15:
                    raise ValueError("Training stopped because of multiple NaN occurence. "
                                     "For ASR, using the Relative Transformer is more stable and recommended.")
            else:
                nan_counter = 0

            if not oom:
                src_size = batch.src_size
                tgt_size = batch.tgt_size

                counter = counter + 1
                num_accumulated_words += tgt_size
                num_accumulated_sents += batch_size

                #   We only update the parameters after getting gradients from n mini-batches
                update_flag = False
                if counter >= opt.update_frequency > 0:
                    update_flag = True
                elif 0 < opt.batch_size_update <= num_accumulated_words:
                    update_flag = True
                elif i == n_samples:  # update for the last minibatch
                    update_flag = True

                if update_flag:
                    # accumulated gradient case, in this case the update frequency
                    if (counter == 1 and self.opt.update_frequency != 1) or counter > 1:
                        grad_denom = 1 / grad_scaler
                        if self.opt.normalize_gradient:
                            grad_denom = num_accumulated_words * grad_denom
                    else:
                        grad_denom = 1
                    # When we accumulate the gradients, each gradient is already normalized by a constant grad_scaler
                    normalize_gradients(amp.master_params(optimizer), grad_denom)
                    # Update the parameters.
                    if self.opt.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.opt.max_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()
                    self.model.zero_grad()
                    counter = 0
                    num_accumulated_words = 0
                    num_accumulated_sents = 0
                    grad_scaler = -1
                    num_updates = self.optim._step
                    if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
                        valid_loss = self.eval(self.valid_data)
                        valid_ppl = math.exp(min(valid_loss, 100))
                        print('Validation perplexity: %g' % valid_ppl)

                        ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)

                        self.save(ep, valid_ppl, itr=data_iterator)

                num_words = tgt_size
                report_loss += loss_data
                report_tgt_words += num_words
                report_src_words += src_size
                total_loss += loss_data
                total_words += num_words
                total_tokens += batch.get('target_output').nelement()
                total_non_pads += batch.get('target_output').ne(onmt.constants.PAD).sum().item()
                optim = self.optim
                batch_efficiency = total_non_pads / total_tokens

                if opt.reconstruct:
                    report_rec_loss += rec_loss_data

                if opt.mirror_loss:
                    report_rev_loss += rev_loss_data
                    report_mirror_loss += mirror_loss_data

                if i == 0 or (i % opt.log_interval == -1 % opt.log_interval):
                    log_string = ("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; " %
                                  (epoch, i + 1, len(data_iterator),
                                   math.exp(report_loss / report_tgt_words)))

                    if opt.reconstruct:
                        rec_ppl = math.exp(report_rec_loss / report_src_words.item())
                        log_string += (" rec_ppl: %6.2f ; " % rec_ppl)

                    if opt.mirror_loss:
                        rev_ppl = math.exp(report_rev_loss / report_tgt_words)
                        log_string += (" rev_ppl: %6.2f ; " % rev_ppl)
                        # mirror loss per word
                        log_string += (" mir_loss: %6.2f ; " % (report_mirror_loss / report_tgt_words))

                    log_string += ("lr: %.7f ; updates: %7d; " %
                                   (optim.getLearningRate(),
                                    optim._step))

                    log_string += ("%5.0f src tok/s; %5.0f tgt tok/s; " %
                                   (report_src_words / (time.time() - start),
                                    report_tgt_words / (time.time() - start)))

                    log_string += ("%s elapsed" %
                                   str(datetime.timedelta(seconds=int(time.time() - self.start_time))))

                    print(log_string)

                    report_loss = 0
                    report_tgt_words, report_src_words = 0, 0
                    report_rec_loss, report_rev_loss, report_mirror_loss = 0, 0, 0
                    start = time.time()

                i = i + 1

        return total_loss / total_words

    # def run(self, save_file=None):
    def run(self, checkpoint=None):

        opt = self.opt
        model = self.model
        optim = self.optim

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model'])
            prec_opt = checkpoint['opt'] if 'opt' in checkpoint else None

            if not opt.reset_optim:
                print("* Loading optimizer states ... ")
                self.optim.load_state_dict(checkpoint['optim'])
                if prec_opt is not None and hasattr(prec_opt, "fp16_mixed"):
                    # Only load amp information if the mode is the same
                    # Maybe its better to change between optimization mode?
                    if opt.fp16_mixed == prec_opt.fp16_mixed and opt.fp16 == prec_opt.fp16:
                        if 'amp' in checkpoint:
                            amp.load_state_dict(checkpoint['amp'])

                # Only load the progress when we use the same optimizer
                if 'itr' in checkpoint:
                    itr_progress = checkpoint['itr']
                else:
                    itr_progress = None

                resume = True
                start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 1
                if start_epoch is None:
                    start_epoch = 1
            else:
                itr_progress = None
                resume = False
                start_epoch = 1

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            itr_progress = None
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume = False
            start_epoch = 1

        if opt.load_encoder_from:
            self.load_encoder_weight(opt.load_encoder_from)

        if opt.load_decoder_from:
            self.load_decoder_weight(opt.load_decoder_from)

        # if we are on a GPU: warm up the memory allocator
        self.start_time = time.time()

        if self.cuda:
            self.warm_up()

            valid_loss = self.eval(self.valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))

            print('Validation perplexity: %g' % valid_ppl)

            # valid_loss = self.train_epoch(0)
            # valid_ppl = math.exp(min(valid_loss, 100))
            #
            # print('Validation perplexity: %g' % valid_ppl)

        for epoch in range(start_epoch, start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, resume=resume, itr_progress=itr_progress)
            train_ppl = math.exp(min(train_loss, 100))
            print('Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss = self.eval(self.valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %g' % valid_ppl)

            self.save(epoch, valid_ppl)
            itr_progress = None
            resume = False
