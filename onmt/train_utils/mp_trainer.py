from __future__ import division

import datetime
import gc
import inspect
import math
import os
import re
import time
import torch
from apex import amp

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

import torch.distributed as dist
# from onmt.multiprocessing.multiprocessing_wrapper import MultiprocessingRunner


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def prepare_sample(batch, device=None, fp16=False):

    # TODO: sample is a Batch object. This function probably
    if isinstance(batch, list):
        batch = batch[0]
    batch = rewrap(batch)
    batch.cuda(fp16=fp16, device=device)

    return batch


def generate_data_iterator(dataset, rank, world_size, seed,
                           num_workers=1, epoch=1., buffer_size=0):

    # check if dataset is a list:
    if isinstance(dataset, list):
        # this is a multidataset
        data_iterator = MultiDataIterator(dataset, seed=seed, num_workers=num_workers,
                                          epoch=epoch, buffer_size=buffer_size)
    else:

        data_iterator = DataIterator(dataset, dataset.collater, dataset.batches, seed=seed,
                                     num_workers=num_workers, epoch=epoch, buffer_size=buffer_size,
                                     num_shards=world_size, shard_id=rank)

    return data_iterator


class Trainer(object):

    def __init__(self, model, device, loss_function, train_data, valid_data, dicts, opt):
        """
        :param model:
        :param device: int (GPU id)
        :param loss_function:
        :param train_data:
        :param valid_data:
        :param dicts:
        :param opt:
        """

        # self.model = model
        init_model_parameters(model, opt)
        self.model = model
        self.loss_function = loss_function
        self.device = device
        self.world_size = len(opt.gpus)

        # make a group to later use with dist.all_reduce
        self.group = dist.new_group(range(self.world_size))

        if opt.load_encoder_from:
            self.load_encoder_weight(opt.load_encoder_from)

        if opt.load_decoder_from:
            self.load_decoder_weight(opt.load_decoder_from)

        self.model = None

        self.train_data = train_data
        self.valid_data = valid_data

        self.dicts = dicts
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1 and opt.gpus[0] >= 0)

        self.loss_function = loss_function
        self.start_time = 0

        # setting up models and others

        if opt.lfv_multilingual:
            from onmt.models.speech_recognizer.lid_loss import CrossEntropyLIDLoss
            lid_loss = CrossEntropyLIDLoss(opt.n_languages, opt.label_smoothing, opt.fast_xentropy)
            self.loss_function.add_loss_function(lid_loss, 'lid_loss')

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            if self.opt.seed >= 0:
                torch.manual_seed(self.opt.seed)
            self.loss_function = self.loss_function.cuda(device=self.device)
            self.model = self.model.cuda(device=self.device)

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
                self.model, self.optim.optimizer = amp.initialize(self.model,
                                                                  self.optim.optimizer,
                                                                  opt_level=opt_level,
                                                                  keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                  loss_scale="dynamic",
                                                                  verbosity=1 if self.opt.verbose else 0)

            # TODO: wrap DDP
            # self.model = DDP(self.model)

    def run(self, *opt, **kwopt):

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

    def warm_up(self):
        """
        Warmup the memory allocator, by attempting to fit the largest batch
        :return:
        """

        # if self.opt.memory_profiling:
        #     from pytorch_memlab import MemReporter
        #     reporter = MemReporter()
        #
        # batch = self.train_data[0].get_largest_batch() if isinstance(self.train_data, list) \
        #     else self.train_data.get_largest_batch()
        # opt = self.opt
        #
        # if self.cuda:
        #     batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)
        #
        # self.model.train()
        # self.model.zero_grad()
        # oom = False
        #
        # if self.opt.memory_profiling:
        #     print("Input size: ")
        #     print(batch.size, batch.src_size, batch.tgt_size)
        #
        # if opt.streaming:
        #     streaming_state = self.model.init_stream()
        # else:
        #     streaming_state = None
        #
        # try:
        #     targets = batch.get('target_output')
        #     tgt_mask = None
        #     outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
        #                          zero_encoder=opt.zero_encoder,
        #                          mirror=opt.mirror_loss, streaming_state=streaming_state,
        #                          nce=opt.nce)
        #
        #     outputs['tgt_mask'] = tgt_mask
        #
        #     loss_dict = self.loss_function(outputs, targets, model=self.model)
        #     loss = loss_dict['loss']  # a little trick to avoid gradient overflow with fp16
        #     full_loss = loss
        #
        #     if opt.mirror_loss:
        #         rev_loss = loss_dict['rev_loss']
        #         mirror_loss = loss_dict['mirror_loss']
        #         full_loss = full_loss + rev_loss + mirror_loss
        #
        #     # reconstruction loss
        #     if opt.reconstruct:
        #         rec_loss = loss_dict['rec_loss']
        #         rec_loss = rec_loss
        #         full_loss = full_loss + rec_loss
        #
        #     if opt.lfv_multilingual:
        #         lid_logits = outputs['lid_logits']
        #         lid_labels = batch.get('target_lang')
        #         lid_loss_function = self.loss_function.get_loss_function('lid_loss')
        #         lid_loss = lid_loss_function(lid_logits, lid_labels)
        #         full_loss = full_loss + lid_loss
        #
        #     optimizer = self.optim.optimizer
        #
        #     if self.opt.memory_profiling:
        #         reporter.report(verbose=True)
        #
        #         # for obj in gc.get_objects():
        #         #     try:
        #         #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #         #             # print(varname(obj))
        #         #             # we can rule out parameter cost later
        #         #             # if 'parameter' not in type(obj):
        #         #             # if len(obj.shape) == 3:
        #         #             # if not isinstance(obj, torch.nn.parameter.Parameter):
        #         #             #     tensor = obj
        #         #             #     numel = tensor.
        #         #             print(type(obj), obj.type(), obj.size())
        #         #     except:
        #         #         pass
        #
        #         # print("Memory profiling complete.")
        #         # print(torch.cuda.memory_summary())
        #         # exit()
        #
        #     if self.cuda:
        #         with amp.scale_loss(full_loss, optimizer) as scaled_loss:
        #             scaled_loss.backward()
        #     else:
        #         loss.div_(batch.tgt_size).backward()
        #
        #     if self.opt.memory_profiling:
        #         print('========= after backward =========')
        #         reporter.report(verbose=True)
        #
        #     self.model.zero_grad()
        #     self.optim.zero_grad()
        #     # self.optim.step()
        #     # self.optim.reset()
        #
        # except RuntimeError as e:
        #     if 'out of memory' in str(e):
        #         oom = True
        #     else:
        #         raise e
        #
        # if oom:
        #     print("* Warning: out-of-memory in warming up. This is due to the largest batch is too big for the GPU.")
        # else:
        #     print("* Warming up successuflly.")
        #
        # if self.opt.memory_profiling:
        #     if hasattr(torch.cuda, 'memory_summary'):
        #         print(torch.cuda.memory_summary())
        #     exit()

        pass

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
            'amp': amp_state_dict
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
        rank = self.device
        world_size = self.world_size

        # the data iterator creates an epoch iterator
        data_iterator = generate_data_iterator(data, rank, world_size, seed=self.opt.seed,
                                               num_workers=opt.num_workers, epoch=1, buffer_size=opt.buffer_size)
        epoch_iterator = data_iterator.next_epoch_itr(False, pin_memory=False)

        data_size = len(epoch_iterator)
        i = 0

        self.model.eval()
        self.loss_function.eval()
        self.model.reset_states()

        total_loss = torch.Tensor([0]).cuda()
        total_words = torch.Tensor([0]).cuda()

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        with torch.no_grad():
            while not data_iterator.end_of_epoch():
                samples = next(epoch_iterator)

                batch = prepare_sample(samples, device=self.device, fp16=self.opt.fp16 and not self.opt.fp16_mixed)

                # calling model.forward() here

                total_loss.add_(0)
                total_words.add_(batch.tgt_size)
                i = i + 1

        # allreduce the total loss and total words from other processes
        dist.all_reduce(total_loss, op=dist.reduce_op.SUM, group=self.group)
        dist.all_reduce(total_words, op=dist.reduce_op.SUM, group=self.group)

        return total_loss / total_words

    def train_epoch(self, epoch, resume=False, itr_progress=None):

        global rec_ppl
        opt = self.opt
        train_data = self.train_data
        streaming = opt.streaming

        # Clear the gradients of the model
        self.model_wrapper.zero_grad()
        # self.model.reset_states()

        dataset = train_data
        data_iterator = generate_data_iterator(dataset, seed=self.opt.seed, num_workers=opt.num_workers,
                                               epoch=epoch, buffer_size=opt.buffer_size)

        # TODO: fix resume which is currently buggy
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
            # TODO: move everything to the multiGPU trainer
            samples = next(epoch_iterator)
            
            batch = prepare_sample(samples, device=self.device, fp16=self.opt.fp16 and not self.opt.fp16_mixed)

            # MODEL FORWARD here

            if grad_scaler == -1:
                grad_scaler = 1

            oom = False

            logging_output = self.model_wrapper.forward(samples, eval=False, backward=True)
            batch_size = logging_output['batch_size']

            # loss_data = logging_output['loss']
            # rev_loss_data = logging_output['rev_loss_data']
            # mirror_loss_data = logging_output['mirror_loss_data']
            # rec_loss_data = logging_output['rec_loss_data']
            #
            # # if not oom:
            # src_size = logging_output['src_size']
            # tgt_size = logging_output['tgt_size']

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
                    # normalize_gradients(amp.master_params(optimizer), grad_denom)
                    # Update the parameters.
                    self.model_wrapper.update_parameters(grad_denom=grad_denom)
                    self.model_wrapper.zero_grad()
                    # self.optim.step()
                    # self.optim.zero_grad()
                    # self.model.zero_grad()
                    counter = 0
                    num_accumulated_words = 0
                    num_accumulated_sents = 0
                    grad_scaler = -1
                    num_updates = self.model_wrapper.get_optim()._step

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
            # total_tokens += batch.get('target_output').nelement()
            # total_non_pads += batch.get('target_output').ne(onmt.constants.PAD).sum().item()
            optim = self.model_wrapper.get_optim()
            # batch_efficiency = total_non_pads / total_tokens

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
        # model = self.model
        # optim = self.optim

        if checkpoint is not None:
            raise NotImplementedError

            self.model.load_state_dict(checkpoint['model'])
            prec_opt = checkpoint['opt'] if 'opt' in checkpoint else None

            opt.reset_optim = True

            if not opt.reset_optim:

                # print("* Loading optimizer states ... ")
                # self.optim.load_state_dict(checkpoint['optim'])
                # if prec_opt is not None and hasattr(prec_opt, "fp16_mixed"):
                #     # Only load amp information if the mode is the same
                #     # Maybe its better to change between optimization mode?
                #     if opt.fp16_mixed == prec_opt.fp16_mixed and opt.fp16 == prec_opt.fp16:
                #         if 'amp' in checkpoint:
                #             amp.load_state_dict(checkpoint['amp'])
                #
                # # Only load the progress when we use the same optimizer
                # if 'itr' in checkpoint:
                #     itr_progress = checkpoint['itr']
                # else:
                #     itr_progress = None
                #
                # resume = True
                # start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 1
                # if start_epoch is None:
                #     start_epoch = 1
                self.dummy = 'dummy'
            else:
                itr_progress = None
                resume = False
                start_epoch = 1

            del checkpoint['model']
            optim_state_dict = checkpoint['optim']
            # del checkpoint['optim']
            del checkpoint
        else:
            itr_progress = None
            print('Initializing model parameters')
            # init_model_parameters(model, opt)
            resume = False
            start_epoch = 1
            optim_state_dict = None

        # if opt.load_encoder_from:
        #     self.load_encoder_weight(opt.load_encoder_from)
        #
        # if opt.load_decoder_from:
        #     self.load_decoder_weight(opt.load_decoder_from)

        # if we are on a GPU: warm up the memory allocator
        # if self.cuda:
        #     self.warm_up()

        valid_loss = self.eval(self.valid_data)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)

        self.start_time = time.time()

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


