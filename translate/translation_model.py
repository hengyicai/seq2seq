# coding=utf-8
import tensorflow as tf
import os
import pickle
import re
import time
import numpy as np
import sys
import math
import shutil
import itertools
from collections import OrderedDict
from translate import utils, evaluation
from translate.seq2seq_model import Seq2SeqModel
from subprocess import Popen, PIPE


class TranslationModel:
    def __init__(self, encoders, decoders, checkpoint_dir, learning_rate, learning_rate_decay_factor,
                 batch_size, keep_best=1, dev_prefix=None, name=None, ref_ext=None,
                 pred_edits=False, dual_output=False, binary=None, truncate_lines=True, ensemble=False,
                 checkpoints=None, beam_size=1, len_normalization=1, lexicon=None, **kwargs):

        self.batch_size = batch_size
        self.character_level = {}
        self.binary = []

        for encoder_or_decoder in encoders + decoders:
            encoder_or_decoder.ext = encoder_or_decoder.ext or encoder_or_decoder.name
            self.character_level[encoder_or_decoder.ext] = encoder_or_decoder.character_level
            self.binary.append(encoder_or_decoder.get('binary', False))

        self.encoders, self.decoders = encoders, decoders

        self.char_output = decoders[0].character_level

        self.src_ext = [encoder.ext for encoder in encoders]
        self.trg_ext = [decoder.ext for decoder in decoders]

        self.extensions = self.src_ext + self.trg_ext

        self.ref_ext = ref_ext
        if self.ref_ext is not None:
            self.binary.append(False)

        self.pred_edits = pred_edits
        self.dual_output = dual_output

        self.dev_prefix = dev_prefix
        self.name = name

        self.max_input_len = [encoder.max_len for encoder in encoders]
        self.max_output_len = [decoder.max_len for decoder in decoders]

        if truncate_lines:
            self.max_len = None  # we let seq2seq.get_batch handle long lines (by truncating them)
        else:  # the line reader will drop lines that are too long
            self.max_len = dict(zip(self.extensions, self.max_input_len + self.max_output_len))

        self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate', dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        with tf.device('/cpu:0'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.baseline_step = tf.Variable(0, trainable=False, name='baseline_step')

        self.filenames = utils.get_filenames(extensions=self.extensions, dev_prefix=dev_prefix, name=name,
                                             ref_ext=ref_ext, binary=self.binary, **kwargs)
        utils.debug('reading vocabularies')
        self.vocabs = None
        self.src_vocab, self.trg_vocab = None, None
        self.read_vocab()

        for encoder_or_decoder, vocab in zip(encoders + decoders, self.vocabs):
            if vocab:
                if encoder_or_decoder.vocab_size:  # reduce vocab size
                    vocab.reverse[:] = vocab.reverse[:encoder_or_decoder.vocab_size]
                    for token, token_id in list(vocab.vocab.items()):
                        if token_id >= encoder_or_decoder.vocab_size:
                            del vocab.vocab[token]
                else:
                    encoder_or_decoder.vocab_size = len(vocab.reverse)

        utils.debug('creating model')

        self.models = []
        if ensemble and checkpoints is not None:
            for i, _ in enumerate(checkpoints, 1):
                with tf.variable_scope('model_{}'.format(i)):
                    model = Seq2SeqModel(encoders, decoders, self.learning_rate, self.global_step, name=name,
                                         pred_edits=pred_edits, dual_output=dual_output,
                                         baseline_step=self.baseline_step, **kwargs)
                    self.models.append(model)
            self.seq2seq_model = self.models[0]
        else:
            self.seq2seq_model = Seq2SeqModel(encoders, decoders, self.learning_rate, self.global_step, name=name,
                                              pred_edits=pred_edits, dual_output=dual_output,
                                              baseline_step=self.baseline_step, **kwargs)
            self.models.append(self.seq2seq_model)

        self.seq2seq_model.create_beam_op(self.models, beam_size, len_normalization)

        self.batch_iterator = None
        self.dev_batches = None
        self.train_size = None
        self.saver = None
        self.keep_best = keep_best
        self.checkpoint_dir = checkpoint_dir
        self.epoch = None

        self.training = utils.AttrDict()  # used to keep track of training

        if lexicon:
            with open(lexicon) as lexicon_file:
                self.lexicon = dict(line.split() for line in lexicon_file)
        else:
            self.lexicon = None

    def read_data(self, max_train_size, max_dev_size, read_ahead=10, batch_mode='standard', shuffle=True,
                  crash_test=False, **kwargs):
        utils.debug('reading training data')
        self.batch_iterator, self.train_size = utils.get_batch_iterator(
            self.filenames.train, self.extensions, self.vocabs, self.batch_size,
            max_size=max_train_size, character_level=self.character_level, max_seq_len=self.max_len,
            read_ahead=read_ahead, mode=batch_mode, shuffle=shuffle, binary=self.binary, crash_test=crash_test
        )

        utils.debug('reading development data')

        dev_sets = [
            utils.read_dataset(dev, self.extensions, self.vocabs, max_size=max_dev_size,
                               character_level=self.character_level, binary=self.binary)[0]
            for dev in self.filenames.dev
        ]
        # subset of the dev set whose loss is periodically evaluated
        self.dev_batches = [utils.get_batches(dev_set, batch_size=self.batch_size) for dev_set in dev_sets]

    def read_vocab(self):
        # don't try reading vocabulary for encoders that take pre-computed features
        # [({key:val, key:val,....}, [word, word, ...., word]), (), (), () ,....., ()]
        self.vocabs = [
            None if binary else utils.initialize_vocabulary(vocab_path)
            for vocab_path, binary in zip(self.filenames.vocab, self.binary)
        ]
        # trg_vocab: [(name1: {key:val, key:val,....}, name2:[word, word, ...., word])]
        self.src_vocab, self.trg_vocab = self.vocabs[:len(self.src_ext)], self.vocabs[len(self.src_ext):]

    def decode_sentence(self, sentence_tuple, remove_unk=False):
        return next(self.decode_batch([sentence_tuple], remove_unk))

    def decode_batch(self, sentence_tuples, batch_size, remove_unk=False, fix_edits=True, unk_replace=False,
                     align=False, reverse=False, output=None):
        utils.log("start decode batch")
        if batch_size == 1:
            batches = ([sentence_tuple] for sentence_tuple in sentence_tuples)  # lazy
        else:
            batch_count = int(math.ceil(len(sentence_tuples) / batch_size))
            batches = [sentence_tuples[i * batch_size:(i + 1) * batch_size] for i in range(batch_count)]

        def map_to_ids(sentence_tuple):
            token_ids = [
                sentence if vocab is None else
                utils.sentence_to_token_ids(sentence, vocab.vocab, character_level=self.character_level.get(ext))
                for ext, vocab, sentence in zip(self.extensions, self.vocabs, sentence_tuple)
            ]
            return token_ids

        line_id = 0
        for batch_id, batch in enumerate(batches):
            token_ids = list(map(map_to_ids, batch))
            batch_token_ids, batch_weights = self.seq2seq_model.greedy_decoding(token_ids, align=unk_replace or align)
            utils.log("batch_token_ids")
            utils.log(batch_token_ids)
            utils.log(len(batch_token_ids))
            utils.log(len(batch_token_ids[0]))
            utils.log(len(batch_token_ids[0][0]))
            utils.log(len(batch_token_ids[0][0][0]))
            batch_token_ids = zip(*batch_token_ids)

            for sentence_id, (src_tokens, trg_token_ids) in enumerate(zip(batch, batch_token_ids)):
                # trg_token_ids, shape(64,10,50), [[[....50num....],[....50num....],[....50num....],....,[....50num....]]]
                line_id += 1

                trg_tokens = []

                # for single_trg_token_id in trg_token_ids:
                # single_trg_token_id, shape(50), [....50num....]
                for trg_token_ids_, vocab in zip(trg_token_ids, self.trg_vocab):
                    # trg_token_ids_, shape(10,50)
                    top_10_trg_tokens = []
                    for single_trg_token_ids in trg_token_ids_:
                        # single_trg_token_ids, [,,,,,,,] 50 nums
                        single_trg_token_ids = list(single_trg_token_ids)
                        if utils.EOS_ID in single_trg_token_ids:
                            single_trg_token_ids = single_trg_token_ids[:single_trg_token_ids.index(utils.EOS_ID)]
                        single_trg_token_ids = [vocab.reverse[i] if i < len(vocab.reverse) else utils._UNK
                                                for i in single_trg_token_ids]
                        top_10_trg_tokens.append(single_trg_token_ids)

                    # trg_token_ids_ = list(trg_token_ids_)  # from np array to list
                    # if utils.EOS_ID in trg_token_ids_:
                    #     trg_token_ids_ = trg_token_ids_[:trg_token_ids_.index(utils.EOS_ID)]
                    #
                    # trg_tokens_ = [vocab.reverse[i] if i < len(vocab.reverse) else utils._UNK
                    #            for i in trg_token_ids_]
                    # trg_tokens.append(trg_tokens_)
                    trg_tokens.append(top_10_trg_tokens)
                    # trg_tokens, shape(64, 10, ?)
                #   beam_trg_tokens.append(trg_tokens)
                #   trg_tokens = []

                if align:
                    weights_ = batch_weights[sentence_id].squeeze()
                    max_len_ = weights_.shape[1]
                    src_tokens_ = src_tokens[0].split()[:max_len_ - 1] + [utils._EOS]
                    src_tokens_ = [token if token in self.src_vocab[0].vocab else utils._UNK for token in src_tokens_]
                    trg_tokens_ = trg_tokens[0][0][:weights_.shape[0] - 1] + [utils._EOS]

                    weights_ = weights_[:len(trg_tokens_), :len(src_tokens_)]
                    output_file = output and '{}.{}.pdf'.format(output, line_id)
                    utils.heatmap(src_tokens_, trg_tokens_, weights_, reverse=reverse, output_file=output_file)

                if unk_replace:
                    weights = batch_weights[sentence_id]
                    src_words = src_tokens[0].split()
                    align_ids = np.argmax(weights[:, :len(src_words)], axis=1)

                    def replace(token, align_id):
                        if token == utils._UNK:
                            token = src_words[align_id]
                            if not token[0].isupper() and self.lexicon is not None and token in self.lexicon:
                                token = self.lexicon[token]
                        return token

                    for i in range(len(trg_tokens[0])):
                        trg_tokens[0][i] = [replace(token, align_id) for align_id, token in
                                            zip(align_ids, trg_tokens[0][i])]

                #########################################################################
                if self.pred_edits:
                    # first output is ops, second output is words
                    raw_hypothesis = ' '.join('_'.join(tokens) for tokens in zip(*trg_tokens))
                    src_words = src_tokens[0].split()
                    trg_tokens = utils.reverse_edits(src_words, trg_tokens, fix=fix_edits)
                    trg_tokens = [token for token in trg_tokens if token not in utils._START_VOCAB]
                    # FIXME: char-level
                else:
                    trg_tokens = trg_tokens[0]
                    raw_hypothesis = []
                    for single_trg_tokens in trg_tokens:
                        single_raw_hypothesis = ''.join(single_trg_tokens) if self.char_output else ' '.join(
                            single_trg_tokens)
                        raw_hypothesis.append(single_raw_hypothesis)
                    # raw_hypothesis = ''.join(trg_tokens) if self.char_output else ' '.join(trg_tokens)

                if remove_unk:
                    for i in range(len(trg_tokens)):
                        trg_tokens[i] = [token for token in trg_tokens[i] if token != utils._UNK]

                if self.char_output:
                    hypothesis = []
                    for i in range(len(trg_tokens)):
                        hypothesis.append(''.join(trg_tokens[i]))
                    # hypothesis = ''.join(trg_tokens)
                else:
                    hypothesis = []
                    for i in range(len(trg_tokens)):
                        hypothesis.append(' '.join(trg_tokens[i]).replace('@@ ', ''))
                    # hypothesis = ' '.join(trg_tokens).replace('@@ ', '')  # merge subwords units

                yield hypothesis, raw_hypothesis

    def align(self, output=None, align_encoder_id=0, reverse=False, **kwargs):
        if len(self.filenames.test) != len(self.extensions):
            raise Exception('wrong number of input files')

        binary = self.binary and any(self.binary)

        paths = self.filenames.test or [None]
        lines = utils.read_lines(paths, binary=self.binary)

        for line_id, lines in enumerate(lines):
            token_ids = [
                sentence if vocab is None else
                utils.sentence_to_token_ids(sentence, vocab.vocab, character_level=self.character_level.get(ext))
                for ext, vocab, sentence in zip(self.extensions, self.vocabs, lines)
            ]

            _, weights = self.seq2seq_model.step(data=[token_ids], align=True, update_model=False)

            trg_vocab = self.trg_vocab[0]
            trg_token_ids = token_ids[len(self.src_ext)]
            trg_tokens = [trg_vocab.reverse[i] if i < len(trg_vocab.reverse) else utils._UNK for i in trg_token_ids]

            weights = weights.squeeze()
            max_len = weights.shape[1]

            if binary:
                src_tokens = None
            else:
                src_tokens = lines[align_encoder_id].split()[:max_len - 1] + [utils._EOS]
            trg_tokens = trg_tokens[:weights.shape[0] - 1] + [utils._EOS]

            output_file = output and '{}.{}.pdf'.format(output, line_id + 1)

            utils.heatmap(src_tokens, trg_tokens, weights, output_file=output_file, reverse=reverse)

    def decode(self, output=None, remove_unk=False, raw_output=False, max_test_size=None, unk_replace=False,
               align=False, reverse=False, **kwargs):
        utils.log('starting decoding')

        # empty `test` means that we read from standard input, which is not possible with multiple encoders
        # assert len(self.src_ext) == 1 or self.filenames.test
        # check that there is the right number of files for decoding
        # assert not self.filenames.test or len(self.filenames.test) == len(self.src_ext)

        output_file = None
        try:
            output_file = sys.stdout if output is None else open(output, 'w')
            paths = self.filenames.test or [None]
            lines = utils.read_lines(paths, binary=self.binary)

            if max_test_size:
                lines = itertools.islice(lines, max_test_size)

            if not self.filenames.test:  # interactive mode
                batch_size = 1
            else:
                batch_size = self.batch_size
                lines = list(lines)

            hypothesis_iter = self.decode_batch(lines, batch_size, remove_unk=remove_unk, unk_replace=unk_replace,
                                                align=align, reverse=reverse, output=output)

            for hypothesis, raw in hypothesis_iter:
                if raw_output:
                    hypothesis = raw

                output_file.write(hypothesis + '\n')
                output_file.flush()
        finally:
            if output_file is not None:
                output_file.close()

    def evaluate(self, score_functions, on_dev=True, output=None, remove_unk=False, max_dev_size=None,
                 raw_output=False, fix_edits=True, max_test_size=None, post_process_script=None,
                 unk_replace=False, **kwargs):
        """
        Decode a dev or test set, and perform evaluation with respect to gold standard, using the provided
        scoring function. If `output` is defined, also save the decoding output to this file.
        When evaluating development data (`on_dev` to True), several dev sets can be specified (`dev_prefix` parameter
        in configuration files), and a score is computed for each of them.

        :param score_function: name of the scoring function used to score and rank models (typically 'bleu_score')
        :param on_dev: if True, evaluate the dev corpus, otherwise evaluate the test corpus
        :param output: save the hypotheses to this file
        :param remove_unk: remove the UNK symbols from the output
        :param max_dev_size: maximum number of lines to read from dev files
        :param max_test_size: maximum number of lines to read from test files
        :param raw_output: save raw decoder output (don't do post-processing like UNK deletion or subword
            concatenation). The evaluation is still done with the post-processed output.
        :param fix_edits: when predicting edit operations, pad shorter hypotheses with KEEP symbols.
        :return: scores of each corpus to evaluate
        """
        utils.log('starting evaluation')

        if on_dev:
            filenames = self.filenames.dev
        else:
            filenames = [self.filenames.test]

        # convert `output` into a list, for zip
        if isinstance(output, str):
            output = [output]
        elif output is None:
            output = [None] * len(filenames)

        scores = []
        utils.log('show output')
        utils.log(output)

        # evaluation on multiple corpora
        for dev_id, (filenames_, output_, prefix) in enumerate(zip(filenames, output, self.dev_prefix)):
            utils.log('filenames, output, self.dev_prefix')
            utils.log(filenames)
            utils.log(output)

            if self.dev_batches:
                dev_batches = self.dev_batches[dev_id]
                dev_loss = sum(self.seq2seq_model.step(batch, update_model=False).loss * len(batch)
                               for batch in dev_batches)
                dev_loss /= sum(map(len, dev_batches))
            else:  # TODO
                dev_loss = 0

            extensions = list(self.extensions)
            if self.ref_ext is not None:
                extensions.append(self.ref_ext)

            lines = list(utils.read_lines(filenames_, binary=self.binary))

            if on_dev and max_dev_size:
                lines = lines[:max_dev_size]
            elif not on_dev and max_test_size:
                lines = lines[:max_test_size]

            hypotheses = []
            references = []
            utils.log("making hypotheses")
            output_file = None
            try:
                if output_ is not None:
                    output_file = open(output_, 'w', encoding='utf-8')

                lines_ = list(zip(*lines))

                src_sentences = list(zip(*lines_[:len(self.src_ext)]))
                trg_sentences = list(zip(*lines_[len(self.src_ext):]))

                utils.log("making decode_batch")
                hypothesis_iter = self.decode_batch(lines, self.batch_size, remove_unk=remove_unk,
                                                    fix_edits=fix_edits, unk_replace=unk_replace)

                for i, (sources, hypothesis, reference) in enumerate(zip(src_sentences, hypothesis_iter,
                                                                         trg_sentences)):
                    if self.ref_ext is not None and on_dev:
                        reference = reference[-1]
                    else:
                        reference = reference[0]  # single output for now

                    hypothesis, raw = hypothesis
                    # hypothesis: [10items],each item is a "token sequence"
                    hypotheses.append(hypothesis)
                    references.append(reference.strip().replace('@@ ', ''))

                    if output_file is not None:
                        if raw_output:
                            hypothesis = raw
                        line = "source:\t" + sources + "\nref:\t" + reference + "\n"
                        for item in hypothesis:
                            line += item + '\n'
                        line += "\n"
                        # line = hypothesis + '\n'
                        output_file.write(line)
                        output_file.flush()

            finally:
                if output_file is not None:
                    output_file.close()

            if post_process_script is not None:
                data = '\n'.join(hypotheses).encode()
                data = Popen([post_process_script], stdout=PIPE, stdin=PIPE).communicate(input=data)[0].decode()
                hypotheses = data.splitlines()

            scores_ = []
            summary = None

            for score_function in score_functions:
                try:
                    if score_function == 'loss':
                        score = dev_loss
                        reversed_ = True
                    else:
                        fun = getattr(evaluation, 'corpus_' + score_function)
                        try:
                            reversed_ = fun.reversed
                        except AttributeError:
                            reversed_ = False

                        func_arg = []
                        for item in hypotheses:
                            func_arg.append(item[0])
                        score, score_summary = fun(func_arg, references)
                        summary = summary or score_summary

                    scores_.append((score_function, score, reversed_))
                except:
                    pass

            score_info = ['{}={:.2f}'.format(key, value) for key, value, _ in scores_]
            score_info.insert(0, prefix)
            if summary:
                score_info.append(summary)

            if self.name is not None:
                score_info.insert(0, self.name)

            utils.log(' '.join(map(str, score_info)))

            # main score
            _, score, reversed_ = scores_[0]
            scores.append(-score if reversed_ else score)

        return scores

    def train(self, baseline_steps=0, loss_function='xent', use_baseline=True, **kwargs):
        self.init_training(**kwargs)

        if (loss_function == 'reinforce' and use_baseline and baseline_steps > 0 and
                self.baseline_step.eval() < baseline_steps):
            utils.log('pre-training reinforce baseline')
            for i in range(baseline_steps - self.baseline_step.eval()):
                self.seq2seq_model.reinforce_step(next(self.batch_iterator), update_model=False,
                                                  use_sgd=False, update_baseline=True)

        utils.log('starting training')
        while True:
            try:
                self.train_step(loss_function=loss_function, use_baseline=use_baseline, **kwargs)
            except (utils.FinishedTrainingException, KeyboardInterrupt):
                utils.log('exiting...')
                self.save()
                return
            except utils.EvalException:
                self.save()
                step, score = self.training.scores[-1]
                self.manage_best_checkpoints(step, score)
            except utils.CheckpointException:
                self.save()

    def init_training(self, sgd_after_n_epoch=None, **kwargs):
        self.read_data(**kwargs)
        self.epoch = self.batch_size * self.global_step // self.train_size

        global_step = self.global_step.eval()
        epoch = self.epoch.eval()
        if sgd_after_n_epoch is not None and epoch >= sgd_after_n_epoch:  # already switched to SGD
            self.training.use_sgd = True
        else:
            self.training.use_sgd = False

        if kwargs.get('batch_mode') != 'random' and not kwargs.get('shuffle'):
            # read all the data up to this step (only if the batch iteration method is deterministic)
            for _ in range(global_step):
                next(self.batch_iterator)

        # those parameters are used to track the progress of training
        self.training.time = 0
        self.training.steps = 0
        self.training.loss = 0
        self.training.baseline_loss = 0
        self.training.losses = []
        self.training.last_decay = global_step
        self.training.scores = []

    def train_step(self, steps_per_checkpoint, model_dir, steps_per_eval=None, max_steps=0,
                   max_epochs=0, eval_burn_in=0, decay_if_no_progress=None, decay_after_n_epoch=None,
                   decay_every_n_epoch=None, sgd_after_n_epoch=None, sgd_learning_rate=None, min_learning_rate=None,
                   loss_function='xent', use_baseline=True, **kwargs):
        if min_learning_rate is not None and self.learning_rate.eval() < min_learning_rate:
            utils.debug('learning rate is too small: stopping')
            raise utils.FinishedTrainingException
        if 0 < max_steps <= self.global_step.eval() or 0 < max_epochs <= self.epoch.eval():
            utils.log("max_steps")
            utils.log(max_steps)
            utils.log("self.global_step.eval()")
            utils.log(self.global_step.eval())
            utils.log("max_epochs")
            utils.log(max_epochs)
            utils.log("self.epoch.eval()")
            utils.log(self.epoch.eval())
            raise utils.FinishedTrainingException

        start_time = time.time()

        if loss_function == 'reinforce':
            step_function = self.seq2seq_model.reinforce_step
        else:
            step_function = self.seq2seq_model.step

        next_data = next(self.batch_iterator)
        res = step_function(next_data, update_model=True, use_sgd=self.training.use_sgd, align=True,
                            update_baseline=True)

        self.training.loss += res.loss
        self.training.baseline_loss += getattr(res, 'baseline_loss', 0)
        utils.log("weight_juzhen")
        utils.log(res.weights)
        utils.log(len(res.weights))
        utils.log(len(res.weights[0]))
        utils.log(len(res.weights[0][0]))
        utils.log(sum(res.weights[0][0]))
        utils.log("next_data")
        utils.log(next_data)
        # self.atten_weight = res.weight

        self.training.time += time.time() - start_time
        self.training.steps += 1

        global_step = self.global_step.eval()
        epoch = self.epoch.eval()

        if decay_after_n_epoch is not None and self.batch_size * global_step >= decay_after_n_epoch * self.train_size:
            if decay_every_n_epoch is not None and (self.batch_size * (global_step - self.training.last_decay)
                                                    >= decay_every_n_epoch * self.train_size):
                self.learning_rate_decay_op.eval()
                utils.debug('  decaying learning rate to: {:.3g}'.format(self.learning_rate.eval()))
                self.training.last_decay = global_step

        if sgd_after_n_epoch is not None and epoch >= sgd_after_n_epoch:
            if not self.training.use_sgd:
                utils.debug('epoch {}, starting to use SGD'.format(epoch + 1))
                self.training.use_sgd = True
                if sgd_learning_rate is not None:
                    self.learning_rate.assign(sgd_learning_rate).eval()
                self.training.last_decay = global_step  # reset learning rate decay

        # if steps_per_checkpoint and global_step % steps_per_checkpoint == 0:
        #    symmary = 'attention weight: {}'.format(self.atten_weight)
        #    utils.log(summary)

        if steps_per_checkpoint and global_step % steps_per_checkpoint == 0:
            loss = self.training.loss / self.training.steps
            baseline_loss = self.training.baseline_loss / self.training.steps
            step_time = self.training.time / self.training.steps

            summary = 'step {} epoch {} learning rate {:.3g} step-time {:.3f} loss {:.3f}'.format(
                global_step, epoch + 1, self.learning_rate.eval(), step_time, loss)

            if self.name is not None:
                summary = '{} {}'.format(self.name, summary)
            if use_baseline and loss_function == 'reinforce':
                summary = '{} baseline-loss {:.4f}'.format(summary, baseline_loss)

            utils.log(summary)

            if decay_if_no_progress and len(self.training.losses) >= decay_if_no_progress:
                if loss >= max(self.training.losses[:decay_if_no_progress]):
                    self.learning_rate_decay_op.eval()

            self.training.losses.append(loss)
            self.training.loss, self.training.time, self.training.steps, self.training.baseline_loss = 0, 0, 0, 0

        if steps_per_eval and global_step % steps_per_eval == 0 and 0 <= eval_burn_in <= global_step:
            eval_dir = 'eval' if self.name is None else 'eval_{}'.format(self.name)
            eval_output = os.path.join(model_dir, eval_dir)

            os.makedirs(eval_output, exist_ok=True)

            # if there are several dev files, we define several output files
            output = [
                os.path.join(eval_output, '{}.{}.out'.format(prefix, global_step))
                for prefix in self.dev_prefix
            ]

            kwargs_ = dict(kwargs)
            kwargs_['output'] = output
            score, *_ = self.evaluate(on_dev=True, **kwargs_)
            self.training.scores.append((global_step, score))

        if steps_per_eval and global_step % steps_per_eval == 0:
            raise utils.EvalException
        elif steps_per_checkpoint and global_step % steps_per_checkpoint == 0:
            raise utils.CheckpointException

    def manage_best_checkpoints(self, step, score):
        score_filename = os.path.join(self.checkpoint_dir, 'scores.txt')
        # try loading previous scores
        try:
            with open(score_filename) as f:
                # list of pairs (score, step)
                scores = [(float(line.split()[0]), int(line.split()[1])) for line in f]
        except IOError:
            scores = []

        if any(step_ >= step for _, step_ in scores):
            utils.warn('inconsistent scores.txt file')

        best_scores = sorted(scores, reverse=True)[:self.keep_best]

        def full_path(filename):
            return os.path.join(self.checkpoint_dir, filename)

        if any(score_ < score for score_, _ in best_scores) or not best_scores:
            # if this checkpoint is in the top, save it under a special name

            prefix = 'translate-{}.'.format(step)
            dest_prefix = 'best-{}.'.format(step)

            absolute_best = all(score_ < score for score_, _ in best_scores)
            if absolute_best:
                utils.log('new best model')

            for filename in os.listdir(self.checkpoint_dir):
                if filename.startswith(prefix):
                    dest_filename = filename.replace(prefix, dest_prefix)
                    shutil.copy(full_path(filename), full_path(dest_filename))

                    # also copy to `best` if this checkpoint is the absolute best
                    if absolute_best:
                        dest_filename = filename.replace(prefix, 'best.')
                        shutil.copy(full_path(filename), full_path(dest_filename))

            best_scores = sorted(best_scores + [(score, step)], reverse=True)

            for _, step_ in best_scores[self.keep_best:]:
                # remove checkpoints that are not in the top anymore
                prefix = 'best-{}'.format(step_)
                for filename in os.listdir(self.checkpoint_dir):
                    if filename.startswith(prefix):
                        os.remove(full_path(filename))

        # save scores
        scores.append((score, step))

        with open(score_filename, 'w') as f:
            for score_, step_ in scores:
                f.write('{:.2f} {}\n'.format(score_, step_))

    def initialize(self, checkpoints=None, reset=False, reset_learning_rate=False, max_to_keep=1,
                   keep_every_n_hours=0, sess=None, **kwargs):
        """
        :param checkpoints: list of checkpoints to load (instead of latest checkpoint)
        :param reset: don't load latest checkpoint, reset learning rate and global step
        :param reset_learning_rate: reset the learning rate to its initial value
        :param max_to_keep: keep this many latest checkpoints at all times
        :param keep_every_n_hours: and keep checkpoints every n hours
        """
        sess = sess or tf.get_default_session()

        if keep_every_n_hours <= 0 or keep_every_n_hours is None:
            keep_every_n_hours = float('inf')

        self.saver = tf.train.Saver(max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=keep_every_n_hours,
                                    sharded=False)

        sess.run(tf.global_variables_initializer())

        # load pre-trained embeddings
        for encoder_or_decoder, vocab in zip(self.encoders + self.decoders, self.vocabs):
            if encoder_or_decoder.embedding_file:
                utils.log('loading embeddings from: {}'.format(encoder_or_decoder.embedding_file))
                embeddings = {}
                with open(encoder_or_decoder.embedding_file) as embedding_file:
                    for line in embedding_file:
                        word, vector = line.split(' ', 1)
                        if word in vocab.vocab:
                            embeddings[word] = np.array(list(map(float, vector.split())))
                # standardize (mean of 0, std of 0.01)
                mean = sum(embeddings.values()) / len(embeddings)
                std = np.sqrt(sum((value - mean) ** 2 for value in embeddings.values())) / (len(embeddings) - 1)
                for key in embeddings:
                    embeddings[key] = 0.01 * (embeddings[key] - mean) / std

                # change TensorFlow variable's value
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    embedding_var = tf.get_variable('embedding_' + encoder_or_decoder.name)
                    embedding_value = embedding_var.eval()
                    for word, i in vocab.vocab.items():
                        if word in embeddings:
                            embedding_value[i] = embeddings[word]
                    sess.run(embedding_var.assign(embedding_value))

        blacklist = ['dropout_keep_prob']

        if reset_learning_rate or reset:
            blacklist.append('learning_rate')
        if reset:
            blacklist.append('global_step')

        params = {k: kwargs.get(k) for k in ('variable_mapping', 'reverse_mapping')}

        if checkpoints and len(self.models) > 1:
            assert len(self.models) == len(checkpoints)
            for i, checkpoint in enumerate(checkpoints, 1):
                load_checkpoint(sess, None, checkpoint, blacklist=blacklist, prefix='model_{}'.format(i), **params)
        elif checkpoints:  # load partial checkpoints
            for checkpoint in checkpoints:  # checkpoint files to load
                load_checkpoint(sess, None, checkpoint, blacklist=blacklist, **params)
        elif not reset:
            load_checkpoint(sess, self.checkpoint_dir, blacklist=blacklist, **params)

        utils.debug('global step: {}'.format(self.global_step.eval()))
        utils.debug('baseline step: {}'.format(self.baseline_step.eval()))

    def save(self):
        save_checkpoint(tf.get_default_session(), self.saver, self.checkpoint_dir, self.global_step)


# hard-coded variables which can also be defined in config file (variable_mapping and reverse_mapping)
global_variable_mapping = []  # map old names to new names
global_reverse_mapping = [  # map new names to old names
    (r'decoder_(.*?)/.*/initial_state_projection/', r'decoder_\1/initial_state_projection/'),
]


def load_checkpoint(sess, checkpoint_dir, filename=None, blacklist=(), prefix=None, variable_mapping=None,
                    reverse_mapping=None):
    """
    if `filename` is None, we load last checkpoint, otherwise
      we ignore `checkpoint_dir` and load the given checkpoint file.
    """
    variable_mapping = variable_mapping or []
    reverse_mapping = reverse_mapping or []

    variable_mapping = list(variable_mapping) + global_variable_mapping
    reverse_mapping = list(reverse_mapping) + global_reverse_mapping

    if filename is None:
        # load last checkpoint
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt is not None:
            filename = ckpt.model_checkpoint_path
    else:
        checkpoint_dir = os.path.dirname(filename)

    vars_ = []
    var_names = []
    for var in tf.global_variables():
        if prefix is None or var.name.startswith(prefix):
            name = var.name if prefix is None else var.name[len(prefix) + 1:]
            vars_.append(var)
            var_names.append(name)

    var_file = os.path.join(checkpoint_dir, 'vars.pkl')
    if os.path.exists(var_file):
        with open(var_file, 'rb') as f:
            old_names = pickle.load(f)
    else:
        old_names = list(var_names)

    name_mapping = {}
    for name in old_names:
        name_ = name
        for key, value in variable_mapping:
            name_ = re.sub(key, value, name_)
        name_mapping[name] = name_

    var_names_ = []
    for name in var_names:
        name_ = name
        for key, value in reverse_mapping:
            name_ = re.sub(key, value, name_)
        if name_ in list(name_mapping.values()):
            name = name_
        var_names_.append(name)
    vars_ = dict(zip(var_names_, vars_))

    variables = {old_name[:-2]: vars_[new_name] for old_name, new_name in name_mapping.items()
                 if new_name in vars_ and not any(prefix in new_name for prefix in blacklist)}

    if filename is not None:
        utils.log('reading model parameters from {}'.format(filename))
        tf.train.Saver(variables).restore(sess, filename)

        utils.debug('retrieved parameters ({})'.format(len(variables)))
        for var in sorted(variables.values(), key=lambda var: var.name):
            utils.debug('  {} {}'.format(var.name, var.get_shape()))


def save_checkpoint(sess, saver, checkpoint_dir, step=None, name=None):
    var_file = os.path.join(checkpoint_dir, 'vars.pkl')
    name = name or 'translate'
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(var_file, 'wb') as f:
        var_names = [var.name for var in tf.global_variables()]
        pickle.dump(var_names, f)

    utils.log('saving model to {}'.format(checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, name)
    saver.save(sess, checkpoint_path, step, write_meta_graph=False)

    utils.log('finished saving model')
