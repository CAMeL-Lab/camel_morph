#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
from tqdm import tqdm
from edit_distance import SequenceMatcher

import numpy as np


def pad_sents_char(sents, char_pad_token, max_sent_length=None, max_word_length=None):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()`
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = max_word_length + \
        1 if max_word_length else max(len(w) for s in sents for w in s)

    sents_padded = []
    lengths = [len(s) for s in sents]
    maxlen = max_sent_length + 1 if max_sent_length else max(lengths)
    for i, sent in enumerate(sents):
        sent_tokens = [w + [char_pad_token] * (max_word_length-len(w))
                       if len(w) < max_word_length else w[:max_word_length] for w in sent]
        pad_tokens = [[char_pad_token] * max_word_length] * (maxlen-lengths[i])
        sents_padded.append(sent_tokens + pad_tokens)

    return sents_padded


def pad_sents(sents, pad_token, maxlen):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []
    lengths = [len(s) for s in sents]
    maxlen += 1
    for i, sent in enumerate(sents):
        sents_padded.append(sent + [pad_token] * (maxlen-lengths[i]))

    return sents_padded


def read_corpus(file_path, tokenizer=None):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        data.append(line.strip())
    if tokenizer is not None:
        data[0] = tokenizer(data[0], padding=True)
    return data


class AlignmentHandler:
    def __init__(self, already_split, pad_label=None, eos_label=None, n=2):
        self.already_split = already_split
        self.pad_label = pad_label
        self.eos_label = eos_label
        self.n = n

    def merge_split_src_tgt(self, src, tgt):
        x_merged, y_split = [], []
        for i, (x, y) in enumerate(tqdm(zip(src, tgt))):
            if not self.already_split:
                x = re.sub(r'(\b)(\w+)(\b)', r'\1 \2 \3', x)
                x = re.sub(r'( ){2,}', r'\1', x)
                y = re.sub(r'(\b)(\w+)(\b)', r'\1 \2 \3', y)
                y = re.sub(r'( ){2,}', r'\1', y)
                x, y = x.strip(), y.strip()
            if not x or not y:
                continue
            x_a, y_a = self.tokenize_and_align(
                x, y,
                mask_indexes=(self.pad_label, self.pad_label),
                align_subsequences=True)
            if '' in [w[0] for w in x_a + y_a]:
                continue
            x_merged.append([re.sub(r'<space>', '', token[0])
                            for token in x_a])
            y_split.append([re.sub(r'<space>', ' ', token[0])
                           for token in y_a])
            assert len(x_merged[-1]) == len(y_split[-1]), f'Wrong merging'
        return x_merged, y_split


    def tokenize_and_align(self, seq1, seq2, mask_indexes, align_subsequences):
        """Returns an aligned version of the gold and pred tokens."""
        tokens = []
        for s_type, sent in enumerate([seq1, seq2]):
            tokens.append([])
            tokenized_sent = sent if self.already_split else sent.strip().split()
            if mask_indexes[0] and mask_indexes[1]:
                mask_index = self.find_mask_index(
                    tokenized_sent, mask_indexes[s_type])
                tokenized_sent = tokenized_sent[:mask_index]
            for i, token in enumerate(tokenized_sent):
                tokens[-1].append([token, None])

        seq1, seq2 = AlignmentHandler.align(tokens[0], tokens[1])
        if align_subsequences:
            seq1, seq2 = AlignmentHandler.align_subsequences(seq1, seq2)
        return seq1, seq2

    def find_mask_index(self, seq, s_type):
        if isinstance(seq, list):
            try:
                mask_index = seq.index(s_type)
            except:
                mask_index = len(seq)
        else:
            mask_index = np.where(seq == 0)[0]
            if mask_index.size:
                mask_index = mask_index[0]
            else:
                mask_index = len(seq)

    @staticmethod
    def align(src, tgt):
        """Corrects misalignments between the gold and predicted tokens
        which will almost almost always have different lengths due to inserted, 
        deleted, or substituted tookens in the predicted systme output."""

        sm = SequenceMatcher(
            a=list(map(lambda x: x[0], tgt)), b=list(map(lambda x: x[0], src)))
        tgt_temp, src_temp = [], []
        opcodes = sm.get_opcodes()
        for tag, i1, i2, j1, j2 in opcodes:
            # If they are equal, do nothing except lowercase them
            if tag == 'equal':
                for i in range(i1, i2):
                    tgt[i][1] = 'e'
                    tgt_temp.append(tgt[i])
                for i in range(j1, j2):
                    src[i][1] = 'e'
                    src_temp.append(src[i])
            # For insertions and deletions, put a filler of '***' on the other one, and
            # make the other all caps
            elif tag == 'delete':
                for i in range(i1, i2):
                    tgt[i][1] = 'd'
                    tgt_temp.append(tgt[i])
                for i in range(i1, i2):
                    src_temp.append(tgt[i])
            elif tag == 'insert':
                for i in range(j1, j2):
                    src[i][1] = 'i'
                    tgt_temp.append(src[i])
                for i in range(j1, j2):
                    src_temp.append(src[i])
            # More complicated logic for a substitution
            elif tag == 'replace':
                for i in range(i1, i2):
                    tgt[i][1] = 's'
                for i in range(j1, j2):
                    src[i][1] = 's'
                tgt_temp += tgt[i1:i2]
                src_temp += src[j1:j2]

        return src_temp, tgt_temp

    @staticmethod
    def align_subsequences(src_sub, tgt_sub):
        def process_ne(src_sub, tgt_sub):
            src_temp, tgt_temp = [], []
            # If there are 'i' and 'd' tokens in addition to 's', then there is splitting
            # We should should align at the character level
            if [True for t in src_sub if t[1] != 's']:
                src_temp_, tgt_temp_ = AlignmentHandler.soft_align(
                    tgt_sub, src_sub)
                src_temp += src_temp_
                tgt_temp += tgt_temp_
            # Else they are already aligned but not equal
            else:
                for j in range(len(src_sub)):
                    src_temp.append((src_sub[j][0], 'ne'))
                    tgt_temp.append((tgt_sub[j][0], 'ne'))
            return src_temp, tgt_temp

        start, end = -1, -1
        src_temp, tgt_temp = [], []
        for i, token in enumerate(src_sub):
            op = token[1]
            if start == -1 and op == 'e':
                src_temp.append(tuple(src_sub[i]))
                tgt_temp.append(tuple(tgt_sub[i]))
            elif start == -1 and op != 'e':
                start = i
            elif start != -1 and op == 'e':
                end = i
                src_temp_, tgt_temp_ = process_ne(
                    src_sub[start:end], tgt_sub[start:end])
                src_temp += src_temp_
                tgt_temp += tgt_temp_
                # Add first token with value 'e'
                src_temp.append(tuple(src_sub[i]))
                tgt_temp.append(tuple(tgt_sub[i]))
                start, end = -1, -1
        end = i + 1
        # If last operation is not e and we are in the
        # middle of a (possibly) badly aligned subsequence
        if start != -1:
            src_temp_, tgt_temp_ = process_ne(
                src_sub[start:end], tgt_sub[start:end])
            src_temp += src_temp_
            tgt_temp += tgt_temp_

        return src_temp, tgt_temp

    @staticmethod
    def soft_align(tgt, src):
        """Alignment at the character level."""
        src = ' '.join([token[0] for token in src if token[1] != 'd'])
        tgt = ' '.join([token[0] for token in tgt if token[1] != 'i'])
        src_temp = [[char, 'n'] for char in src]
        tgt_temp = [[char, 'n'] for char in tgt]
        src_temp, tgt_temp = AlignmentHandler.align(src_temp, tgt_temp)
        space_anchors = [0]
        for i, char in enumerate(src_temp):
            if char[0] == ' ' and char[1] == 'e':
                space_anchors.append(i + 1)
        space_anchors.append(len(src_temp) + 1)

        # At this point, each sequence of characters delimited by two space anchors
        # is most definitely a word unit (which may or may not be split)
        src_temp_, tgt_temp_ = [], []
        for i in range(len(space_anchors) - 1):
            src_sub_temp = src_temp[space_anchors[i]:space_anchors[i+1] - 1]
            tgt_sub_temp = tgt_temp[space_anchors[i]:space_anchors[i+1] - 1]
            src_sub_temp = ''.join([char[0] if char[0] != ' ' else '<space>'
                                    for char in src_sub_temp if char[1] != 'd'])
            tgt_sub_temp = ''.join([char[0] if char[0] != ' ' else '<space>'
                                    for char in tgt_sub_temp if char[1] != 'i'])
            src_temp_.append((src_sub_temp, 'ne'))
            tgt_temp_.append((tgt_sub_temp, 'ne'))
        return src_temp_, tgt_temp_
