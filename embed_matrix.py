import numpy as np
from collections import Counter
import tensorflow as tf
import re
import utils
import errors


class EmbedMatrix:
    """
    Holds TensorFlow embedding matrix variable and operations related to vocabulary and embedding.
    
    Logs:
        ``...embed_matrix.log``: Execution log.
        
    Raises:
        `errors.EmbedMatrixError`
    """
    def __init__(self, config):
        """
        Args:
            config (`Namespace`/`namedtuple`): Holds run configuration.
        """
        self._logger = utils.get_named_logger(config, 'embed_matrix')
        self._logger.debug("========== EMBED MATRIX ==========")
        self._config = config
        
        self._load_embed_file()
        self._add_pad_and_unknown()
        
        self._logger.debug("vocab size: %s" % self._vocab_size)
        self._logger.debug("embed size: %s" % self._embed_size)
        self._logger.debug("=" * 30)
    
    
    def log_unks_in_data(self, dm):
        if dm.__class__.__name__ == 'DevDataManager' or dm.__class__.__name__ == 'TrainDataManager':
            cols = dm._dsm._csv._train_data_types['str']
        elif dm.__class__.__name__ == 'TestDataManager':
            cols = dm._dsm._csv._test_data_types['str']

        dm_name = "_".join(re.findall('[A-Z][^A-Z]*', dm.__class__.__name__)[:-1]).lower()
        logger = utils.get_named_logger(self._config, 'unknown_words_%s' % dm_name)
            
        all_words = ""
        words = []
        for col in cols:
            words += list(dm._df[col].str.split(' ', expand=True).stack().unique())
            all_words += (" ".join(dm._df[col]) + " ")
        words = list(set(words))
        
        wc = Counter(all_words.split())
        logger.info("number of unique words: %s" % len(wc))
        
        unks = [word for word in words if not self.word_exists(word)]
        unks_with_count = [(unk, wc[unk]) for unk in unks]
        unks_with_count.sort(key=lambda tup: tup[1], reverse=True)
        
        sv_unks = sorted([unk for unk in unks if dm._dsm._ont.is_slot_or_value(unk)])

        logger.info("========== SLOT/VALUE UNKNOWNS ==========")
        for unk in sv_unks:
            logger.info("%-15s" % unk)
        
        logger.info("========== ALL UNKNOWNS ==========")
        for unk, count in unks_with_count:
            logger.info("%-15s: %s" % (unk, count))
        
    
    def _load_embed_file(self):
        embeds = []
        words = []

        with open(utils.get_abs_path(self._config.embed_dir, self._config.embed_name), 'r') as f:
            for line in f:
                line = line.split(' ', 1)
                word = line[0].lower()
                embed = np.fromstring(line[1], dtype='float32', sep=' ')

                words.append(word)
                embeds.append(embed)

        self._np_matrix = np.asarray(embeds)
        self._np_matrix = self._np_matrix[:self._config.max_vocab_size]
        
        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.idx2word = {idx: word for idx, word in enumerate(words)}
        self._vocab_size = self._np_matrix.shape[0]
        self._embed_size = self._np_matrix.shape[1]

        if not len(self._np_matrix.shape) == 2:
                raise errors.EmbedMatrixError("shape of embed matrix must be 2-dim: %s" % self._np_matrix.shape)

        if self._config.norm_embed:
            self._normalize_embed()
            
            
    def _add_pad_and_unknown(self):
        # average embedding for unknown word
        self._np_matrix = np.vstack([self._np_matrix, np.average(self._np_matrix, axis=0)])
        self.word2idx[self._config.unk_word] = self._vocab_size
        self.idx2word[self._vocab_size] = self._config.unk_word
        self._logger.debug("index of %s: %s" % (self._config.unk_word, self._vocab_size))
        self._vocab_size += 1
        
        # add zero padding for padding word
        self._np_matrix = np.vstack([self._np_matrix, [0 for i in range(self._embed_size)]])
        self.word2idx[self._config.pad_word] = self._vocab_size
        self.idx2word[self._vocab_size] = self._config.pad_word
        self._logger.debug("index of %s: %s" % (self._config.pad_word, self._vocab_size))
        
        self._zero_mask = np.ones((self._vocab_size, self._embed_size))
        self._zero_mask = np.vstack([self._zero_mask, [0 for i in range(self._embed_size)]])
        
        with tf.variable_scope('embed'):
            self._matrix = tf.Variable(self._np_matrix, trainable=self._config.train_embed,
                                      name='embed_matrix')
            self._zero_mask = tf.Variable(self._zero_mask, trainable=False, name='zero_mask')
            self._masked_matrix = tf.multiply(self._matrix, self._zero_mask)
        
        self._vocab_size += 1
    
    
    def get_embed_from_sentence(self, sentence):
        """
        Deprecated: use get_idx_from_sentence() and get_embed_from_idx() separately.
        Converts sentence to embedding tensor.
        
        Args:
            sentence (str): Sentence to embed.
        
        Returns:
            (`tf.Tensor`): TensorFlow Tensor which holds the embedding of the sentence.
        """
        idxs = self.get_idxs_from_sentence(sentence)
        return self.get_embed_from_idxs(idxs)
    
    
    def get_embeds_from_sentences(self, sentences):
        """
        Deprecated: use get_idxs_idxs_from_sentences() and get_embeds_from_idxs_idxs() separately.
        Converts sentences to embedding tensor.
        
        Args:
            sentences (list of str): Sentences to embed.
        
        Returns:
            (`tf.Tensor`): TensorFlow Tensor which holds the embedding of the sentences.
        """
        idxs_idxs = self.get_idxs_idxs_from_sentences(sentences)
        return self.get_embeds_from_idxs_idxs(idxs_idxs)

    def get_sentence_unknowns_removed(self, sentence):
        words = sentence.split()
        if not all(map(self.word_exists, words)) and not self._config.use_unk:
            unks = [word for word in words if not self.word_exists(word)]
            words = [word for word in words if word not in unks]
            sentence = " ".join(words)
        return sentence
    
    def get_idxs_from_sentence(self, sentence):
        """
        Converts sentence to word indices.
        
        Args:
            sentence (str): Sentence to embed.
        
        Returns:
            idxs (list of int): Indices of words in the sentence.
        """
        if not utils.is_str_or_npstr(sentence):
            raise errors.EmbedMatrixError("must be a str: %s" % sentence)
        
        sentence = self.get_sentence_unknowns_removed(sentence)
        
        if self._config.max_utterance_len is None:
            max_word_count = max(len(sentence.split()), 1)
            max_word_count += (max(self._config.filters_sizes) - 1)
        else:
            max_word_count = self._config.max_utterance_len
        sentence = utils.pad_or_cut_str(
            sentence, max_word_count, self._config.pad_word)
        words = np.asarray(sentence.split())
        idxs = self.get_idxs_from_words(words)
        return idxs
    
    
    def get_idxs_idxs_from_sentences(self, sentences):
        """
        Converts sentences to list of word indices.
        
        Args:
            sentences (list/`np.ndarray` of str): Sentences to embed.
        
        Returns:
            idxs_idxs (`np.ndarray` of `np.ndarray` of int): List of indices of words in the sentences.
        """
        if not utils.is_list_or_ndarray(sentences):
            raise errors.EmbedMatrixError("must be either a list or a numpy array: %s" % sentences)
        
        sentences = [self.get_sentence_unknowns_removed(sentence) for sentence in sentences]
        
        if self._config.max_utterance_len is None:
            max_word_count = max([len(sentence.split()) for sentence in sentences])
            max_word_count = max(max_word_count, max(self._config.filters_sizes))
            max_word_count += (max(self._config.filters_sizes) - 1)
        else:
            max_word_count = self._config.max_utterance_len

        sentences = np.asarray([utils.pad_or_cut_str(
            sentence, max_word_count, self._config.pad_word)
                                       for sentence in sentences])
        words_words = np.asarray(
            [np.asarray(sentence.split()) for sentence in sentences])
        idxs_idxs = np.asarray([self.get_idxs_from_words(words) for words in words_words])
        return idxs_idxs
    
    
    def get_embed_from_idxs(self, idxs):
        """
        Converts indices of words to embeddings.
        
        Args:
            idxs (list/`np.ndarray`): Indices of words.
        
        Returns:
            (`tf.Tensor`): Embeddings for the indices of words.
        """

        if self._config.train_embed:
            matrix = self._masked_matrix
        else:
            matrix = self._matrix
        return tf.gather(matrix, idxs)
    
    
    def get_embeds_from_idxs_idxs(self, idxs_idxs):
        """
        Converts list of indices of words to embeddings.
        
        Args:
            idxs_idxs (`np.ndarray` of `np.ndarray` of int): List of indices of words.
        
        Returns:
            (`tf.Tensor`): Embeddings for the list of indices of words.
        """
        if self._config.train_embed:
            matrix = self._masked_matrix
        else:
            matrix = self._matrix
        return tf.gather(matrix, idxs_idxs)

    
    def idx_exists(self, idx):
        """
        Returns whether word with the given index exists in the vocabulary.
        
        Args:
            idx (int): Word index.
        
        Returns:
            (bool): True if word with the given index exists in the vocabulary.
        """
        return idx in self.idx2word
    
    
    def word_exists(self, word):
        """
        Returns whether the given word exists in the vocabulary.
        
        Args:
            word (str): Word to check the existence.
        
        Returns:
            (bool): True if the given word exists in the vocabulary.
        """
        return word in self.word2idx
    
    
    def get_words_from_idxs(self, idxs):
        """
        Returns words that correspond to given indices respectively.
        
        Args:
            idxs (list/`np.ndarray` of int): Indices of words.
        
        Returns:
            (`np.ndarray` of str): Words that correspond to given indices respectively.
        """
        if not utils.is_list_or_ndarray(idxs):
            raise errors.EmbedMatrixError("must be either a list or a numpy array: %s" % idxs)
            
        for idx in idxs:
            if not utils.is_int_or_npint(idx):
                raise errors.EmbedMatrixError("must be an int: %s" % idx)
        
        for idx in idxs:
            if not self.idx_exists(idx):
                raise errors.EmbedMatrixError("index does not exist: %s" % idx)
        
        return np.asarray(list(map(self.get_word_from_idx, idxs)))
    
    
    def get_word_from_idx(self, idx):
        """
        Returns word that corresponds to the given index.
        
        Args:
            idx (int): Word index.
        
        Returns:
            (str): Word that correspond to the given index.
        """
        if not utils.is_int_or_npint(idx):
            raise errors.EmbedMatrixError("must be an int: %s" % idx)
        if not self.idx_exists(idx):
            raise errors.EmbedMatrixError("index does not exist: %s" % idx)
        return self.idx2word[idx]
    
    
    def get_idxs_from_words(self, words):
        """
        Returns indices that correspond to given words respectively.
        
        Args:
            words (list/`np.ndarray` of str): Words.
        
        Returns:
            (`np.ndarray` of int): Indices that correspond to given indices respectively.
        """
        if not utils.is_list_or_ndarray(words):
            raise errors.EmbedMatrixError("must be either a list or a numpy array: %s" % idxs)
        
        for word in words:
            if not utils.is_str_or_npstr(word):
                raise errors.EmbedMatrixError("must be a str: %s" % word)
        
        if not all(map(self.word_exists, words)) and not self._config.use_unk:
            unks = [word for word in words if not self.word_exists(word)]
            # self._logger.debug("removing unknown words: %s" % unks)
            raise errors.EmbedMatrixError("words does not exist: %s" % unks)

        return np.asarray(list(map(
            lambda word: self.get_idx_from_word(word), words)))

    
    def get_idx_from_word(self, word):
        """
        Returns index that corresponds to the given word.
        
        Args:
            word (str): Word.
        
        Returns:
            (int): Index that correspond to the given word.
        """
        if not utils.is_str_or_npstr(word):
            raise errors.EmbedMatrixError("must be a str: %s" % word)
        if not self.word_exists(word):
            if self._config.use_unk:
                return self.word2idx[self._config.unk_word]
            else:
                raise errors.EmbedMatrixError("word does not exist: %s" % word)
        return self.word2idx[word]
     
    
    def _normalize_embed(self):
        denom = np.sqrt(np.sum(self._np_matrix ** 2, axis=1) + self._config.eps)
        self._np_matrix = (self._np_matrix / np.reshape(
            denom, (denom.shape[0], -1))) * self._config.norm