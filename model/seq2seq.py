import tensorflow as tf
from .utils import create_embedding
from .utils import encoder
from .utils import train_decoder
from .utils import beamsearch_infer_decoder


class Seq2SeqModel(object):
    def __init__(self, config):
        print(" # Creating Seq2Seq Model.")
        self.config = config

        self.build_global_helper()
        self.build_forward_graph()
        self.build_backward_graph()

        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.init_op = tf.global_variables_initializer()

    def build_global_helper(self):
        """
        创建占位符
        :return:
        """
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None])
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None])
        self.decoder_targets = tf.placeholder(tf.int32, [None, None])

        self.encoder_lengths = tf.count_nonzero(self.encoder_inputs, -1, dtype=tf.int32)
        self.decoder_lengths = tf.count_nonzero(self.decoder_inputs, -1, dtype=tf.int32)

        self.keep_prob = tf.placeholder(tf.float32, [])

        self.batch_size = tf.shape(self.encoder_inputs)[0]
        self.global_step = tf.Variable(0, trainable=False)

        self.dec_max_len = tf.reduce_max(self.decoder_lengths, name="dec_max_len")
        self.decoder_weights = tf.sequence_mask(self.decoder_lengths, self.dec_max_len, dtype=tf.float32)

        self.lr = tf.Variable(self.config.learning_rate, trainable=False)
        self.new_lr = tf.placeholder(tf.float32, [])
        self.update_lr_op = tf.assign(self.lr, self.new_lr)

    def build_forward_graph(self):
        """
        建立前馈网络
        :return:
        """
        with tf.variable_scope('encoder'):
            """
            建立encoder，获得encoder_outputs, encoder_states
            encoder_outputs 是一个array，形状是 (batch_size, encoder_length, hidden_dim)
            encoder_states 是一个tuple，里面是一个个array，形状是 (batch_size, hidden_dim) * num_layer
            """
            embedding = create_embedding(self.config.vocab_size, self.config.embedding_dim)
            embedded_encoder_inputs = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
            embedded_decoder_inputs = tf.nn.embedding_lookup(embedding, self.decoder_inputs)
            encoder_outputs, encoder_states = encoder(embedded_encoder_inputs, self.encoder_lengths,
                                                      self.config, self.keep_prob)
            self.encoder_outputs = encoder_outputs
            self.encoder_states = encoder_states

        with tf.variable_scope('decoder'):
            """
            建立decoder，获得train_logits, train_sample_id
            train_logits 是一个array，形状是(batch_size, dec_max_length, vocab_size)
            train_sample_id 是一个array，形状是(batch_size, dec_max_length)
            """
            self.train_logits, self.train_sample_id = train_decoder(decoder_init_state=encoder_states,
                                                                    embedded_decoder_inputs=embedded_decoder_inputs,
                                                                    decoder_lenghts=self.decoder_lengths,
                                                                    hparams=self.config,
                                                                    keep_prob=self.keep_prob)

        with tf.variable_scope('decoder', reuse=True):
            self.infer_sample_id = beamsearch_infer_decoder(decoder_init_state=encoder_states,
                                                            config=self.config,
                                                            keep_prob=self.keep_prob,
                                                            embedding=embedding,
                                                            SOS_ID=self.config.SOS_ID,
                                                            EOS_ID=self.config.EOS_ID,
                                                            batch_size=self.batch_size)

    def build_backward_graph(self):
        """
        建立反馈网络
        :return:
        """
        temp = tf.cast(tf.not_equal(self.train_sample_id, self.decoder_targets), tf.float32)
        temp = tf.multiply(temp, self.decoder_weights)
        self.word_error_rate = tf.reduce_sum(temp) / tf.reduce_sum(self.decoder_weights)
        """
        loss 计算交叉熵，是一个一维array，大小为dec_max_length
        """
        loss = tf.contrib.seq2seq.sequence_loss(logits=self.train_logits,
                                                targets=self.decoder_targets,
                                                weights=self.decoder_weights,
                                                average_across_timesteps=False,
                                                average_across_batch=True)
        self.nll_loss = tf.reduce_mean(loss)
        self.total_loss = self.nll_loss

        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.total_loss, tvars)
        clip_grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(zip(clip_grads, tvars), global_step=self.global_step)

    def encoder_states_session(self, sess, enc_inp):
        enc_inp = enc_inp.reshape([-1, enc_inp.shape[-1]])  # 将多个utterance合并为一个utterance
        feed_dict = {
            self.encoder_inputs: enc_inp,
            self.keep_prob: self.config.keep_prob
        }
        fetches = [self.encoder_outputs, self.encoder_states]
        encoder_outputs, encoder_states = sess.run(fetches, feed_dict)
        return {'encoder_outputs': encoder_outputs, 'encoder_states': encoder_states}

    def train_session(self, sess, enc_inp, dec_inp, dec_tar):
        enc_inp = enc_inp.reshape([-1, enc_inp.shape[-1]]) # 将多个utterance合并为一个utterance
        dec_inp = dec_inp.reshape([-1, dec_inp.shape[-1]])
        dec_tar = dec_tar.reshape([-1, dec_tar.shape[-1]])
        feed_dict = {
            self.encoder_inputs: enc_inp,
            self.decoder_inputs: dec_inp,
            self.decoder_targets: dec_tar,
            self.keep_prob:self.config.keep_prob
        }
        fetch = [self.train_op, self.total_loss, self.nll_loss, self.word_error_rate, self.global_step]
        _, total_loss, nll_loss, word_error_rate, step = sess.run(fetch, feed_dict)
        return {'total_loss': total_loss, 'nll_loss': nll_loss, 'word_error_rate': word_error_rate, 'step': step}

    def infer_session(self, sess, enc_inp):
        enc_inp = enc_inp.reshape([-1, enc_inp.shape[-1]]) # 将多个utterance合并为一个utterance
        feed_dict = {
            self.encoder_inputs: enc_inp,
            self.keep_prob: 1.0
        }
        fetch = [self.infer_sample_id]
        infer_sample_id = sess.run(fetch, feed_dict)
        return {'infer_sample_id': infer_sample_id}

    def eval_session(self, sess, enc_inp, dec_inp, dec_tar):
        enc_inp = enc_inp.reshape([-1, enc_inp.shape[-1]])  # 将多个utterance合并为一个utterance
        dec_inp = dec_inp.reshape([-1, dec_inp.shape[-1]])
        dec_tar = dec_tar.reshape([-1, dec_tar.shape[-1]])
        feed_dict = {
            self.encoder_inputs: enc_inp,
            self.decoder_inputs: dec_inp,
            self.decoder_targets: dec_tar,
            self.keep_prob: 1.0
        }
        fetches = [self.total_loss, self.nll_loss, self.word_error_rate]
        total_loss, nll_loss, word_error_rate = sess.run(fetches, feed_dict)
        return {'total_loss': total_loss, 'nll_loss': nll_loss,
                'word_error_rate': word_error_rate}

    def get_parameter_size(self):
        all_vars = tf.global_variables()
        total_count = 0
        for item in all_vars:
            if 'Adam' in item.name:
                continue
            shape = item.get_shape().as_list()
            if len(shape) == 0:
                total_count += 1
            else:
                size = 1
                for val in shape:
                    size *= val
                total_count += size
        return total_count
