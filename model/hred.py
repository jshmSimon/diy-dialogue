import numpy as np
import tensorflow as tf

from .utils import create_embedding
from .utils import encoder
from .utils import train_decoder_with_concat
from .utils import beamsearch_infer_decoder_with_concat
from .utils import create_multi_rnn_cell


class HREDModel(object):
	def __init__(self, config):
		print(" # Creating HRED Model.")
		self.config = config
		self.num_utterance = self.config.num_utterance
		print(self.num_utterance)

		self.build_global_helper()
		self.build_forward_graph()
		self.build_backward_graph()
		self.build_summary_graph()

		self.merged = tf.summary.merge_all()
		self.saver = tf.train.Saver()
		self.init_op = tf.global_variables_initializer()


	def build_global_helper(self):

		self.encoder_inputs = tf.placeholder(tf.int32, [None, None, None])
		self.decoder_inputs = tf.placeholder(tf.int32, [None, None, None])
		self.decoder_targets = tf.placeholder(tf.int32, [None, None, None])
		self.encoder_lengths = tf.count_nonzero(self.encoder_inputs, [-1], dtype=tf.int32)
		self.decoder_lengths = tf.count_nonzero(self.decoder_inputs, [-1], dtype=tf.int32)

		self.enc_max_len = tf.reduce_max(self.encoder_lengths, -1)
		self.dec_max_len = tf.reduce_max(self.decoder_lengths, -1)

		self.keep_prob = tf.placeholder(tf.float32, [])
		self.batch_size = tf.shape(self.encoder_inputs)[1]

		self.global_step = tf.Variable(0, trainable=False)

		self.lr = tf.Variable(self.config.learning_rate, trainable=False)
		self.new_lr = tf.placeholder(tf.float32, [])
		self.update_lr_op = tf.assign(self.lr, self.new_lr)


	def build_forward_graph(self):

		with tf.variable_scope("encoder"):
			embedding = create_embedding(self.config.vocab_size, self.config.embedding_dim)

			self.context_inputs = [] #(num_utterance, batch_size, hidden_dim)
			self.encoder_states = [] #(num_utterance, num_layer, batch_size, hidden_dim)
			self.encoder_outputs = [] #(num_utterance, batch_size, enc_max_len, hidden_dim)

			for time_step in range(self.num_utterance):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()

				encoder_output, encoder_state = encoder(
						tf.nn.embedding_lookup(embedding, self.encoder_inputs[time_step]),
						self.encoder_lengths[time_step],
						self.config,
						self.keep_prob)

				self.context_inputs.append(encoder_state[-1]) #最后一个layer的state  (batch_size, hidden_dim)
				self.encoder_states.append(encoder_state)
				self.encoder_outputs.append(encoder_output)


			assert len(self.context_inputs) == self.num_utterance


		with tf.variable_scope("context"):
			context_cell = create_multi_rnn_cell(
				self.config.rnn_type,
				self.config.hidden_dim,
				self.keep_prob,
				self.config.num_layer)
			state = self.encoder_states[-1] # context_cell.zero_state(self.batch_size, dtype=tf.float32)
			# 最后一个utterance的state (num_layer, batch_size, hidden_dim)

			self.context_states = [] # (num_utterance, num_layer, batch_size, hidden_dim)
			for time_step in range(self.num_utterance):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				# self.context_inputs[time_step]: 当前utterance的最后一层的state (batch_size, hidden_dim)
				# state: 最后一个utterance的所有层的state (num_layer, batch_size, hidden_dim)
				(cell_output, state) = context_cell(self.context_inputs[time_step], state)
				self.context_states.append(state)

			assert len(self.context_states) == self.num_utterance

		with tf.variable_scope("decoder"):

			self.total_opt_loss = []
			self.total_nll_loss = []
			self.total_word_error_rate = []
			self.train_sample_id = []
			self.infer_sample_id = []

			for time_step in range(self.num_utterance):
				dec_targets = self.decoder_targets[time_step]
				dec_weights = tf.sequence_mask(self.decoder_lengths[time_step], self.dec_max_len[time_step], dtype=tf.float32)
				dec_inputs = self.decoder_inputs[time_step]
				dec_lengths = self.decoder_lengths[time_step]

				concat_vector =  self.context_states[time_step][-1] #最后一个layer的states  (batch_size, hidden_dim)
				dec_init_state = self.encoder_states[time_step]

				with tf.variable_scope("RNN"):
					if time_step > 0:
						tf.get_variable_scope().reuse_variables()

					train_logits, train_sample_id = train_decoder_with_concat(
						dec_init_state,
						self.config,
						self.keep_prob,
						tf.nn.embedding_lookup(embedding, dec_inputs),
						dec_lengths,
						concat_vector)

					self.train_logits = train_logits

					self.train_sample_id.append(train_sample_id)
					temp = tf.cast(tf.not_equal(train_sample_id, dec_targets), tf.float32)
					temp = tf.multiply(temp, dec_weights)
					cur_word_error_rate = tf.reduce_sum(temp) / tf.reduce_sum(dec_weights)
					self.total_word_error_rate.append(cur_word_error_rate)


					nll_loss = tf.contrib.seq2seq.sequence_loss(
						logits=train_logits,
						targets=dec_targets,
						weights=dec_weights,
						average_across_timesteps=False,
						average_across_batch=True)

					self.total_nll_loss.append(tf.reduce_mean(nll_loss))
					self.total_opt_loss.append(tf.reduce_sum(nll_loss))


				with tf.variable_scope("RNN", reuse=True):
					infer_sample_id = beamsearch_infer_decoder_with_concat(
						dec_init_state,
						self.config,
						self.keep_prob,
						embedding,
						self.config.SOS_ID,
						self.config.EOS_ID,
						self.batch_size,
						concat_vector)

					self.infer_sample_id.append(infer_sample_id)



	def build_backward_graph(self):

		self.last_nll_loss = self.total_nll_loss[1]
		self.last_word_error_rate = self.total_word_error_rate[1]

		self.nll_loss = tf.reduce_mean(self.total_nll_loss)
		self.word_error_rate = tf.reduce_mean(self.total_word_error_rate)
		self.total_loss = self.nll_loss # tf.reduce_mean(self.total_opt_loss)

		optimizer = tf.train.AdamOptimizer(self.lr)
		tvars = tf.trainable_variables()
		grads = tf.gradients(self.total_loss, tvars)
		clip_grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)
		self.train_op = optimizer.apply_gradients(zip(clip_grads, tvars), global_step=self.global_step)



	def train_session(self, sess, enc_inp_list, dec_inp_list, dec_tar_list):
		feed_dict = {}
		feed_dict[self.encoder_inputs] = enc_inp_list # (utterance_num, batch_size, word_num)
		feed_dict[self.decoder_inputs] = dec_inp_list
		feed_dict[self.decoder_targets] = dec_tar_list
		feed_dict[self.keep_prob] = self.config.keep_prob

		_, total_loss, nll_loss, last_nll_loss, word_error_rate, last_word_error_rate, summary, step = sess.run(
			[self.train_op,
			 self.total_loss,
			 self.nll_loss,
			 self.last_nll_loss,
			 self.word_error_rate,
			 self.last_word_error_rate,
			 self.merged,
			 self.global_step], feed_dict=feed_dict)

		return {"total_loss":total_loss,
				"nll_loss": nll_loss,
				"last_nll_loss": last_nll_loss,
				"word_error_rate":word_error_rate,
				"last_word_error_rate": last_word_error_rate,
				"summary": summary,
				"step": step}

	def build_summary_graph(self):
		tf.summary.scalar('last_nll_loss', self.last_nll_loss)
		tf.summary.scalar('nll_loss', self.nll_loss)
		tf.summary.scalar('word_error_rate', self.word_error_rate)
		tf.summary.scalar('last_word_error_rate', self.last_word_error_rate)
		tf.summary.scalar('perplexity', tf.exp(self.nll_loss))
		tf.summary.scalar('last_perplexity', tf.exp(self.last_nll_loss))


	def eval_session(self, sess, enc_inp_list, dec_inp_list, dec_tar_list):
		feed_dict = {}
		feed_dict[self.encoder_inputs] = enc_inp_list
		feed_dict[self.decoder_inputs] = dec_inp_list
		feed_dict[self.decoder_targets] = dec_tar_list
		feed_dict[self.keep_prob] = 1.0

		total_loss, nll_loss, last_nll_loss, word_error_rate, last_word_error_rate, summary = sess.run(
			[self.total_loss,
			 self.nll_loss,
			 self.last_nll_loss,
			 self.word_error_rate,
			 self.last_word_error_rate,
			 self.merged], feed_dict=feed_dict)

		return {"total_loss":total_loss,
				"nll_loss": nll_loss,
				"last_nll_loss": last_nll_loss,
				"word_error_rate":word_error_rate,
				"last_word_error_rate": last_word_error_rate,
				"summary": summary}


	def infer_session(self, sess, enc_inp_list, dec_inp_list, dec_tar_list):
		feed_dict = {}
		feed_dict[self.encoder_inputs] = enc_inp_list
		feed_dict[self.decoder_inputs] = dec_inp_list
		feed_dict[self.decoder_targets] = dec_tar_list
		feed_dict[self.keep_prob] = 1.0

		train_sample_id, infer_sample_id = sess.run(
			[self.train_sample_id, self.infer_sample_id], feed_dict=feed_dict)

		return {"train_sample_id": train_sample_id, "infer_sample_id":infer_sample_id}


	def get_parameter_size(self):
		all_vars = tf.global_variables()
		total_count = 0
		for item in all_vars:
			if "Adam" in item.name:
				continue
			shape = item.get_shape().as_list()
			if len(shape) == 0:
				total_count += 1
			else:
				size =  1
				for val in shape:
					size *= val
				total_count += size
		return total_count