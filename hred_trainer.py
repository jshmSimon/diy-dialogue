import datetime
import math
import os
import gensim
import random
import nltk

from nltk.util import ngrams
from collections import Counter
import tensorflow as tf
from tqdm import tqdm
import numpy as np

from data_utils import DataLoader
from models.hred import HREDModel as Model
from embedding_metric import embedding_metric
from config import HConfig


tf.app.flags.DEFINE_string("mode", "train", "[train, evaluate]")
tf.app.flags.DEFINE_boolean("retrain", True, "")
FLAGS = tf.app.flags.FLAGS


class HREDChatbot(object):
	def __init__(self):
		print(" # Welcome to HRED Chatbot.")
		print(" # Tensorflow detected: v{}".format(tf.__version__))
		print()

		self.config = HConfig()

		self.dataloader = DataLoader(
			self.config.num_utterance,
			self.config.max_length,
			self.config.split_ratio,
			 self.config.batch_size)
		self.word_to_id = self.dataloader.word_to_id
		self.id_to_word = self.dataloader.id_to_word
		self.config.vocab_size = self.dataloader.vocab_size
		self.config.SOS_ID = self.dataloader.SOS_ID
		self.config.EOS_ID = self.dataloader.EOS_ID

		self.model = Model(self.config)
		print()
		print(" # Parameter Size: {}".format(self.model.get_parameter_size()))
		print()

		### session
		self.sess = tf.Session()

		self.config.checkpoint_dir = os.path.join("save",self.model.__class__.__name__)
		print(" # Save directory: {}".format(self.config.checkpoint_dir))


	def main(self):
		if FLAGS.mode == "train":
			ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) and (not FLAGS.retrain):
				print(" # Restoring model parameters from %s." % ckpt.model_checkpoint_path)
				self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
			else:
				print(" # Creating model with fresh parameters.")
				self.sess.run(self.model.init_op)
			self.train_model(self.sess)

		elif FLAGS.mode == "evaluate":
			ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				print(" # Restoring model parameters from %s." % ckpt.model_checkpoint_path)
				self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
			self.evaluate_model(self.sess)
			self.evaluate_embedding(self.sess)

		elif FLAGS.mode == "embedding":
			ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				print(" # Restoring model parameters from %s." % ckpt.model_checkpoint_path)
				self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
			self.evaluate_embedding(self.sess)


		elif FLAGS.mode == "generate":
			ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				print(" # Restoring model parameters from %s." % ckpt.model_checkpoint_path)
				self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
			self.generate(self.sess)



	def train_model(self, sess):
		best_result_loss = 1000.0
		for epoch in range(self.config.num_epoch):
			print()
			print("---- epoch: {}/{} | lr: {} ----".format(epoch, self.config.num_epoch, sess.run(self.model.lr)))

			tic = datetime.datetime.now()
			train_iterator = self.dataloader.train_generator()
			test_iterator = self.dataloader.test_generator()
			train_batch_num = self.dataloader.train_batch_num
			test_batch_num = self.dataloader.test_batch_num

			total_loss = 0.0
			nll_loss = 0.0
			word_error_rate = 0.0
			count = 0
			last_nll_loss = 0.0
			last_word_error_rate = 0.0

			for (enc_inp, dec_inp, dec_tar) in tqdm(train_iterator, desc="training"):

				train_out = self.model.train_session(sess, enc_inp, dec_inp, dec_tar)

				count += 1
				step = train_out["step"]
				total_loss += train_out["total_loss"]
				nll_loss += train_out["nll_loss"]
				word_error_rate += train_out["word_error_rate"]
				last_nll_loss += train_out["last_nll_loss"]
				last_word_error_rate += train_out["last_word_error_rate"]

				if step % 50 == 0:
					cur_loss = total_loss / count
					cur_word_error_rate = word_error_rate / count
					cur_last_word_error_rate = last_word_error_rate / count
					cur_nll_loss = nll_loss / count
					cur_last_nll_loss = last_nll_loss / count
					cur_perplexity = math.exp(float(cur_nll_loss)) if cur_nll_loss < 300 else float("inf")
					cur_last_perplexity = math.exp(float(cur_last_nll_loss)) if cur_last_nll_loss < 300 else float("inf")
					print(" Step %4d | Batch [%3d/%3d] | Loss %.6f | PPL %.6f | PPL@L %.6f | WER %.6f | WER@L %.6f" %
						(step, count, train_batch_num, cur_loss, cur_perplexity, cur_last_perplexity, cur_word_error_rate,cur_last_word_error_rate))

			print("\n")
			total_loss /= count
			nll_loss /= count
			word_error_rate /= count
			last_nll_loss /= count
			last_word_error_rate /= count
			last_perplexity = math.exp(float(last_nll_loss)) if last_nll_loss < 300 else float("inf")
			perplexity = math.exp(float(nll_loss)) if nll_loss < 300 else float("inf")
			print(" Train Epoch %4d | Loss %.6f | PPL %.6f | PPL@L %.6f | WER %.6f | WER@L %.6f" %
						(epoch, total_loss, perplexity, last_perplexity, word_error_rate, last_word_error_rate))


			test_loss = 0.0
			test_nll_loss = 0.0
			test_count = 0
			test_rate = 0.0
			test_last_nll_loss = 0.0
			test_last_word_error_rate = 0.0
			for (enc_inp, dec_inp, dec_tar) in tqdm(test_iterator, desc="testing"):
				# print(np.array(enc_inp).shape)

				test_outputs = self.model.eval_session(sess, enc_inp, dec_inp, dec_tar)
				test_loss += test_outputs["total_loss"]
				test_nll_loss += test_outputs["nll_loss"]
				test_rate += test_outputs["word_error_rate"]
				test_last_nll_loss += test_outputs["last_nll_loss"]
				test_last_word_error_rate += test_outputs["last_word_error_rate"]
				test_count += 1
			test_loss /= test_count
			test_rate /= test_count
			test_nll_loss /= test_count
			test_last_word_error_rate/=test_count
			test_last_nll_loss /= test_count
			test_last_perp = math.exp(float(test_last_nll_loss)) if test_last_nll_loss < 300 else float("inf")
			test_perp = math.exp(float(test_nll_loss)) if test_nll_loss < 300 else float("inf")
			print(" Test Epoch %d | Loss %.6f | PPL %.6f | PPL@L %.6f | WER %.6f | WER@L %.6f" % (epoch, test_loss, test_perp, test_last_perp, test_rate, test_last_word_error_rate))
			print()

			if test_loss < best_result_loss:
				self.save_session(sess)

				if np.abs(best_result_loss - test_loss) < 0.1:
					cur_lr = sess.run(self.model.lr)
					sess.run(self.model.update_lr_op, feed_dict={self.model.new_lr: cur_lr * 0.5})

				best_result_loss = test_loss
			toc = datetime.datetime.now()
			print(" # Epoch finished in {}".format(toc-tic))

	def save_session(self, sess):
		print(" # Saving checkpoints.")

		save_dir = os.path.join(self.config.checkpoint_dir)
		model_name = self.model.__class__.__name__ + ".ckpt"
		checkpoint_path = os.path.join(save_dir, model_name)

		self.model.saver.save(sess, checkpoint_path)
		print(' # Model saved.')


	def evaluate_model(self, sess):
		print()
		print(" # Start Evaluating Metrics on Test Dataset.")
		print(" # Evaluating Per-word Perplexity and Word Error Rate on Test Dataset...")
		test_iterator = self.dataloader.test_generator()
		test_loss = 0.0
		test_nll_loss = 0.0
		test_count = 0
		test_rate = 0.0
		test_last_nll_loss = 0.0
		test_last_word_error_rate = 0.0
		for (enc_inp, dec_inp, dec_tar) in tqdm(test_iterator, desc="testing"):

			test_outputs = self.model.eval_session(sess, enc_inp, dec_inp, dec_tar)
			test_loss += test_outputs["total_loss"]
			test_nll_loss += test_outputs["nll_loss"]
			test_rate += test_outputs["word_error_rate"]
			test_last_nll_loss += test_outputs["last_nll_loss"]
			test_last_word_error_rate += test_outputs["last_word_error_rate"]
			test_count += 1
		test_loss /= test_count
		test_rate /= test_count
		test_nll_loss /= test_count
		test_last_word_error_rate/=test_count
		test_last_nll_loss /= test_count
		test_last_perp = math.exp(float(test_last_nll_loss)) if test_last_nll_loss < 300 else float("inf")
		test_perp = math.exp(float(test_nll_loss)) if test_nll_loss < 300 else float("inf")
		print(" Test Epoch | Loss %.6f | PPL %.6f | PPL@L %.6f | WER %.6f | WER@L %.6f" % (test_loss, test_perp, test_last_perp, test_rate, test_last_word_error_rate))
		print()


	def evaluate_embedding(self, sess):
		print()
		print(" # Evaluating Embedding-based Metrics on Test Dataset.")
		print(" # Loading word2vec embedding...")
		word2vec_path = "data/GoogleNews-vectors-negative300.bin"
		word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
		keys = word2vec.vocab

		test_iterator = self.dataloader.test_generator()
		metric_average_history = []
		metric_extrema_history = []
		metric_greedy_history = []
		metric_average_history_1 = []
		metric_extrema_history_1 = []
		metric_greedy_history_1 = []

		for (enc_inp, dec_inp, dec_tar) in tqdm(test_iterator, desc="testing"):
			test_outputs = self.model.infer_session(sess, enc_inp, dec_inp, dec_tar)
			infer_sample_id = test_outputs["infer_sample_id"][1]
			train_sample_id = test_outputs["train_sample_id"][1]
			ground_truth = dec_tar[1]


			ground_truth = [[self.id_to_word.get(idx, "<unk>") for idx in sent] for sent in ground_truth]
			infer_sample_id = [[self.id_to_word.get(idx, "<unk>") for idx in sent] for sent in infer_sample_id]
			train_sample_id = [[self.id_to_word.get(idx, "<unk>") for idx in sent] for sent in train_sample_id]

			ground_truth = [[word2vec[w] for w in sent if w in keys] for sent in ground_truth]
			infer_sample_id = [[word2vec[w] for w in sent if w in keys] for sent in infer_sample_id]
			train_sample_id = [[word2vec[w] for w in sent if w in keys] for sent in train_sample_id]

			infer_indices = [i for i, s, g in zip(range(len(infer_sample_id)), infer_sample_id, ground_truth) if s != [] and g != []]
			train_indices = [i for i, s, g in zip(range(len(train_sample_id)), train_sample_id, ground_truth) if s != [] and g != []]
			infer_samples = [infer_sample_id[i] for i in infer_indices]
			train_samples = [train_sample_id[i] for i in train_indices]
			infer_ground_truth = [ground_truth[i] for i in infer_indices]
			train_ground_truth = [ground_truth[i] for i in train_indices]

			metric_average = embedding_metric(infer_samples, infer_ground_truth, word2vec, 'average')
			metric_extrema = embedding_metric(infer_samples, infer_ground_truth, word2vec, 'extrema')
			metric_greedy = embedding_metric(infer_samples, infer_ground_truth, word2vec, 'greedy')
			metric_average_history.append(metric_average)
			metric_extrema_history.append(metric_extrema)
			metric_greedy_history.append(metric_greedy)

			avg = embedding_metric(train_samples, train_ground_truth, word2vec, "average")
			ext = embedding_metric(train_samples, train_ground_truth, word2vec, "extrema")
			gre = embedding_metric(train_samples, train_ground_truth, word2vec, "greedy")
			metric_average_history_1.append(avg)
			metric_extrema_history_1.append(ext)
			metric_greedy_history_1.append(gre)


		epoch_average = np.mean(np.concatenate(metric_average_history), axis=0)
		epoch_extrema = np.mean(np.concatenate(metric_extrema_history), axis=0)
		epoch_greedy = np.mean(np.concatenate(metric_greedy_history), axis=0)
		print()
		print(' # Embedding Metrics | Average: %.6f | Extrema: %.6f | Greedy: %.6f' % (epoch_average, epoch_extrema, epoch_greedy))
		print()

		epoch_average = np.mean(np.concatenate(metric_average_history_1), axis=0)
		epoch_extrema = np.mean(np.concatenate(metric_extrema_history_1), axis=0)
		epoch_greedy = np.mean(np.concatenate(metric_greedy_history_1), axis=0)
		print()
		print(' # Embedding Metrics | Average: %.6f | Extrema: %.6f | Greedy: %.6f' % (epoch_average, epoch_extrema, epoch_greedy))
		print()



	def generate(self, sess):
		test_iterator = self.dataloader.test_generator()
		for (enc_inp, dec_inp, dec_tar) in tqdm(test_iterator, desc="testing"):
			test_outputs = self.model.infer_session(sess, enc_inp, dec_inp, dec_tar)
			infer_sample_id = test_outputs["infer_sample_id"][1]
			ground_truth = dec_tar[0]

			for i in range(self.config.batch_size):
				ground = ground_truth[i]
				gener = infer_sample_id[i]

				ground_list = [self.id_to_word.get(idx, "<unk>") for idx in ground]
				gener_list = [self.id_to_word.get(idx, "<unk>") for idx in gener]

				print(" ".join(ground_list))
				print(" ".join(gener_list))
				print()





if __name__ == '__main__':
	chatbot = HREDChatbot()
	chatbot.main()