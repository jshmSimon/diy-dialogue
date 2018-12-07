import datetime
import math
import os
import gensim
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from data_utils import DataLoader
from model.seq2seq import Seq2SeqModel as Model
from config import Config


"""
tf.app.flags.FLAGS 是全局变量
tf.app.flags.FLAGS 用于在命令行状态下运行时传递参数
tf.app.flags.DEFINE_string() 定义了一个string类型的参数，函数内第一个参数是该参数的名称，第二个是默认值，第三个是描述
"""
tf.app.flags.DEFINE_string('mode', 'train', '[train]')
tf.app.flags.DEFINE_boolean('retrain', True, '')
FLAGS = tf.app.flags.FLAGS

class Seq2SeqChatbot(object):
    def __init__(self):
        print(" # Welcome to Seq2Seq Chatbot.")
        print(" # Tensorflow detected: v{}".format(tf.__version__))
        print()
        self.config = Config

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

        self.sess = tf.Session()
        self.config.checkpoint_dir = os.path.join("save", self.model.__class__.__name__)
        print(" # Save directory: {}".format(self.config.checkpoint_dir))

    def main(self):
        # self.encoder_states(self.sess)
        # self.train_model(self.sess)
        if FLAGS.mode == 'train':
            ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) and (not FLAGS.retrain):
                print(" # Restoring model parameters from %s." % ckpt.model_checkpoint_path)
                self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                print(" # Creating model with fresh parameters.")
                self.sess.run(self.model.init_op)
            self.train_model(self.sess)

    def encoder_states(self, sess):
        f = 0
        count = 0
        for (enc_inp, dec_inp, dec_tar) in tqdm(self.dataloader.data_generator(flag='test')):
            outputs = self.model.encoder_states_session(sess, enc_inp)
            encoder_states = outputs['encoder_states']
            encoder_outputs = outputs['encoder_outputs']
            if f <= 2:
                print('number of layer: {}'.format(len(encoder_states)))
                for state in encoder_states:
                    print('shape of encoder_states: {}'.format(state.shape))
                print('shape of encoder_outputs: {}'.format(encoder_outputs.shape))
                f += 1
            print(count)
            count += 1

    def save_session(self, sess):
        print(" # Saving checkpoints.")
        save_dir = os.path.join(self.config.checkpoint_dir)
        model_name = self.model.__class__.__name__ + '.ckpt'
        checkpoint_path = os.path.join(save_dir, model_name)
        self.model.saver.save(sess, checkpoint_path)
        print(' # Model saved.')

    def train_model(self, sess):
        best_result_loss = 1000.0
        for epoch in range(self.config.num_epoch):
            print()
            print('----epoch: {}/{} | lr: {}'.format(epoch, self.config.num_epoch, sess.run(self.model.lr)))

            tic = datetime.datetime.now()
            train_iterator = self.dataloader.data_generator(flag='train')
            test_iterator = self.dataloader.data_generator(flag='test')
            train_batch_num = self.dataloader.train_batch_num
            # test_batch_num = self.dataloader.test_batch_num

            total_loss = 0.0
            nll_loss = 0.0
            word_error_rate = 0.0
            count = 0

            for (enc_inp, dec_inp, dec_tar) in tqdm(train_iterator, desc='training'):
                train_out = self.model.train_session(sess, enc_inp, dec_inp, dec_tar)

                count += 1
                step = train_out["step"] # step 表示训练了多少个batch
                total_loss += train_out["total_loss"]
                nll_loss += train_out["nll_loss"]
                word_error_rate += train_out["word_error_rate"]

                if step % 50 == 0:
                    cur_loss = total_loss / count
                    cur_nll_loss = nll_loss / count
                    cur_word_error_rate = word_error_rate / count
                    cur_perplexity = math.exp(float(cur_nll_loss)) if cur_nll_loss < 300 else float("inf")
                    print(" Step %4d | Batch [%3d/%3d] | Loss %.6f | PPL %.6f | WER %.6f" %
                          (step, count, train_batch_num, cur_loss, cur_perplexity, cur_word_error_rate))
            print()
            total_loss /= count
            nll_loss /= count
            word_error_rate /= count
            perplexity = math.exp(float(nll_loss)) if nll_loss < 300 else float("inf")
            print(" Train Epoch %4d | Loss %.6f | PPL %.6f | WER %.6f" %
                  (epoch, total_loss, perplexity, word_error_rate))

            # testing after every epoch
            test_loss = 0.0
            test_nll_loss = 0.0
            test_count = 0
            test_rate = 0.0
            for (enc_inp, dec_inp, dec_tar) in tqdm(test_iterator, desc="testing"):
                test_outputs = self.model.eval_session(sess, enc_inp, dec_inp, dec_tar)
                test_loss += test_outputs["total_loss"]
                test_nll_loss += test_outputs["nll_loss"]
                test_rate += test_outputs["word_error_rate"]
                test_count += 1
            test_loss /= test_count
            test_rate /= test_count
            test_nll_loss /= test_count
            test_perp = math.exp(float(test_nll_loss)) if test_nll_loss < 300 else float("inf")
            print(" Test Epoch %d | Loss %.6f | PPL %.6f | WER %.6f" % (epoch, test_loss, test_perp, test_rate))
            print()

            if test_loss < best_result_loss:
                self.save_session(sess)
                if np.abs(best_result_loss - test_loss) < 0.03:
                    cur_lr = sess.run(self.model.lr)
                    sess.run(self.model.update_lr_op, feed_dict={self.model.new_lr: cur_lr * 0.5})
                best_result_loss = test_loss
            toc = datetime.datetime.now()
            print(" # Epoch finished in {}".format(toc - tic))


if __name__ == '__main__':
    chatbot = Seq2SeqChatbot()
    chatbot.main()