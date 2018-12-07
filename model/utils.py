import tensorflow as tf


def create_embedding(vocab_size, embedding_dim):
    embedding = tf.get_variable('embedding', [vocab_size, embedding_dim], dtype=tf.float32)
    return embedding


def single_rnn_cell(rnn_type, hidden_dim, output_keep_prob):
    if rnn_type.lower() == 'gru':
        cell = tf.contrib.rnn.GRUCell(hidden_dim)
    elif rnn_type.lower() == 'lstm':
        cell = tf.contrib.rnn.LSTMCell(hidden_dim)
    else:
        raise ValueError('Unsupported rnn type: {}'.format(rnn_type))
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=output_keep_prob)
    return cell


def create_multi_rnn_cell(rnn_type, hidden_dim, output_keep_prob, num_layer):
    """"
    注意：一定要在下面的MultiRNNCell的[]里调用函数生成每个单层的cell，不能先生成cell，再在[]里堆叠
    下面这种方式是错误的：
    cell = single_rnn_cell(rnn_type, hidden_dim, output_keep_prob)
    multi_cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layer)], state_is_tuple=True)
    """
    multi_cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell(rnn_type, hidden_dim, output_keep_prob) for _ in range(num_layer)], state_is_tuple=True)
    return multi_cell


def encoder(embedded_inputs, seq_lens, hparams, keep_prob):
    rnn_type = hparams.rnn_type
    hidden_dim = hparams.hidden_dim
    num_layer = hparams.num_layer
    output_keep_prob = keep_prob
    encoder_cell = create_multi_rnn_cell(rnn_type, hidden_dim, output_keep_prob, num_layer)
    outputs, states = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=embedded_inputs, sequence_length=seq_lens,
                                        dtype=tf.float32)
    return outputs, states


def train_decoder(decoder_init_state, embedded_decoder_inputs, decoder_lenghts, hparams, keep_prob):
    rnn_type = hparams.rnn_type
    hidden_dim = hparams.hidden_dim
    num_layer = hparams.num_layer
    output_keep_prob = keep_prob
    vocab_size = hparams.vocab_size
    decoder_cell = create_multi_rnn_cell(rnn_type, hidden_dim, output_keep_prob, num_layer)

    output_layer = tf.layers.Dense(vocab_size,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   _scope='decoder/dense')
    train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=embedded_decoder_inputs,
                                                     sequence_length=decoder_lenghts)
    train_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                    helper=train_helper,
                                                    initial_state=decoder_init_state,
                                                    output_layer=output_layer)
    train_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=train_decoder,
                                                           swap_memory=True,
                                                           maximum_iterations=tf.reduce_max(decoder_lenghts))
    logist = train_output.rnn_output
    sample_id = train_output.sample_id
    return logist, sample_id


def beamsearch_infer_decoder(decoder_init_state, config, keep_prob, embedding, SOS_ID, EOS_ID, batch_size):
    decoder_cell = create_multi_rnn_cell(config.rnn_type, config.hidden_dim, keep_prob, config.num_layer)
    output_layer = tf.layers.Dense(config.vocab_size,
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                   _scope='decoder/dense', _reuse=True)
    beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding,
        start_tokens=tf.tile(tf.constant([SOS_ID], tf.int32), [batch_size]),
        end_token=EOS_ID,
        initial_state=tf.contrib.seq2seq.tile_batch(decoder_init_state, config.batch_size),
        beam_width=config.beam_size,
        output_layer=output_layer)
    beam_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=beam_decoder,
        swap_memory=True,
        maximum_iterations=config.max_length * 2
    )
    beam_predicted_ids = beam_output.predicted_ids
    return beam_predicted_ids[:, :, 0]  # [batch_size, seq_len]


