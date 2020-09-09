"""
DOCSTRING
"""
import nottingham_util
import tensorflow

class Model:
    """ 
    Cross-Entropy Naive Formulation

    A single time step may have multiple notes active,
    so a sigmoid cross entropy loss is used to match targets.

    seq_input: a [ T x B x D ] matrix, where T is the time steps in the batch,
        B is the batch size, and D is the amount of dimensions.
    """
    def __init__(self, config, training=False):
        self.config = config
        self.time_batch_len = time_batch_len = config.time_batch_len
        self.input_dim = input_dim = config.input_dim
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        dropout_prob = config.dropout_prob
        input_dropout_prob = config.input_dropout_prob
        cell_type = config.cell_type
        self.seq_input = tensorflow.placeholder(
            tensorflow.float32, shape=[self.time_batch_len, None, input_dim])
        if (dropout_prob <= 0.0 or dropout_prob > 1.0):
            raise Exception("Invalid dropout probability: {}".format(dropout_prob))
        if (input_dropout_prob <= 0.0 or input_dropout_prob > 1.0):
            raise Exception("Invalid input dropout probability: {}".format(input_dropout_prob))
        with tensorflow.variable_scope("rnnlstm"):
            output_W = tensorflow.get_variable("output_w", [hidden_size, input_dim])
            output_b = tensorflow.get_variable("output_b", [input_dim])
            self.lr = tensorflow.constant(config.learning_rate, name="learning_rate")
            self.lr_decay = tensorflow.constant(
                config.learning_rate_decay, name="learning_rate_decay")
        if training:
            self.seq_input_dropout = tensorflow.nn.dropout(
                self.seq_input, keep_prob = input_dropout_prob)
        else:
            self.seq_input_dropout = self.seq_input
        self.cell = tensorflow.models.rnn.rnn_cell.MultiRNNCell([
            self.create_cell(cell_type, input_dim)] + [
                self.create_cell(cell_type, hidden_size) for i in range(1, num_layers)])
        batch_size = tensorflow.shape(self.seq_input_dropout)[0]
        self.initial_state = self.cell.zero_state(batch_size, tensorflow.float32)
        inputs_list = tensorflow.unpack(self.seq_input_dropout)
        # rnn outputs a list of [batch_size x H] outputs
        outputs_list, self.final_state = tensorflow.models.rnn.rnn.rnn(
            self.cell, inputs_list, initial_state=self.initial_state)
        outputs = tensorflow.pack(outputs_list)
        outputs_concat = tensorflow.reshape(outputs, [-1, hidden_size])
        logits_concat = tensorflow.matmul(outputs_concat, output_W) + output_b
        logits = tensorflow.reshape(logits_concat, [self.time_batch_len, -1, input_dim])
        # probabilities of each note
        self.probs = self.calculate_probs(logits)
        self.loss = self.init_loss(logits, logits_concat)
        self.train_step = tensorflow.train.RMSPropOptimizer(
            self.lr, decay=self.lr_decay).minimize(self.loss)

    def calculate_probs(self, logits):
        """
        DOCSTRING
        """
        return tensorflow.sigmoid(logits)

    def create_cell(self, cell_type, input_size):
        """
        DOCSTRING
        """
        if cell_type == "vanilla":
            cell_class = tensorflow.models.rnn.rnn_cell.BasicRNNCell
        elif cell_type == "gru":
            cell_class = tensorflow.models.rnn.rnn_cell.BasicGRUCell
        elif cell_type == "lstm":
            cell_class = tensorflow.models.rnn.rnn_cell.BasicLSTMCell
        else:
            raise Exception("Invalid cell type: {}".format(cell_type))
        cell = cell_class(hidden_size, input_size = input_size)
        if training:
            return tensorflow.models.rnn.rnn_cell.DropoutWrapper(
                cell, output_keep_prob = dropout_prob)
        else:
            return cell

    def get_cell_zero_state(self, session, batch_size):
        """
        DOCSTRING
        """
        return self.cell.zero_state(batch_size, tensorflow.float32).eval(session=session)

    def init_loss(self, outputs, _):
        """
        DOCSTRING
        """
        self.seq_targets = tensorflow.placeholder(
            tensorflow.float32, [self.time_batch_len, None, self.input_dim])
        batch_size = tensorflow.shape(self.seq_input_dropout)
        cross_ent = tensorflow.nn.sigmoid_cross_entropy_with_logits(outputs, self.seq_targets)
        return tensorflow.reduce_sum(
            cross_ent) / self.time_batch_len / tensorflow.to_float(batch_size)

class NottinghamModel(Model):
    """ 
    Dual softmax formulation

    A single time step should be a concatenation of two one-hot-encoding binary vectors.
    Loss function is a sum of two softmax loss functions over [:r] and [r:] respectively,
    where r is the number of melody classes.
    """
    def assign_melody_coeff(self, session, melody_coeff):
        """
        DOCSTRING
        """
        if melody_coeff < 0.0 or melody_coeff > 1.0:
            raise Exception("Invalid melody coeffecient")
        session.run(tensorflow.assign(self.melody_coeff, melody_coeff))

    def calculate_probs(self, logits):
        """
        DOCSTRING
        """
        steps = list()
        for t in range(self.time_batch_len):
            melody_softmax = tensorflow.nn.softmax(
                logits[t, :, :nottingham_util.NOTTINGHAM_MELODY_RANGE])
            harmony_softmax = tensorflow.nn.softmax(
                logits[t, :, nottingham_util.NOTTINGHAM_MELODY_RANGE:])
            steps.append(tensorflow.concat(1, [melody_softmax, harmony_softmax]))
        return tensorflow.pack(steps)

    def init_loss(self, outputs, outputs_concat):
        """
        DOCSTRING
        """
        self.seq_targets = tensorflow.placeholder(tensorflow.int64, [self.time_batch_len, None, 2])
        batch_size = tensorflow.shape(self.seq_targets)[1]
        with tensorflow.variable_scope("rnnlstm"):
            self.melody_coeff = tensorflow.constant(self.config.melody_coeff)
        r = nottingham_util.NOTTINGHAM_MELODY_RANGE
        targets_concat = tensorflow.reshape(self.seq_targets, [-1, 2])
        melody_loss = tensorflow.nn.sparse_softmax_cross_entropy_with_logits(
            outputs_concat[:, :r], targets_concat[:, 0])
        harmony_loss = tensorflow.nn.sparse_softmax_cross_entropy_with_logits(
            outputs_concat[:, r:], targets_concat[:, 1])
        losses = tensorflow.add(
            self.melody_coeff * melody_loss, (1 - self.melody_coeff) * harmony_loss)
        return tensorflow.reduce_sum(losses) / self.time_batch_len / tensorflow.to_float(batch_size)

class NottinghamSeparate(Model):
    """ 
    Single softmax formulation 
    
    Regular single classification formulation,
    used to train baseline models where the melody and harmony are trained separately.
    """
    def calculate_probs(self, logits):
        """
        DOCSTRING
        """
        steps = list()
        for t in range(self.time_batch_len):
            softmax = tensorflow.nn.softmax(logits[t, :, :])
            steps.append(softmax)
        return tensorflow.pack(steps)

    def init_loss(self, outputs, outputs_concat):
        """
        DOCSTRING
        """
        self.seq_targets = tensorflow.placeholder(
            tensorflow.int64, [self.time_batch_len, None])
        batch_size = tensorflow.shape(self.seq_targets)[1]
        with tensorflow.variable_scope("rnnlstm"):
            self.melody_coeff = tensorflow.constant(self.config.melody_coeff)
        targets_concat = tensorflow.reshape(self.seq_targets, [-1])
        losses = tensorflow.nn.sparse_softmax_cross_entropy_with_logits(
            outputs_concat, targets_concat)
        return tensorflow.reduce_sum(
            losses) / self.time_batch_len / tensorflow.to_float(batch_size)
