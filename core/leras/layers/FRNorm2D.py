from core.leras import nn
tf = nn.tf

class FRNorm2D(nn.LayerBase):
    """
    Tensorflow implementation of
    Filter Response Normalization Layer: Eliminating Batch Dependence in theTraining of Deep Neural Networks
    https://arxiv.org/pdf/1911.09737.pdf
    """
    def __init__(self, in_ch, dtype=None, **kwargs):
        self.in_ch = in_ch

        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype

        super().__init__(**kwargs)

    def build_weights(self):
        # Create weight variable
        weight_shape = (self.in_ch,)
        weight_init = tf.initializers.ones()(weight_shape)

        with tf.name_scope(self.name):
            self.weight = tf.Variable(weight_init, dtype=self.dtype, trainable=True, name="weight")

        # Create bias variable
        bias_shape = (self.in_ch,)
        bias_init = tf.initializers.zeros()(bias_shape)

        with tf.name_scope(self.name):
            self.bias = tf.Variable(bias_init, dtype=self.dtype, trainable=True, name="bias")

        # Create eps variable
        eps_shape = (1,)
        eps_init = tf.initializers.constant(1e-6)(eps_shape)

        with tf.name_scope(self.name):
            self.eps = tf.Variable(eps_init, dtype=self.dtype, trainable=True, name="eps")

    def get_weights(self):
        return [self.weight, self.bias, self.eps]

    def forward(self, x):
        if nn.data_format == "NHWC":
            shape = (1,1,1,self.in_ch)
        else:
            shape = (1,self.in_ch,1,1)
        weight       = tf.reshape ( self.weight, shape )
        bias         = tf.reshape ( self.bias  , shape )
        nu2 = tf.reduce_mean(tf.square(x), axis=nn.conv2d_spatial_axes, keepdims=True)
        x = x * ( 1.0/tf.sqrt(nu2 + tf.abs(self.eps) ) )

        return x*weight + bias
nn.FRNorm2D = FRNorm2D