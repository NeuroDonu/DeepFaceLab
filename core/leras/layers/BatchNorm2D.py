from core.leras import nn
tf = nn.tf

class BatchNorm2D(nn.LayerBase):
    """
    currently not for training
    """
    def __init__(self, dim, eps=1e-05, momentum=0.1, dtype=None, **kwargs):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype
        super().__init__(**kwargs)

    def build_weights(self):
        # Create weight variable
        weight_shape = (self.dim,)
        weight_init = tf.initializers.ones()(weight_shape)

        with tf.name_scope(self.name):
            self.weight = tf.Variable(weight_init, dtype=self.dtype, trainable=True, name="weight")

        # Create bias variable
        bias_shape = (self.dim,)
        bias_init = tf.initializers.zeros()(bias_shape)

        with tf.name_scope(self.name):
            self.bias = tf.Variable(bias_init, dtype=self.dtype, trainable=True, name="bias")

        # Create running_mean variable (non-trainable)
        running_mean_shape = (self.dim,)
        running_mean_init = tf.initializers.zeros()(running_mean_shape)

        with tf.name_scope(self.name):
            self.running_mean = tf.Variable(running_mean_init, dtype=self.dtype, trainable=False, name="running_mean")

        # Create running_var variable (non-trainable)
        running_var_shape = (self.dim,)
        running_var_init = tf.initializers.zeros()(running_var_shape)

        with tf.name_scope(self.name):
            self.running_var = tf.Variable(running_var_init, dtype=self.dtype, trainable=False, name="running_var")

    def get_weights(self):
        return [self.weight, self.bias, self.running_mean, self.running_var]

    def forward(self, x):
        if nn.data_format == "NHWC":
            shape = (1,1,1,self.dim)
        else:
            shape = (1,self.dim,1,1)

        weight       = tf.reshape ( self.weight      , shape )
        bias         = tf.reshape ( self.bias        , shape )
        running_mean = tf.reshape ( self.running_mean, shape )
        running_var  = tf.reshape ( self.running_var , shape )

        x = (x - running_mean) / tf.sqrt( running_var + self.eps )
        x *= weight
        x += bias
        return x

nn.BatchNorm2D = BatchNorm2D