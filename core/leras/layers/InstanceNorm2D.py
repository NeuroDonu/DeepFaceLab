from core.leras import nn
tf = nn.tf

class InstanceNorm2D(nn.LayerBase):
    def __init__(self, in_ch, dtype=None, **kwargs):
        self.in_ch = in_ch

        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype

        super().__init__(**kwargs)

    def build_weights(self):
        # Create weight variable
        kernel_initializer = tf.initializers.glorot_uniform(dtype=self.dtype)
        weight_shape = (self.in_ch,)
        weight_init = kernel_initializer(weight_shape)

        with tf.name_scope(self.name):
            self.weight = tf.Variable(weight_init, dtype=self.dtype, trainable=True, name="weight")

        # Create bias variable
        bias_shape = (self.in_ch,)
        bias_init = tf.initializers.zeros()(bias_shape)

        with tf.name_scope(self.name):
            self.bias = tf.Variable(bias_init, dtype=self.dtype, trainable=True, name="bias")

    def get_weights(self):
        return [self.weight, self.bias]

    def forward(self, x):
        if nn.data_format == "NHWC":
            shape = (1,1,1,self.in_ch)
        else:
            shape = (1,self.in_ch,1,1)

        weight       = tf.reshape ( self.weight      , shape )
        bias         = tf.reshape ( self.bias        , shape )

        x_mean = tf.reduce_mean(x, axis=nn.conv2d_spatial_axes, keepdims=True )
        x_std  = tf.math.reduce_std(x, axis=nn.conv2d_spatial_axes, keepdims=True ) + 1e-5

        x = (x - x_mean) / x_std
        x *= weight
        x += bias

        return x

nn.InstanceNorm2D = InstanceNorm2D