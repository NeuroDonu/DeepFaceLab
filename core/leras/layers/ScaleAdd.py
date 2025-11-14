from core.leras import nn
tf = nn.tf

class ScaleAdd(nn.LayerBase):
    def __init__(self, ch, dtype=None, **kwargs):
        if dtype is None:
            dtype = nn.floatx
        self.dtype = dtype
        self.ch = ch

        super().__init__(**kwargs)

    def build_weights(self):
        # Create weight variable
        weight_shape = (self.ch,)
        weight_init = tf.initializers.zeros()(weight_shape)

        with tf.name_scope(self.name):
            self.weight = tf.Variable(weight_init, dtype=self.dtype, trainable=True, name="weight")

    def get_weights(self):
        return [self.weight]

    def forward(self, inputs):
        if nn.data_format == "NHWC":
            shape = (1,1,1,self.ch)
        else:
            shape = (1,self.ch,1,1)

        weight = tf.reshape ( self.weight, shape )

        x0, x1 = inputs
        x = x0 + x1*weight

        return x
nn.ScaleAdd = ScaleAdd