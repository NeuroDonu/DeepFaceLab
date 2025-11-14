import pickle
from pathlib import Path
from core import pathex
import numpy as np

from core.leras import nn

tf = nn.tf

class Saveable():
    def __init__(self, name=None):
        self.name = name

    #override
    def get_weights(self):
        #return tf tensors that should be initialized/loaded/saved
        return []

    #override
    def get_weights_np(self):
        weights = self.get_weights()
        if len(weights) == 0:
            return []
        # TF 2.x eager execution: convert variables directly to numpy
        return [w.numpy() for w in weights]

    def set_weights(self, new_weights):
        weights = self.get_weights()
        if len(weights) != len(new_weights):
            raise ValueError ('len of lists mismatch')

        # TF 2.x eager execution: use .assign() instead of batch_set_value
        for w, new_w in zip(weights, new_weights):
            if len(w.shape) != new_w.shape:
                new_w = new_w.reshape(w.shape)
            w.assign(new_w)

    def save_weights(self, filename, force_dtype=None):
        d = {}
        weights = self.get_weights()

        if self.name is None:
            raise Exception("name must be defined.")

        name = self.name

        for w in weights:
            # TF 2.x eager execution: use .numpy() instead of tf_sess.run()
            w_val = w.numpy().copy()
            w_name_split = w.name.split('/', 1)
            if name != w_name_split[0]:
                raise Exception("weight first name != Saveable.name")

            if force_dtype is not None:
                w_val = w_val.astype(force_dtype)

            d[ w_name_split[1] ] = w_val

        d_dumped = pickle.dumps (d, 4)
        pathex.write_bytes_safe ( Path(filename), d_dumped )

    def load_weights(self, filename):
        """
        returns True if file exists
        """
        filepath = Path(filename)
        if filepath.exists():
            result = True
            d_dumped = filepath.read_bytes()
            d = pickle.loads(d_dumped)
        else:
            return False

        weights = self.get_weights()

        if self.name is None:
            raise Exception("name must be defined.")

        try:
            # TF 2.x eager execution: use .assign() instead of batch_set_value
            for w in weights:
                w_name_split = w.name.split('/')
                if self.name != w_name_split[0]:
                    raise Exception("weight first name != Saveable.name")

                sub_w_name = "/".join(w_name_split[1:])

                w_val = d.get(sub_w_name, None)

                if w_val is not None:
                    # Load and assign the weight value
                    w_val = np.reshape( w_val, w.shape.as_list() )
                    w.assign(w_val)
                # If w_val is None, skip (weight not found in file, keep current value)
                # In TF 2.x eager mode, variables are already initialized, no need for w.initializer
        except:
            return False

        return True

    def init_weights(self):
        nn.init_weights(self.get_weights())

nn.Saveable = Saveable
