#%%
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.ops import nn_ops, math_ops, sparse_ops, embedding_ops, gen_math_ops, standard_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.eager import context
# %%

class DropConnect(Dense):
    def __init__(self, p_dropout, *args, **kwargs):
        self.p_dropout = p_dropout  
        super(DropConnect, self).__init__(*args, **kwargs)
    
    @property
    def p_dropout(self):
        return self._p_dropout

    @p_dropout.setter
    def p_dropout(self, p_dropout):
        assert 0 <= p_dropout <= 1, f"prob needs to be a valid probability instead got {p_dropout}"
        self._p_dropout = p_dropout
        
    def call(self, inputs, training = False):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)
        #Drop Connect Code to mask the kernel
        if training:
            mask = tf.cast(tf.random.uniform(shape=self.kernel.shape) <= self.p_dropout, dtype=self.kernel.dtype)
            kernel = mask * self.kernel
            tf.print(kernel)
            tf.print(self.kernel)
        else:
            kernel = self.kernel
        #Code below from Tensorflow Dense Class
        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            if isinstance(inputs, sparse_tensor.SparseTensor):
                inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
                ids = sparse_tensor.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape)
                weights = inputs
                outputs = embedding_ops.embedding_lookup_sparse_v2(
                    kernel, ids, weights, combiner='sum')
            else:
                outputs = gen_math_ops.MatMul(a=inputs, b=kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
        # Reshape the output back to the original ndim of the input.
        if not context.executing_eagerly():
            shape = inputs.shape.as_list()
            output_shape = shape[:-1] + [kernel.shape[-1]]
            outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = nn_ops.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

class MCDropConnect(DropConnect):
    '''Call method that uses drop connect without having to set training parameter.
    Does not work with default .predict method. Need to implement custom predict method.'''
    def call(self, x):
        return super().call(x, training=True)

class MCDropout(Dropout):
    def call(self, x):
        return super().call(x, training=True)

#%%
tf.random.set_seed(11234)

class Test(Model):
    def __init__(self, layer):
        super(Test, self).__init__()
        self.layer = layer

    def call(self, x):
        return self.layer(x)


t = tf.constant([[5., 7., 8.]])
y = tf.reshape(tf.constant([4., 6.]), [1,2])

d = Dense(units=2, activation="relu")
mc_dc = MCDropConnect(p_dropout=0.5, units=2, activation="relu")
dc = DropConnect(p_dropout=0.5, units=2, activation="relu")

reg = Test(d)
mcdrop = Test(mc_dc)
dc = Test(dc)

for model in [reg, mcdrop, dc]:
    model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())


# %%

reg_history = reg.fit(t, y, epochs = 5)
mc_history = mcdrop.fit(t, y, epochs=5)
dc_history = dc.fit(t, y, epochs=5)
# %%

mcdrop.set_weights(dc.get_weights())
dc_pred = dc.predict(t)
mc_pred = mcdrop.predict(t)

print(dc_pred)
print(mc_pred)