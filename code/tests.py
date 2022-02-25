#%%

from CustomDropout import DropConnect
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Dropout
from HH_CNN import CNNDropConnect, CNNDropout

#%%
tf.random.set_seed(1234)
x = tf.constant([[5., 3.4, 2.6], [5.4, 3.2, 1.]])
#%%
drop_connect = DropConnect(p_dropout = 0.7, units = 3)

inputs = Input(shape=(3,))
outputs = drop_connect(inputs)
model = Model(inputs=inputs, outputs=outputs)
# %%
assert (tf.equal(model(x), drop_connect(x))).numpy().all()
assert (tf.equal(model.predict(x), model(x))).numpy().all()

# %%
