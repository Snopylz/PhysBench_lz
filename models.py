import tensorflow as tf
for i in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(i,True)
from tensorflow import keras
import keras.backend as K
from tensorflow.keras import layers
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from scipy.sparse import spdiags
import numpy as np

def detrend(signal, Lambda=25):
    def _detrend(signal, Lambda=Lambda):
        """detrend(signal, Lambda) -> filtered_signal
        This code is based on the following article "An advanced detrending method with application
        to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
        """
        signal_length = signal.shape[0]
        H = np.identity(signal_length)
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        diags_index = np.array([0, 1, 2])
        D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
        filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
        return filtered_signal
    rst = np.zeros_like(signal)
    for i in np.arange(0, signal.shape[0], 900):
        if i<=signal.shape[0]-900:
            rst[i:i+900] = _detrend(signal[i:i+900])
        else:
            rst[i:] = _detrend(signal[-900:])[-(rst.shape[0]-i):]
    return rst

def get_flops(model, input_sig=[tf.TensorSpec([1, 450, 8, 8, 3])]):
    forward_pass = tf.function(
        model.call,
        input_signature=input_sig)
    graph_info = profile(forward_pass.get_concrete_function().graph,
                            options=ProfileOptionBuilder.float_operation())
    return graph_info.total_float_ops

def to_tf(datatape, dtype=tf.float16):
    return tf.data.Dataset.from_generator(lambda :datatape, output_types=(dtype, dtype), output_shapes=(datatape.shape, datatape.shape[:1]))

"""
ours ↓
"""

class SpectralTransform(layers.Layer):

    def __init__(self, c, size=3):
        super().__init__()
        self.conv1 = keras.Sequential([
            layers.Conv1D(c*2, size, 1, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
        ])
    
    def call(self, y):
        x = tf.transpose(y, (0, 2, 1))
        x = tf.signal.rfft(x)
        r, i = tf.math.real(x), tf.math.imag(x)
        x = tf.concat([r, i], axis=-2)
        x = tf.transpose(x, (0, 2, 1))
        x = self.conv1(x)
        x = tf.transpose(x, (0, 2, 1))
        r, i = tf.split(x, 2, axis=-2)
        x = tf.complex(r, i)
        x = tf.signal.irfft(x)
        x = tf.transpose(x, (0, 2, 1))
        return layers.add([x,y])

class M_1(keras.Model):
    
    def __init__(self):
        super().__init__()
        self.a = keras.Sequential([
            layers.Reshape((1350, -1)), 
            layers.Conv1D(64, 3, 3),
        ])
        self.ST1 = SpectralTransform(64, 5)
        self.conv1_1 = layers.Conv1D(64, 10, 1, padding='same')
        self.atv1 = keras.Sequential([layers.BatchNormalization(), layers.Activation('relu')])
        self.ST2 = SpectralTransform(64, 3)
        self.conv2 = layers.Conv1D(32, 5, 1, padding='same')
        self.atv2 = keras.Sequential([layers.BatchNormalization(), layers.Activation('relu')])
        self.z = keras.Sequential([
            layers.Conv1D(1, 1, 1),
            layers.Reshape((-1,)),
        ])
        self.w = {}

    def call(self, x):
        x = (x - tf.reshape(tf.reduce_mean(x, axis=(1, 2, 3)), (-1, 1, 1, 1, 3)))/tf.reshape(tf.math.reduce_std(x, axis=(1, 2, 3)), (-1, 1, 1, 1, 3))
        
        x = tf.transpose(x, (0, 1, 4, 2, 3))
        x = self.a(x)
        x = self.ST1(x)
        x = self.conv1_1(x)
        x = self.atv1(x)
        x = self.ST2(x)
        x = self.conv2(x)
        x = self.atv2(x)
        return self.z(x)
    
    def cross(self):
        if 'bn' not in self.w:
            self.w['bn'] = self.atv1, self.atv2
            self._ = keras.Sequential([layers.BatchNormalization(), layers.Activation('relu')]), keras.Sequential([layers.BatchNormalization(), layers.Activation('relu')])
            self._[0].build(input_shape=(None, None, 64))
            self._[1].build(input_shape=(None, None, 32))
            self._[0].set_weights(self.atv1.get_weights())
            self._[1].set_weights(self.atv2.get_weights())
            self.atv1, self.atv2 = lambda x:self._[0](x, training=True), lambda x:self._[1](x, training=True)
    
    def intra(self):
        if 'bn' in self.w:
            self.atv1, self.atv2 = self.w['bn']
            self.w.clear()
        
    
class M_2(keras.Model):
    
    def __init__(self):
        super().__init__()
        self.a = keras.Sequential([
            layers.Reshape((1350, -1)), 
            layers.Conv1D(64, 3, 3),
        ])
        #self.ST1 = SpectralTransform(64, 5)
        self.conv1_1 = layers.Conv1D(64, 10, 1, padding='same')
        #self.conv1_2 = layers.Conv1D(32, 5, 1, padding='same')
        self.atv1 = keras.Sequential([layers.BatchNormalization(), layers.Activation('relu')])
        #self.ST2 = SpectralTransform(64, 3)
        self.conv2 = layers.Conv1D(32, 5, 1, padding='same')
        self.atv2 = keras.Sequential([layers.BatchNormalization(), layers.Activation('relu')])
        self.z = keras.Sequential([
            layers.Conv1D(1, 1, 1),
            layers.Reshape((-1,)),
        ])
        self.w = {}

    def call(self, x):
        x = x - tf.reshape(tf.reduce_mean(x, axis=(1, 2, 3)), (-1, 1, 1, 1, 3))
        x = tf.transpose(x, (0, 1, 4, 2, 3))
        x = self.a(x)
        #x = self.ST1(x)
        x = self.conv1_1(x)
        x = self.atv1(x)
        #x = self.ST2(x)
        x = self.conv2(x)
        x = self.atv2(x)
        return self.z(x)
    
    def cross(self):
        if 'atv' not in self.w:
            self.w['atv'] = self.atv1, self.atv2
            self._ = keras.Sequential([layers.BatchNormalization(), layers.Activation('relu')]), keras.Sequential([layers.BatchNormalization(), layers.Activation('relu')])
            self._[0].build(input_shape=(None, None, 64))
            self._[1].build(input_shape=(None, None, 32))
            self._[0].set_weights(self.atv1.get_weights())
            self._[1].set_weights(self.atv2.get_weights())
            self.atv1, self.atv2 = lambda x:self._[0](x, training=True), lambda x:self._[1](x, training=True)
    
    def intra(self):
        if 'atv' in self.w:
            self.atv1, self.atv2 = self.w['atv']
            self.w.clear()
    
class M_3(keras.Model):
    
    def __init__(self):
        super().__init__()
        self.a = keras.Sequential([
            layers.Reshape((1350, -1)), 
            layers.Conv1D(64, 3, 3),
        ])
        self.ST1 = SpectralTransform(64, 5)
        #self.conv1_1 = layers.Conv1D(64, 10, 1, padding='same')
        #self.atv1 = keras.Sequential([layers.BatchNormalization(), layers.Activation('relu')])
        self.ST2 = SpectralTransform(64, 3)
        #self.conv2 = layers.Conv1D(32, 5, 1, padding='same')
        #self.atv2 = keras.Sequential([layers.BatchNormalization(), layers.Activation('relu')])
        self.z = keras.Sequential([
            layers.Conv1D(1, 1, 1),
            layers.Reshape((-1,)),
        ])
        self.w = {}

    def call(self, x):
        x = x - tf.reshape(tf.reduce_mean(x, axis=(1, 2, 3)), (-1, 1, 1, 1, 3))

        x = tf.transpose(x, (0, 1, 4, 2, 3))
        x = self.a(x)
        x = self.ST1(x)
        #x = self.conv1_1(x)
        #x = self.atv1(x)
        x = self.ST2(x)
        #x = self.conv2(x)
        #x = self.atv2(x)
        return self.z(x)

"""
TS-CAN & DeepPhys ↓
"""

class Attention_mask(tf.keras.layers.Layer):
    def call(self, x):
        xsum = K.sum(x, axis=1, keepdims=True)
        xsum = K.sum(xsum, axis=2, keepdims=True)
        xshape = K.int_shape(x)
        return x / xsum * xshape[1] * xshape[2] * 0.5
    
    def get_config(self):
        config = super(Attention_mask, self).get_config()
        return config


class TSM(tf.keras.layers.Layer):
    def call(self, x, n_frame, fold_div=3):
        nt, h, w, c = x.shape
        x = K.reshape(x, (-1, n_frame, h, w, c))
        fold = c // fold_div
        last_fold = c - (fold_div - 1) * fold
        out1, out2, out3 = tf.split(x, [fold, fold, last_fold], axis=-1)

        # Shift left
        padding_1 = tf.zeros_like(out1)
        padding_1 = padding_1[:, -1, :, :, :]
        padding_1 = tf.expand_dims(padding_1, 1)
        _, out1 = tf.split(out1, [1, n_frame - 1], axis=1)
        out1 = tf.concat([out1, padding_1], axis=1)

        # Shift right
        padding_2 = tf.zeros_like(out2)
        padding_2 = padding_2[:, 0, :, :, :]
        padding_2 = tf.expand_dims(padding_2, 1)
        out2, _ = tf.split(out2, [n_frame - 1, 1], axis=1)
        out2 = tf.concat([padding_2, out2], axis=1)

        out = tf.concat([out1, out2, out3], axis=-1)
        out = K.reshape(out, (-1, h, w, c))

        return out
    
    def get_config(self):
        config = super(TSM, self).get_config()
        return config
        



def TSM_Cov2D(x, n_frame, nb_filters=128, kernel_size=(3, 3), activation='tanh', padding='same'):
    x = TSM()(x, n_frame)
    x = layers.Conv2D(nb_filters, kernel_size, padding=padding, activation=activation)(x)
    return x

def TS_CAN(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
           pool_size=(2, 2), nb_dense=128):
    diff_input = layers.Input(shape=input_shape, name='input_1')
    rawf_input = layers.Input(shape=input_shape, name='input_2')

    d1 = TSM_Cov2D(diff_input, n_frame, nb_filters1, kernel_size, padding='same', activation='tanh')
    d2 = TSM_Cov2D(d1, n_frame, nb_filters1, kernel_size, padding='valid', activation='tanh')

    r1 = layers.Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = layers.Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = layers.multiply([d2, g1])

    d3 = layers.AveragePooling2D(pool_size)(gated1)
    d4 = layers.Dropout(dropout_rate1)(d3)

    r3 = layers.AveragePooling2D(pool_size)(r2)
    r4 = layers.Dropout(dropout_rate1)(r3)

    d5 = TSM_Cov2D(d4, n_frame, nb_filters2, kernel_size, padding='same', activation='tanh')
    d6 = TSM_Cov2D(d5, n_frame, nb_filters2, kernel_size, padding='valid', activation='tanh')

    r5 = layers.Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = layers.Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = layers.multiply([d6, g2])

    d7 = layers.AveragePooling2D(pool_size)(gated2)
    d8 = layers.Dropout(dropout_rate1)(d7)

    d9 = layers.Flatten()(d8)
    d10 = layers.Dense(nb_dense, activation='tanh')(d9)
    d11 = layers.Dropout(dropout_rate2)(d10)
    out = layers.Dense(1)(d11)
    model = keras.models.Model(inputs=[diff_input, rawf_input], outputs=out)
    return model

def CAN(nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25, dropout_rate2=0.5,
        pool_size=(2, 2), nb_dense=128):
    diff_input = layers.Input(shape=input_shape)
    rawf_input = layers.Input(shape=input_shape)

    d1 = layers.Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(diff_input)
    d2 = layers.Conv2D(nb_filters1, kernel_size, activation='tanh')(d1)

    r1 = layers.Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = layers.Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = layers.multiply([d2, g1])

    d3 = layers.AveragePooling2D(pool_size)(gated1)
    d4 = layers.Dropout(dropout_rate1)(d3)

    r3 = layers.AveragePooling2D(pool_size)(r2)
    r4 = layers.Dropout(dropout_rate1)(r3)

    d5 = layers.Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(d4)
    d6 = layers.Conv2D(nb_filters2, kernel_size, activation='tanh')(d5)

    r5 = layers.Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = layers.Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = layers.multiply([d6, g2])

    d7 = layers.AveragePooling2D(pool_size)(gated2)
    d8 = layers.Dropout(dropout_rate1)(d7)

    d9 = layers.Flatten()(d8)
    d10 = layers.Dense(nb_dense, activation='tanh')(d9)
    d11 = layers.Dropout(dropout_rate2)(d10)
    out = layers.Dense(1)(d11)
    model = keras.models.Model(inputs=[diff_input, rawf_input], outputs=out)
    return model

class DeepPhys_end_to_end(keras.Model):
    def __init__(self, size=(36, 36)):
        super().__init__()
        self.size = size
        self.deepphys = CAN(32, 64, (*size, 3))

    def call(self, x):
        x_ = x[1:] - x[:-1]
        x_ = (x_ - tf.reshape(tf.reduce_mean(x_, axis=(1,2 )), (-1, 1, 1, 3)))/tf.reshape(tf.math.reduce_std(x_, axis=(1, 2))+1e-6, (-1, 1, 1, 3))
        return self.deepphys((tf.concat([x_, tf.zeros([1, *self.size, 3])], axis=0), tf.expand_dims(tf.reduce_mean(x, axis=(0, )), axis=0)))

class TS_CAN_end_to_end(keras.Model):
    def __init__(self, n=16, size=(36, 36)):
        super().__init__()
        self.size = size
        self.ts_can = TS_CAN(n, 32, 64, (*size, 3))

    def call(self, x):
        x_ = x[1:] - x[:-1]
        x_ = (x_ - tf.reshape(tf.reduce_mean(x_, axis=(1,2 )), (-1, 1, 1, 3)))/tf.reshape(tf.math.reduce_std(x_, axis=(1, 2))+1e-6, (-1, 1, 1, 3))
        return self.ts_can((tf.concat([x_, tf.zeros([1, *self.size, 3])], axis=0), tf.expand_dims(tf.reduce_mean(x, axis=(0, )), axis=0)))

"""
PhysNet ↓
"""

class PhysNet(keras.Model):

    def __init__(self, norm='batch'):
        self.norm = norm
        if norm == 'batch':
            norm = layers.BatchNormalization
        if norm == 'layer':
            norm = lambda :layers.LayerNormalization(axis=(1,))
        super().__init__()
        self.ConvBlock1 = keras.Sequential([
            layers.Conv3D(16, kernel_size=(1, 5, 5), strides=1, padding='same'),
            norm(),
            layers.Activation('relu')
        ])
        self.ConvBlock2 = keras.Sequential([
            layers.Conv3D(32, kernel_size=(3, 3, 3), strides=1, padding='same'),
            norm(),
            layers.Activation('relu')
        ])
        self.ConvBlock3 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            norm(),
            layers.Activation('relu')
        ])
        self.ConvBlock4 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            norm(),
            layers.Activation('relu')
        ])
        self.ConvBlock5 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            norm(),
            layers.Activation('relu')
        ])
        self.ConvBlock6 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            norm(),
            layers.Activation('relu')
        ])
        self.ConvBlock7 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            norm(),
            layers.Activation('relu')
        ])
        self.ConvBlock8 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            norm(),
            layers.Activation('relu')
        ])
        self.ConvBlock9 = keras.Sequential([
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same'),
            norm(),
            layers.Activation('relu')
        ])
        self.upsample = keras.Sequential([
            layers.Conv3DTranspose(64, kernel_size=(4, 1, 1), strides=(2, 1, 1), padding='same'),
            norm(),
            layers.Activation('elu')
        ])
        self.upsample2 = keras.Sequential([
            layers.Conv3DTranspose(64, kernel_size=(4, 1, 1), strides=(2, 1, 1), padding='same'),
            norm(),
            layers.Activation('elu')
        ])
        self.convBlock10 = layers.Conv3D(1, kernel_size=(1, 1, 1), strides=1)
        self.MaxpoolSpa = layers.MaxPool3D((1, 2, 2), strides=(1, 2, 2))
        self.MaxpoolSpaTem = layers.MaxPool3D((2, 2, 2), strides=2)
        self.poolspa = layers.AvgPool3D((1, 2, 2))
        self.flatten = layers.Reshape((-1,))

    def call(self, x):
        if self.norm == 'batch':
            training=True
        else:
            training=False
        x = self.ConvBlock1(x, training=training)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x, training=training)
        x = self.ConvBlock3(x, training=training)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x, training=training)
        x = self.ConvBlock5(x, training=training)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x, training=training)
        x = self.ConvBlock7(x, training=training)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock8(x, training=training)
        x = self.ConvBlock9(x, training=training)
        x = self.upsample(x, training=training)
        x = self.upsample2(x, training=training)
        x = self.poolspa(x)
        x = self.convBlock10(x, training=training)
        x = self.flatten(x)
        x = x-tf.expand_dims(tf.reduce_mean(x, axis=-1), -1)
        return x
    
"""
loss ↓
"""

"""
在TS-CAN和DeepPhys中使用以下loss都需要将输出和标签转置
def np_loss_tscan(x, y):
    return np_loss(tf.transpose(x), tf.transpose(y))
"""
def np_loss(x, y):
    x_, y_ = tf.expand_dims(tf.reduce_mean(x, axis=-1), -1), tf.expand_dims(tf.reduce_mean(y, axis=-1), -1)
    return 1-tf.reduce_sum((x-x_)*(y-y_), axis=-1)/(tf.reduce_sum((x-x_)**2, axis=-1)*tf.reduce_sum((y-y_)**2, axis=-1))**0.5

def SNR(x, y):
    x, y = (x-tf.expand_dims(tf.reduce_mean(x, axis=-1), -1))/(tf.expand_dims(tf.math.reduce_std(x, axis=-1), -1)+1e-6), (y-tf.expand_dims(tf.reduce_mean(y, axis=-1), -1))/(tf.expand_dims(tf.math.reduce_std(y, axis=-1), -1)+1e-6)
    A_s = tf.reduce_mean(tf.abs(tf.signal.rfft(x)), axis=-1)
    A_n = tf.reduce_mean(tf.abs(tf.signal.rfft(y-x)), axis=-1)
    return 8.685889*tf.math.log(A_s/A_n)

def nSNR_loss(x, y):
    return -SNR(x, y)