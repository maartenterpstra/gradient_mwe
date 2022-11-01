import os
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.signal import fft2d, ifft2d, ifftshift, fftshift
import random
K.set_image_data_format('channels_first')
from keras_conv import ComplexConv2D
import torch
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_soft_device_placement(False)
random.seed(271828)
np.random.seed(271828)
torch.manual_seed(271828)
tf.random.set_seed(271828)



class UnrolledNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), padding="SAME", kernel_initializer=my_init, bias_initializer=tf.keras.initializers.Zeros())
    def call(self, inputs):
        return batch_crop_center(self.layer(inputs[0]), 320, 320)

def batch_crop_center(img,cropx,cropy):
    # TODO: ugly hardcoded size
    y,x = 640, 368
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, :, starty:starty+cropy,startx:startx+cropx]

def my_init(shape, dtype=None):
    # old seed 42069
    tf.random.set_seed(271828)
    init = tf.keras.initializers.HeNormal(seed=271828)
    return init(shape)

class IFFT2c(torch.nn.Module):
    def forward(self, kspace, *args):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1), norm='ortho'), dim=(-2, -1))

def csm_reduce(x, csm):
    ifft2c = IFFT2c()
    return (ifft2c(x) * torch.conj(csm)).sum(1, keepdims=True)

def calculate_center_mask(shape, center_fraction):
    mask = np.zeros(shape, dtype=np.float32)
    center_width = round(center_fraction * shape[1])
    center = shape[2] // 2
    radius = center_width // 2
    low_freqs = range(center - radius, center + radius + 1)
    mask[:, :, center - radius:center + radius + 1] = 1
    return mask, low_freqs

def get_offset_sampling_mask(factor, shape, center_fraction, offset=1):
    assert len(shape) == 3
    mask, low_freqs = calculate_center_mask(shape, center_fraction)
    num_low_frequencies = len(low_freqs)

    cols_to_sample = round(shape[2] / factor) - num_low_frequencies
    new_factor = round(shape[2] / cols_to_sample)

    if offset % 2 == 0:
        offset_pos, offset_neg = offset + 1, offset + 2
    else:
        offset_pos, offset_neg = offset - 1 + 3, offset - 1

    num_cols = shape[2]

    poslen = (num_cols + 1) // 2
    neglen = num_cols - (num_cols + 1) // 2

    mask_positive = np.zeros(poslen, dtype=np.float32)
    mask_negative = np.zeros(neglen, dtype=np.float32)

    mask_positive[offset_pos::new_factor] = 1
    mask_negative[offset_neg::new_factor] = 1
    mask_negative = np.flip(mask_negative)

    pmask = np.concatenate((mask_positive, mask_negative))
    mask[:, :, np.where(pmask)[0]] = 1
    return mask

if __name__ == "__main__":
    random.seed(271828)
    np.random.seed(271828)
    tf.random.set_seed(271828)
    torch.manual_seed(271828)

    data = np.load('./mri_sample.npz')
    kspace = data['kspace']
    csm = data['csm']
    target = np.abs(data['target'])
    mask = get_offset_sampling_mask(4, kspace.shape, 0.08)

    adjoint = np.squeeze(csm_reduce(torch.from_numpy(kspace*mask).unsqueeze(0), torch.from_numpy(csm).unsqueeze(0))).unsqueeze(0).unsqueeze(0).to(torch.complex64).numpy()

    inputs = [tf.convert_to_tensor(np.abs(adjoint))]
    model = UnrolledNetwork()
    # model = model.build((1, 1, 1, 640, 368))
    with tf.GradientTape() as g:
        [g.watch(x) for x in inputs]
        output = model(inputs)
        # output = 
        loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(tf.transpose(target[np.newaxis, np.newaxis, ...], perm=(0, 2, 3, 1)), tf.transpose(output, perm=(0, 2, 3, 1))))
        print('LOSS: ', loss)

    gradients = g.gradient(loss, model.trainable_variables)
    [print(g.shape, tf.reduce_mean(tf.abs(g))) for g in gradients]
