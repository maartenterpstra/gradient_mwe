import os
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import numpy as np
import tensorflow
import tensorflow as tf
import tensorflow.keras.backend as K
K.set_image_data_format('channels_first')
import torch.nn.functional as F
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_soft_device_placement(False)
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
random.seed(271828)
np.random.seed(271828)
torch.manual_seed(271828)
tf.random.set_seed(271828)
torch.set_printoptions(precision=10)
def batch_crop_center(img,cropx,cropy):
    # TODO: ugly hardcoded size
    y,x = 640, 368
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[:, :, starty:starty+cropy,startx:startx+cropx]


class IFFT2c(torch.nn.Module):
    def forward(self, kspace, *args):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1), norm='ortho'), dim=(-2, -1))

def csm_reduce(x, csm):
    ifft2c = IFFT2c()
    return (ifft2c(x) * torch.conj(csm)).sum(1, keepdims=True)

def my_init(shape, dtype=None):
    # old seed 42069
    tf.random.set_seed(271828)
    init = tf.keras.initializers.HeNormal(seed=271828)
    
    ch_out, ch_in, kx, ky = shape
    return torch.from_numpy(init((kx, ky, ch_in, ch_out)).numpy()).permute((3, 2, 0, 1))
    # return torch.view_as_complex(torch.from_numpy(init((kx, ky, ch_in, ch_out, 2), np.float32).numpy())).permute((3, 2, 0,1))

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


    layer = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding=1, dtype=torch.float32)
    layer.weight.data = my_init(layer.weight.shape, dtype=layer.weight.dtype)
    layer.bias.data = torch.zeros(layer.bias.shape, dtype=layer.weight.dtype)

    data = np.load('./mri_sample.npz')
    kspace = data['kspace']
    csm = data['csm']
    target = torch.from_numpy(data['target']).abs()
    mask = get_offset_sampling_mask(4, kspace.shape, 0.08)

    adjoint = np.squeeze(csm_reduce(torch.from_numpy(kspace*mask).unsqueeze(0), torch.from_numpy(csm).unsqueeze(0))).unsqueeze(0).unsqueeze(0).to(torch.complex64).numpy()

    predict = layer(torch.from_numpy(adjoint).abs())
    cropped_predict = batch_crop_center(predict, 320, 320).squeeze()
    # loss = ComplexMSELoss()
    loss_val = F.mse_loss(cropped_predict, target)
    loss_val.backward()
    
    print(loss_val)
    for n, p in layer.named_parameters():
        if(p.requires_grad) :
            print(p.grad.shape, p.grad.abs().mean())
