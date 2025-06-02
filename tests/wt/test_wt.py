import numpy as np
import ptwt
import pywt
import torch

# generate an input of even length.
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0])
data_torch = torch.from_numpy(data.astype(np.float32))
wavelet = pywt.Wavelet("haar")  # type: ignore

# compare the forward fwt coefficients

coeffs = ptwt.wavedec(data_torch, wavelet, mode="zero", level=2)

for _ in coeffs:
    print(_)

# print(ptwt.waverec(ptwt.wavedec(data_torch, wavelet, mode="zero"), wavelet))
