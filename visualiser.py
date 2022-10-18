from PIL import Image
import numpy as np

a = np.zeros(28,28)
with open("images.ubyte","rb") as f:
    f.read(16)
    while(byte := f.read(1)):
        np.arr
im = Image.fromarray()