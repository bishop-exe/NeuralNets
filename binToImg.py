# will convert binaries to images.
# pass the directory containing the binaries as a 
# command line argument. 

import os
import sys
import array
import numpy
import matplotlib.pyplot as plt
from PIL import Image

rootdir = sys.argv[1]
i = 0

for root, subdirs, files in os.walk(rootdir):
    for filename in files: 
        try:
            file_path = os.path.join(root, filename)
            f = open(file_path, 'rb')
            length = os.path.getsize(file_path)
            width = 256

            rem = length % width

            a = array.array("B")
            a.fromfile(f, length - rem)

            f.close()

            size = round(len(a)/ width)

            g = numpy.reshape(a, (size, width))

            rescaled = (255.0 / g.max() * (g - g.min())).astype(numpy.uint8)
            im = Image.fromarray(rescaled)
            new_im = im.resize((256,256)) #optimal scale?
            new_im.save(sys.argv[2] + filename + '.png')

        except:
            pass