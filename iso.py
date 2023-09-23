import rawpy
import numpy as np
import matplotlib.pyplot as plt
import os

images_dir500 = "/Users/sree/Downloads/Horizon/Varying ISO/ISO500"

images500 = [file for file in os.listdir(images_dir500)]

img500_data = []

for img in images500:
    x = rawpy.imread(os.path.join(images_dir500,img))
    x = x.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
    img500_data.append(x)

mean500=[]

for i in img500_data:
    mean500.append(np.mean(i,axis=(0,1)))

mean500 = np.array(mean500)
sorted_indices = np.argsort(mean500[:, 0])
mean500 = mean500[sorted_indices]
mean500 = np.transpose(mean500)

ISO = [100*(2**i) for i in range(8)]

plt.plot(ISO,mean500[0],'ro-')
plt.plot(ISO,mean500[1],'go-')
plt.plot(ISO,mean500[2],'bo-')

plt.ylabel("Pixel counts")
plt.xlabel("ISO")
plt.xticks(ISO)
plt.show()