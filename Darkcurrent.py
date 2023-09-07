import rawpy 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import math

""" This program will return value of avg dark current, avg read noise, image containing hot and dead pixels, histogram of dark frame"""

class imagestream:
    """This class will encapsulate the following (the same methods and attributes can  carry over with 
    different image types and inner working of the methods):
       1. Opening the file/files/stream of images
       2. Loading the ith images
       3. Displaying the image, histogram and fourier transform"""
    def __init__(self,img_name):
        self.img_name=img_name
        with rawpy.imread(self.img_name[0]) as raw:
            self.photo=np.array(raw.raw_image_visible)
            # The with statement introduces some complications, causing the program to skip all statements involving self.photo
            # Setting self.photo to be np.array(raw.raw_image_visible0 instead of just raw.raw_image_visible solves this issue
    def load(self,i):
            try:
                
                with rawpy.imread(self.img_name[i]) as raw:
                    self.photo=np.array(raw.raw_image_visible) #np.array(raw_image_visible) is specified like before for the same reason
            except:
                print("Image could not be loaded")
    def histogram(self,min_value=0,max_value=2**14):
        arr_clipped = np.clip(self.photo, min_value, max_value)
        # Calculate the histogram
        hist, bins = np.histogram(arr_clipped, bins=max_value-min_value, range=(min_value, max_value))
        return hist

    def display(self):
          # Create a single matplotlib figure with subplots for the image, histogram, and Fourier transform
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))

        # Display the image
        axes[0].imshow(self.photo*4, cmap='gray') # multiplying by as camera record 14 bit image but dtype is uint16
        axes[0].set_title('Image')
        axes[0].axis('off')

        # Display the histogram
        axes[1].plot(np.arange(1000,3000),self.histogram(1000,3000))#histo)#,self.histogram((1000,3000)))#(np.array((8,7)),np.array([2,3]))
        axes[1].set_title('Histogram')
        axes[1].set_xlabel('Pixel Value')
        axes[1].set_ylabel('Frequency')

        # Compute and display the Fourier transform
        f_transform = np.fft.fft2(self.photo)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_transform_shifted)
        magnitude_spectrum = np.log10(magnitude_spectrum)*20

        axes[2].imshow(magnitude_spectrum, cmap='gray')
        axes[2].set_title('Fourier Transform')
        axes[2].axis('off')

        # Display the figure with all subplots
        plt.show()
            

# The code below analyzes the dark frame
image=["IMG_32{}.CR2".format(i) for i in (22,23,25,27)]
img=imagestream(image)
limtimelist=np.array([5,10,15,20]) # list of input brightness
blackchannel=2048
variances=[]
clipped_img=np.zeros((4022,6024),dtype=np.uint16)
for i in range(len(img.img_name)):
    img.load(i)
    arr_clipped = np.clip(img.photo, 1000,3000)
    clipped_img=np.add(arr_clipped//len(img.img_name),clipped_img) # for hot and dead pixel identification
    hist, bins = np.histogram(arr_clipped, bins=2000, range=(1000,3000)) # histogram is made only of pixels in this range to avoid outliers
    pix_count=0
    variance=0
    for i in range(1000,3000):
        pix_count+=hist[i-1000]
        variance+=hist[i-1000]*(blackchannel-i)**2
    variances.append(variance/pix_count)
    variance=0
    pix_count=0

# 

# Plotting variance and doing regression analysis
print("variances= ",variances)
slope, intercept, r_value, p_value, std_err = stats.linregress(limtimelist,variances)
print("Readnoise= ",int(np.sqrt(intercept)),"\nDark Current= ",slope//5,"+-",int(std_err),"\nr_value= ",r_value,"\np_value= ",p_value,)
plt.scatter(limtimelist,variances)
plt.plot([0,35],[intercept, intercept+35*slope])
plt.show()


# Displaying the hot and dead pixels

print(np.max(img.photo),np.min(img.photo))
dead_pix=np.argwhere(clipped_img<1000).T
hot_pix=np.argwhere(clipped_img>3000).T
print(dead_pix.shape,hot_pix.shape)
# Output here iis (2,0) (2,0) so ther are no hot pixels or dead pixels
# Or more likely the camea has internal processing to identify and take care of such pixels
plt.scatter(dead_pix[0,:],dead_pix[1,:],color='blue')
plt.scatter(hot_pix[0,:],hot_pix[1,:],color='red')
plt.show()

"""Printed output is-
variances=  [7145.496550966695, 10454.222334761733, 13839.93581797458, 17028.95323954472]
Readnoise=  62 
Dark Current=  132 +- 5
r_value=  0.9999328656724058 
p_value=  6.713432759419827e-05
"""

"""
Sources of error-
1. It is unclear what algorithm has been used by Canon to modify frames in camera. The histogram of the
dark frames needs to be studied more carefully for any statistical anomalies, and if the dark frame was really just
shifted to a different mean.
2. On including images 5 and 6, the error in dark current increases 8 times, this may be due to nonlinearity, camera algorithm etc
3. Temperature may fluctuate while taking frames
"""
