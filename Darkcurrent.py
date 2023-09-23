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
image=["IMG_32{}.CR2".format(i) for i in (22,23,25,27)]+["IMG_35{}.CR2".format(i) for i in range(88,95)]#+["IMG_360{}.CR2".format(i) for i in (0,1,3,4,5)]
img=imagestream(image)
limtimelist=np.array([5,10,15,20]+[6,8,10,13,15,2.5,13])#,20,25,32,35])#+[40,45,50,67,60]) # list of input brightness
blackchannel=2048
variances=[]
means=[]
clipped_img=np.zeros((4022,6024),dtype=np.uint16)
for i in range(len(img.img_name)):
    img.load(i)
    arr_clipped = np.clip(img.photo, 1000,3000)
    clipped_img=np.add(arr_clipped//len(img.img_name),clipped_img) # for hot and dead pixel identification
    hist, bins = np.histogram(arr_clipped, bins=2000, range=(1000,3000)) # histogram is made only of pixels in this range to avoid outliers
    pix_count=0
    variance=0
    mean=0
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
print("Readnoise= ",int(np.sqrt(intercept)),"\nDark Current= ",int(slope)+1,"+-",int(std_err),"\nr_value= ",r_value,"\np_value= ",p_value,)
plt.scatter(limtimelist,variances)
plt.plot([0,35],[intercept, intercept+35*slope])
plt.title("Graph for dark current determination")
plt.xlabel("Exposure time(s)")
plt.ylabel("Variance")
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
Readnoise=  55-62 
Dark Current=  714 +- 79
r_value=  0.948
p_value=  8.89e-06
"""

"""
Sources of error-
1. It is unclear what algorithm has been used by Canon to modify frames in camera. The histogram of the
dark frames needs to be studied more carefully for any statistical anomalies, and if the dark frame was really just
shifted to a different mean.
2. On including more frames, the dark current reduces toward 614, implying as the sensor fills with dark signal, the process
   does not remain poissonian, at around 30 second, the dark signal approaches saturation, the variace of the dark signal infact
   decreases from 50 seconds onwards. 
3. Temperature may fluctuate while taking frames
"""

"""
Gain-
From this website, https://www.photonstophotos.net/Charts/Measured_ISO.htm, CANON EOS760D has native ISO 112.
        As d(variance in dark frame)/dt=(Gain^2)*dark current, dark current in e-/p/s is the recorded slope (714+-79) divided by the gain,
        At ISO 112 (native ISO), gain is 1, at ISO 6400, gain is 6400/112=57.14.
        Hence, dark current is 0.219 +- 0.024 e-/p/s. This is comparable to a good quality CMOS sensor. In comparison, orion g3, a cheap astronomical
        imaging ccd achieves 0.01 e-/p/s, 1 e-/p/hour is achievable by LN2 cooled professional cameras.  
"""
