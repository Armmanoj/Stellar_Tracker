
import rawpy 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import convolve
import math

""" This program will return value of avg dark current, avg read noise, image containing hot and dead pixels, histogram of dark frame"""

class imagestream:
    """This class will encapsulate the following (the same methods and attributes can  carry over with 
    different image types and inner working of the methods):
       1. Opening the file/files/stream of images
       2. Loading the ith images
       3. Displaying the image, histogram and fourier transform
       4. Stacking the images
       5. Calculating the histogram of the image
    """
    def __init__(self,img_name):
        self.img_name=img_name
        self.frame_count=len(img_name)
        try:
            with rawpy.imread(self.img_name[0]) as raw:
                self.photo=np.array(raw.raw_image_visible)
        except:
            raise FileNotFoundError("{} not found.".format(self.img_name[0]))
            # The with statement introduces some complications, causing the program to skip all statements involving self.photo
            # Setting self.photo to be np.array(raw.raw_image_visible0 instead of just raw.raw_image_visible solves this issue
            
    def load(self,i):
            try: 
                with rawpy.imread(self.img_name[i]) as raw:
                    self.photo=np.array(raw.raw_image_visible) #np.array(raw_image_visible) is specified like before for the same reason
            except:
                raise FileNotFoundError("{} not found.".format(self.img_name[i]))
            
    def histogram(self,min_value=0,max_value=2**14):
        """
            Returns histogram of pixel values in the range (min_value,max_value)
        """
        arr_clipped = np.clip(self.photo, min_value, max_value)
        hist, bins = np.histogram(arr_clipped, bins=max_value-min_value, range=(min_value, max_value))
        return hist
    
    def stack_simple(self):
        """Computes weighted average of all images, default weights is 1"""
        imlist=[]
        for i in range(self.frame_count):
            self.load(i)
            imlist.append(self.photo)
        return np.mean(imlist,axis=0)
    
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


images={}
images['R.004']=imagestream(["IMG_36{}.CR2".format(i) for i in (30,31,32,33,34,71,72,73,74,75,76,77,78)])
images['R.002']=imagestream(["IMG_36{}.CR2".format(i) for i in range(35,40)])
images['G.004']=imagestream(["IMG_36{}.CR2".format(i) for i in (44,45,46,47,60,61,62,63,64,65,66,67)])
images['G.002']=imagestream(["IMG_36{}.CR2".format(i) for i in (40,41,42,43,)])
images['B.004']=imagestream(["IMG_36{}.CR2".format(i) for i in (68,69,70,56,57,58,59,48,49,50,51)])
images['B.002']=imagestream(["IMG_36{}.CR2".format(i) for i in (52,53,54,55)])
R=images['R.004'].stack_simple()#-images['R.002'].stack_simple()
B=images['B.004'].stack_simple()#-images['B.002'].stack_simple()
G=images['G.004'].stack_simple()#-images['G.002'].stack_simple()
print(G[0:4,0:4])
"""
    Printing the above values verifies that the bayer matrix starts with its [0:2,0:2]
    indices as G R /n B G.
"""
Rdata={'R':[],'G':[],'B':[]}
Gdata={'R':[],'G':[],'B':[]}
Bdata={'R':[],'G':[],'B':[]}
for i in range(10):
    for j in range(10):
        Rdata['R'].append(np.mean(R[400*i+1:400*(i+1):2,600*j:600*(j+1):2]))
        Rdata['G'].append(np.mean(R[400*i:400*(i+1):2,600*i:600*(i+1):2])+np.mean(R[400*i+1:400*(i+1):2,600*i+1:600*(i+1):2]))
        Rdata['B'].append(np.mean(R[400*i:400*(i+1):2,600*i+1:600*(i+1):2]))
        
        Gdata['R'].append(np.mean(G[400*i+1:400*(i+1):2,600*j:600*(j+1):2]))
        Gdata['G'].append(np.mean(G[400*i:400*(i+1):2,600*i:600*(i+1):2])+np.mean(G[400*i+1:400*(i+1):2,600*i+1:600*(i+1):2]))
        Gdata['B'].append(np.mean(G[400*i:400*(i+1):2,600*i+1:600*(i+1):2]))
        
        Bdata['R'].append(np.mean(B[400*i+1:400*(i+1):2,600*j:600*(j+1):2]))
        Bdata['G'].append(np.mean(B[400*i:400*(i+1):2,600*i:600*(i+1):2])+np.mean(B[400*i+1:400*(i+1):2,600*i+1:600*(i+1):2]))
        Bdata['B'].append(np.mean(B[400*i:400*(i+1):2,600*i+1:600*(i+1):2]))


# Create a single matplotlib figure with subplots for the image, histogram, and Fourier transform
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Display the plot
axes[0].set_title('Response to red light')
axes[0].scatter(Rdata['R'],Rdata['G'],c='green')
axes[0].scatter(Rdata['R'],Rdata['B'],c='blue')
axes[0].set_xlabel("Red pixel count")
axes[0].set_ylabel("G & B pixel count")

axes[1].set_title('Response to green light')
axes[1].scatter(Gdata['G'],Rdata['R'],c='red')
axes[1].scatter(Gdata['G'],Rdata['B'],c='blue')
axes[1].set_xlabel("Green pixel count")
axes[1].set_ylabel("R & B pixel count")

axes[2].set_title('Response to blue light')
axes[2].scatter(Bdata['B'],Rdata['G'],c='green')
axes[2].scatter(Bdata['B'],Rdata['R'],c='red')
axes[2].set_xlabel("Blue pixel count")
axes[2].set_ylabel("G & R pixel count")

plt.show()
"""
    For streamlining future processing tasks with CR2 images, I have introduced a class imagestream
"""

"""
    The plots are inconclusive. Visualizing the images in rawpy with debayering set to none shows that the screen cannot be regarded as a niform sorce of light, with clearly
    visible pixels, and aliasing as well as lens distortion patterns. Also, an LED screen is backlit with white light whose different frequenncy components
    are filtered to display color. There is leakage of whitelight and hence, we are not seeing pre colors. The apearance of different lins in the response
    graphs indicates this phenomenna, different areas of the image have different fraction of pixel area and leakage light area due to lens distortions,
    this gives distinct lines for each area, and it is impossible to find out what the correct line is. A better way would be to take pictures of many different
    colors printed on a paper and use best linear predictor technique to find the optimum color matrix that converts recorded RGB values to the actual RGB values
    printed out.
"""

"""
    Here I have first stacked all images of the same exposre time. I tried sbstracting 2 exposures to eliminate the offset and then analyzing that, bt the signal was too low then.
    Hence only the longer set of exposures is used. The frame is divided into 100 rectangles of size 400x600 pixels. The net red, green and blue pixel count is
    averaged over each region. A plot is then made for the average count in the other 2 colors, for each color, with datapoints from each sub-rectangle. 
"""
