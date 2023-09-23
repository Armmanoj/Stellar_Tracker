import rawpy
import scipy
from matplotlib import pyplot as plt
import numpy as np
import math


def meannp(x):
    # this returns mean of average pixel value for each color in the pixel array 
    sr=np.mean(x[:,:,0])
    sg=np.mean(x[:,:,1])
    sb=np.mean(x[:,:,2])
    return sr,sg,sb

def stdnp(x):
    # this returns mean of average pixel value for each color in the pixel array 
    sr=np.std(x[:,:,0])
    sg=np.std(x[:,:,1])
    sb=np.std(x[:,:,2])
    return sr,sg,sb



image=["IMG_28{}.CR2".format(i) for i in range(82,91)]
regb=([],[],[]) # stores all rgb values as 3 lists
xlist=np.array([1/800,1/500,1/320,1/250,1/125,1/60,1/30,1/15,1/8]) # list of input brightness
for im in image:
    with rawpy.imread(im) as raw:
        raw_data=(np.array(raw.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)).astype(np.float64))/2**16 #we take rgb values as fraction of max value
        mean=meannp(raw_data)
        regb[0].append(mean[0])
        regb[1].append(mean[1])
        regb[2].append(mean[2])           

regb[0]=[regb[0][i]/regb[0][2] for i in range(len(xlst))]
regb[1]=[regb[11][i]/regb[1][2] for i in range(len(xlst))]
regb[2]=[regb[2][i]/regb[2][2] for i in range(len(xlst))] 

"""Source for scipy code: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter16.04-Least-Squares-Regression-in-Python.html"""

def linfunc(x,a,b):
    return a*x+b

alpha = scipy.optimize.curve_fit(linfunc, xdata = xlist, ydata = np.array(regb[0]))[0]
beta = scipy.optimize.curve_fit(linfunc, xdata = xlist, ydata = np.array(regb[1]))[0]
gamma = scipy.optimize.curve_fit(linfunc, xdata = xlist, ydata = np.array(regb[2]))[0]
# Above returns value of slope(alpha[0]) and intercept(alpha[1]) for the 3 lines which best approximate the curve

plt.plot(xlist,[alpha[0]*i+alpha[1] for i in xlist],color=(0.9,0.1,0.1))
plt.plot(xlist,[beta[0]*i+beta[1] for i in xlist],color=(0.1,0.9,0.1))
plt.plot(xlist,[gamma[0]*i+gamma[1] for i in xlist],color=(0.1,0.1,0.9))

# Absolute errors in pixel counts
errR=np.array([math.fabs(regb[0][i]-alpha[0]*xlist[i]-alpha[1]) for i in range(len(xlist))])
errG=np.array([math.fabs(regb[1][i]-beta[0]*xlist[i]-beta[1]) for i in range(len(xlist))])
errB=errR=np.array([math.fabs(regb[2][i]-gamma[0]*xlist[i]-gamma[1]) for i in range(len(xlist))])

# Accuracy of R/G and R/B values
RbyG=np.array([regb[0][i]/regb[1][i] for i in range(len(xlist))])
RbyB=np.array([regb[0][i]/regb[2][i] for i in range(len(xlist))])

"""
rby_g,rby_b=100*(alpha[0]/beta[0]-1),100*(alpha[0]/gamma[0]-1)
mpv=np.sqrt(sum([(regb[0][i]-alpha[0]*xlist[i]-alpha[1])**2/(1+alpha[0]**2) for i in range(len(xlist))]))
print(mpv,rby_g,rby_b)"""
print("The relative sensitivity of green and blue with respect to red are (in percent):")
print(100*alpha[0]/beta[0],100*alpha[0]/gamma[0])
print("The absolute errors in R,G and B channels are: ")
print(np.mean(errR),np.mean(errG),np.mean(errB))
print("The y intercept of the best fit line is: ")
print(alpha[1],beta[1],gamma[1])
print("The standard deviation in the ratios R/G and R/B are (in percent): ")
print(100*np.std(RbyG),100*np.std(RbyB))
""" We want as output of analysis, the std in y values, the b, correlation of b with bias frame mean, the error in ratios of r,g,b"""

plt.plot(xlist,regb[0],'ro')
plt.plot(xlist,regb[1],'go')
plt.plot(xlist,regb[2],'bo')
plt.show()




"""
with rawpy.imread("IMG_2848.CR2") as raw:
            raw_data=(np.array(raw.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)).astype(np.float64))/2**16 #we take rgb values as fraction of max value
            mean=meannp(raw_data)
            std=stdnp(raw_data)
            print(mean[0])
            print(mean[1])
            print(mean[2])
            print(std[0])
            print(std[1])
            print(std[2])
            # 0.64,0.20,0.34 % of full value is as bias current std
            # 0.49, 0.11, 0.28 is mean of bias
"""

"""
with rawpy.imread("IMG_2827.CR2") as raw:
            raw_data=(np.array(raw.postprocess(gamma=(1,1),no_auto_bright=True,output_bps=16)).astype(np.float64))/2**16 #we take rgb values as fraction of max value
            mean=meannp(raw_data)
            std=stdnp(raw_data)
            print(mean[0])
            print(mean[1])
            print(mean[2])
            print(std[0])
            print(std[1])
            print(std[2])
            # 0.64,0.20,0.34 % of full value is as bias current std
            # 0.49, 0.11, 0.28 is mean of bias
"""
