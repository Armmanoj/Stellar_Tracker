import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2


filepath = '/Users/sree/Documents/Python/RemoteAstrophotography_com-Helix_Nebula/ngc7293_r.fits'
with fits.open(filepath) as hdul:
    # print(hdul.info())
    img = hdul[0].data

# subtracting background:
from astropy.stats import sigma_clipped_stats

mean, median, std = sigma_clipped_stats(img,sigma=3.0)
img_bsub = img - median
# print(mean,median,std)



"""
# Finding stars using DAOStarFinder:
from photutils.detection import DAOStarFinder

daofind = DAOStarFinder(threshold=150*std,fwhm=3)           # set threshold manually, see peak value and value of std and adjust.
sources = daofind(img_bsub)
for col in sources.colnames:  
    if col not in ('id', 'npix'):
        sources[col].info.format = '%.2f'
sources.pprint(max_width=76)
# print(type(sources))
"""



# Finding stars using IRAFStarFinder              (Allows us to ignore stars too close to each other)
from photutils.detection import IRAFStarFinder

iraffind = IRAFStarFinder(threshold=100*std,fwhm=3,minsep_fwhm=3,roundhi=0.5)               # roundhi default value causes problems in detection
sources = iraffind(img_bsub)
for col in sources.colnames:
    if col not in ('id', 'npix'):
        sources[col].info.format = '%.2f'
sources.pprint(max_width=76)


"""
# plotting img and location of sources:

from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
apertures = CircularAperture(positions, r=4.0)
norm = ImageNormalize(stretch=SqrtStretch())
plt.imshow(img_bsub, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
apertures.plot(color='blue', lw=1.5, alpha=0.5)
plt.imshow(img_bsub,cmap='gray')
plt.show()
"""


# Extracting stars:
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import extract_stars

x = sources['xcentroid']
y = sources['ycentroid']
stars_tbl = Table()                     # making a table from the stars' centroids
stars_tbl['x'] = x
stars_tbl['y'] = y
ndimg = NDData(data=img_bsub)           # extract_stars requires input as NDData object
stars = extract_stars(ndimg,stars_tbl,size=15)

"""
# Displaying extracted stars:

import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

fig, ax = plt.subplots(nrows=3, ncols=10, figsize=(15, 5), squeeze=True)
ax = ax.ravel()
for i in range(29):
    norm = simple_norm(stars[i], 'log', percent=99.0)
    ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
plt.show()
"""

# estimating average psf:
from photutils.psf import EPSFBuilder

epsf_builder = EPSFBuilder(oversampling=2,progress_bar=False)
epsf, fitted_stars = epsf_builder(stars)


# viewing the psf
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
norm = simple_norm(epsf.data, 'log', percent=99.0)
plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
# plt.imshow(epsf.data)
plt.colorbar()
plt.show()