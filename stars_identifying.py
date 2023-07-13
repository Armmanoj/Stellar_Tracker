import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from photutils.centroids import centroid_2dg


"""In this classical row by row blob-detection algorithm, first the given image is binarized by setting all pixels below a threshold to black and above to white.
Then the image is scanned row-by-row and a label is assigned to each non-zero pixel. If it shares any one of its side with another already labelled pixel, it is also labelled
the same. If it shares 2 sides with already labelled pixels, it is given the same label if both sides have same label, else it is given the lower of the 2 labels and
the 2 labels are noted as being equivalent, using a UnionFind data structure. If the pixel shares no side with an already labelled pixel, it is given a new label. Then
the equivalence classes of the equivalence relation of connected components are found using the UnionFind data structure. In the second pass (scan), this is used to set
equivalent pixels to the same label. Thus, the blobs of the image are founnd. The centre of mass of each blob is found, weighing each pixel wiith its pixel count in the original image."""



# union-find data structure
class UnionFind:
    def __init__(self):
        self.roots={0:None} # datatype has a dictionary of a UnionFind associated with each value
    def new_root(self,x,u): # creates a new root with value x and UnionFind u
        self.roots[x]=u
    def list_values(self): # returns a list of all the values stored in self
        l=[]
        if len(self.roots.keys())==1: # ie. only {0:None} is there
            return l
        else:
            for i in self.roots.keys():
                if i==0:
                    pass
                else:
                    l.append(i)
                    if self.roots[i]!=None:
                        l.extend(self.roots[i].list_values())
        return l
    def findnode(self,x): # returns disjoint subset with value x in it
        for j in self.roots.keys():
            if j==x:
                return j
            elif self.roots[j]==None:
                continue
            else:
                y=self.roots[j].findnode(x)
                if y!=None:
                    return j
        return None
    def union(self,x,y):  # union of set containing x and set containing y
        Y=self.findnode(y)
        X=self.findnode(x)
        if Y==None or X==None: # if x or y is not there
            return
        elif X==Y:
            return
        elif x<y:
            if self.roots[X]!=None:
                self.roots[X].new_root(Y,self.roots[Y])
            else:
                self.roots[X]=UnionFind()
                self.roots[X].new_root(Y,self.roots[Y])
            del self.roots[Y]
            return
        elif y<x:
            if self.roots[Y]!=None:
                self.roots[Y].new_root(X,self.roots[X])
            else:
                self.roots[Y]=UnionFind()
                self.roots[Y].new_root(X,self.roots[X])
            del self.roots[X]
            return
    def list_of_values(self): # returns list of the list of all values in each child structure of self
        l=[]
        for j in self.roots.keys():
            if j==0:
                pass
            else:
                k=[j]
                if self.roots[j]!=None:
                    k.extend(self.roots[j].list_values())
                l.append(k)
        return l

        
                
def binarize(data,alpha): #alpha is a parameter that decides how bright stars must be detected
    threshold=np.mean(data)+alpha*np.std(data) # sets a threshold for detecting a star
    height,width=data.shape
    return np.array([[int(data[j][i]>threshold) for i in range(width)] for j in range(height)])

def blob(bimage): # input is a binary image
    height,width=bimage.shape
    
    # pass 1
    k=1 # used to label blobs
    equi=UnionFind() # data structure to hold equivalence relations
    left,up=0,0
    for i in range(1,height):
        for j in range(1,width):
            up=bimage[i-1][j]
            left=bimage[i][j-1]
            if bimage[i][j]==1:
                if up>0 and left==0:
                    bimage[i][j]=up
                elif left>0 and up==0:
                    bimage[i][j]=left
                elif left>0 and up>0 and left==up:
                    bimage[i][j]=left
                elif left>0 and up>0:
                    bimage[i][j]=left*(left<up)+up*(up<left) # expression for minimum of "up" and "left"
                    equi.union(left*(left<up)+up*(up<left),left*(left>up)+up*(up>left))
                    
                else:
                    k+=1 # create a new label
                    bimage[i][j]=k # label that pixel with the integer "k"
                    equi.new_root(k,None) # add the label to the UnionFind structure

    # simplifying equivalence expression
    eqclasses=equi.list_of_values()
    print([m[0] for m in eqclasses])
    
    # pass 2
    for i in range(1,height):
        for j in range(1,width):
            k=bimage[i][j]
            for m in eqclasses:
                if k in m:
                    bimage[i][j]=m[0]
    


# function to return centroid and FWHM from numpy array
def star_centroids(data):
    FWHM=1 # Full width Half maximum, the radius around centroid at which intensity roughly halves, now set to 1 for convenience
    height,width=data.shape
    alpha=float(input("Enter value of alpha: "))
    bimage=binarize(data,alpha) # creates a binay image of data
    blob(bimage) # labels blobs
    # displays the labelled image
    plt.imshow(bimage)
    plt.show()
    # creating a list of all labels, ideally can be returned by the blob function
    label=[]
    for i in range(height):
        for j in range(width):
            if bimage[i][j] not in label:
                if bimage[i][j]!=0:
                    label.append(bimage[i][j])
    # finding list of all centroids
    centroids=(len(label)*[0],len(label)*[0]) #x and y coordinates of the centroids of stars in the image
    starlist=([],[])
    star_pix_count=0
    kk=0
    for k in label:
        for i in range(height):
            for j in range(width):
                if bimage[i][j]==k:
                    star_pix_count+=data[i][j]
                    starlist[0].append(i*data[i][j]) # we are weighting the positions by the pixel counts
                    starlist[1].append(j*data[i][j])
        centroid_i=int(sum(starlist[0])/star_pix_count)
        centroid_j=int(sum(starlist[1])/star_pix_count)
        centroids[0][kk]=centroid_i
        centroids[1][kk]=centroid_j
        star_pix_count=0
        starlist=([],[])
        kk+=1
    return (centroids,FWHM)

# displaying the location of stars
filename="M-31Andromed220221022931.FITS" # this is an example file
with fits.open(filename) as hdul: # returns a list of header-data-units 
    data=hdul[0].data # a header-data-unit object has a string(header) and a numpy array(image data)
    print(np.mean(data),np.std(data))
    centroids,FWHM=star_centroids(data) 
    plt.scatter(centroids[0],centroids[1],s=np.pi*FWHM*FWHM,c='tab:blue') # scatter plot showing location of stars
    plt.show()



