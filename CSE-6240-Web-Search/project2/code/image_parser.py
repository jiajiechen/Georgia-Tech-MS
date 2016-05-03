from __future__ import print_function
import os
import re
from PIL import Image
import numpy as np

class image_parser(object):    
    def __init__(self,datapath=None,folder=None,images=None,image_names=None,
                 min_rows=100,min_cols=100,verbose=False):
        self.datapath=datapath
        self.folder=folder
        self.images=images
        self.image_names=image_names
        self.min_rows=min_rows
        self.min_cols=min_cols
        self.verbose=verbose

    def read(self):
        self.images=np.empty(self.min_rows*self.min_cols)
        self.image_names=[]
        for base, dirs, files in os.walk (self.datapath+'/'+self.folder+'/'):
            for filename in files:
                if self.verbose: print("reading..."
                                  +self.datapath+'/'+self.folder+'/'+filename)
                name_JPEG = re.match (r'^(.*)\.JPEG$',filename)
                if name_JPEG:
                    filepath = os.path.join (base, filename)
                    image = Image.open(filepath,'r'
                                      ).resize(
                                        (self.min_rows,self.min_cols)
                                      ).convert("L")
                    image = np.array(image).reshape(-1)
                    self.images=np.vstack((self.images,image))
                    self.image_names.append(filename)
        #delete the first row because it was a placeholder
        self.images = np.delete(self.images, obj=0, axis=0)
#end


