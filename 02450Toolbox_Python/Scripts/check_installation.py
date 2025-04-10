""" 
This is a helper function which can help you debug the Python installation

v20240126a

"""
import os
import sklearn
import numpy as np
import pathlib
import torch

print('------------------------------------------------------------------')
print('Path of this file {}'.format(os.path.abspath(__file__)))
print('Current working directory {}.'.format(pathlib.Path().resolve()))
print('')
print('The numpy version is {}.'.format(np.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The torch version is {}.'.format(torch.__version__))

"""
Check that the course specific tools can be imported 
"""
import dtuimldmtools 
print('The dtuimldmtools package {}.'.format(dtuimldmtools))

"""
Check that pandas can be imported (used in ex1)

Note_ If this fails you need to install panda manually https://pandas.pydata.org/docs/getting_started/install.html

"""
import pandas 
print('The panda package {}.'.format(pandas.__version__))
print('------------------------------------------------------------------')

