# 02450 Toolbox - Python

## Installation
To download/install the toolbox:

* ensure you have an Python environment running **Python version 3.8-3.11** (either as basic Python installation or ideally as a virtual environment)
  - see [DTU Support](https://pythonsupport.dtu.dk/) for assisatnce on installing Python and [virtual environments](https://pythonsupport.dtu.dk/python/environments.html#python-environments)
  - Basic Python / Pip: You can create a virtual environemnt with the correct python version by following from a terminal (with your standard Python activated): TODO
  - Conda/Minicoda: You can create a virtual environemnt with the correct python version by running the following from a terminal (with your standard Conda environment activated): TODO  
*  download the '02450Toolbox_Python' as a zip-file using the follwing [link](
https://lab.compute.dtu.dk/02450/02450students/-/archive/main/02450students-main.zip?path=exercises/02450Toolbox_Python)
* unzip the file, locate the folder ´02450Toolbox_Python´ and copy to your working directory (optional)
* the toolbox (i.e. the `/Scripts` folder) depends on a course specific python package [dtuimldmtools](https://pypi.org/project/dtuimldmtools/) which needs to be installed. We recommend using a Python Virtual environment using [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) and installing the package inside it. To set up such an environment follow the guide provided by [DTU Python support](https://pythonsupport.dtu.dk/python/install-conda.html). Once setup, the package can be installed by running the following command (make sure you have activated the correct environment before installing):
  ```
  pip install dtuimldmtools
  ```

## Report an issue / contact the authors
E-mail: 02450@compute.dtu.dk (subject: 02450students repo - Python)

## Dataset
Description of the datasets in the Data folder:

body.mat
This is a subset of the dataset on body dimenstions available at http://www.sci.usq.edu.au/courses/STA3301/resources/Data/ 
and described in 
G. Heinz, L. J. Peterson, R. W. Johnson, and C. J. Kerk, “Exploring relationships in body dimensions,” Journal of Statistics Education, vol. 11, no. 2, 2003.

faithful.mat and faithful.txt
Dataset on eruption of the Old Faithful geyser described in
A. Azzalini and A. Bowman, “A look at some data on the old faithful geyser,” Applied Statistics, pp. 357–365, 1990.
W. Härdle, Smoothing techniques: with implementation in S. Springer, 1991

female.txt and male.txt
Data is taken from http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/,
Please consult the accompanying readme_male_female.txt file in the Data folder.

iris.xls
Fisher's Iris data, for a description see also http://en.wikipedia.org/wiki/Iris_flower_data_set. The data has been downloaded from http://archive.ics.uci.edu/ml/datasets/Iris.

nanonose.xls
This data has been taken from the nanonose project, see also http://www.nanonose.dk, it is described in 
T. S. Alstrøm, J. Larsen, C. H. Nielsen, and N. B. Larsen, “Data-driven modeling of nano-nose gas sensor arrays,” in SPIE Defense, Security, and Sensing. International Society for Optics and Photonics, 2010, pp. 76 970U–76 970U.

StopWords
A txt file of list of common words provided in the TMG toolbox.

textDocs.txt
This example of documents for a term-document matrix is taken from 
L. Eldén, Matrix Methods in Data Mining and Pattern Recognition. Philadelphia, PA, USA: Society for Industrial and Applied Mathematics, 2007.

Wine.mat and Wine2.mat
P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547–553, 2009.
downloaded from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
Wine2 is same as Wine but with some outliers removed.

zipdata.mat and digits.mat
USPS handwritten digits availabe at http://www.cad.zju.edu.cn/home/dengcai/Data/MLData.html, see also
J. J. Hull, “A database for handwritten text recognition research,” Pattern Analysis and Machine Intelligence, IEEE Transactions on, vol. 16, no. 5, pp. 550–554,
1994.

wildfaces.mat and wildfaces_grayscale.mat
Taken from http://tamaraberg.com/faceDataset/ and described in Tamara L. Berg, Alexander C. Berg, Jaety Edwards, David A. Forsyth 
Neural Information Processing Systems (NIPS), 2004. 
The wildfaces.mat is an extract with 1000 examples of the original dataset and wildfaces_grayscale a gray scale converted version of these 1000 examples taken from the original data.

messy_data.data
This dataset is an excerpt of the Auto MPG Data Set which has been heavily formatted to introduce comming data preprocessing isusues.
Revised from CMU StatLib library, data concerns city-cycle fuel consumption https://archive.ics.uci.edu/ml/datasets/auto+mpg 
This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University. The dataset was used in the 1983 American Statistical Association Exposition.
