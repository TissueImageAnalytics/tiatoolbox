![TIA Centre Logo Light@3x](https://user-images.githubusercontent.com/74412979/145181206-1dc0a0cf-ef6d-47ff-8d1d-3bd662b5fdeb.png)
# TIAToolbox example notebooks

In this directory you will find some example use cases of the TIAToolbox functionalities in the form of Jupyter Notebooks. All of these example notebooks are designed and maintained to run on Colab and Kaggle platforms (unless otherwise stated) but you can run them on your system as well. In the first cell of each example notebook there are two Colab ![colab badge](https://colab.research.google.com/assets/colab-badge.svg) and Kaggle ![kaggle badge](https://kaggle.com/static/images/open-in-kaggle.svg) badges that allow you to open that example notebook in the preferred platform, simply by clicking on the badge. Each notebook contains all the information you need to run the example on any computer with a standard browser and no prior installation of any programming language is required. To run the notebook on any platform, except for Colab or Kaggle, set up your Python environment, as explained in the [installation guide](https://tia-toolbox.readthedocs.io/en/latest/installation.html).

Here, the structure of example directory and a brief description of the included notebooks are explained. The example directory contains a subfolder named `pipelines`  which includes high-level use cases of TIAToolbox (such as patient survival prediction and MSI status prediction from H&E whole slide images) and the main directory includes general example notebooks explaining different functinoalities/modules of incorporated in the TIAToolbox.

## General example notebooks
Jupyter notebooks dedicated to explain the essential functionalities of TIAToolbox.
### 1- Reading Whole Slide Images ([01-wsi-reading.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/01-wsi-reading.ipynb))
This notebook shows you how you can use TIAToolbox to read different kinds of WSIs. TIAToolbox supports various WSI formats aquired by different sources a list of which is available [here](https://tia-toolbox.readthedocs.io/en/latest/usage.html?highlight=wsiread#tiatoolbox.wsicore.wsireader.get_wsireader). In this example, you will also learn about a couple of well-known techniques for WSI mask and patch extraction.

![image](https://user-images.githubusercontent.com/74412979/145223963-f5cc3efc-5762-43c1-b040-c1f738a98e1b.png) ![image](https://user-images.githubusercontent.com/74412979/145224002-b61eb074-5b55-45c9-a45c-9b527437be2c.png)

## 2- Stain normalisation of histology images ([02-stain-normalisation.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/02-stain-normalisation.ipynb))
Stain normalisation is a common pre-processing step in computational pathology, whose objective is to reduce the colour variation in histology images (caused by using scanners or different staining protocols) to a minimum that has no significant impact on clinical/computational workflow. It has been shown in many studies that stain normalisation can make algorithm more robust aginst the domain shift problem seen in histology images taken under different circumstances. TIAToolbox makes a few different stain-normalisation algorithms  available to the user, and we demonstrate how to use them in this notebook. The implemented stain normalisatin methods in TIAToolbox are:
- Reinhard stain normalisation
- Ruifork 
- Macenko
- Vahadane

but you can use custom stain matrix for stain normalisation as well.

![image](https://user-images.githubusercontent.com/74412979/145226029-0cdcf94b-eb65-46ba-8f35-94bb6c457fab.png)
