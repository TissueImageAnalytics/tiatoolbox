![TIA Centre Logo Light@3x](https://user-images.githubusercontent.com/74412979/145181206-1dc0a0cf-ef6d-47ff-8d1d-3bd662b5fdeb.png)
# TIAToolbox example notebooks

In this directory you will find some example use cases of the TIAToolbox functionalities in the form of Jupyter Notebooks. All of these example notebooks are designed and maintained to run on Colab and Kaggle platforms (unless otherwise stated) but you can run them on your system as well. In the first cell of each example notebook there are two Colab ![colab badge](https://colab.research.google.com/assets/colab-badge.svg) and Kaggle ![kaggle badge](https://kaggle.com/static/images/open-in-kaggle.svg) badges that allow you to open that example notebook in the preferred platform, simply by clicking on the badge. Each notebook contains all the information you need to run the example on any computer with a standard browser and no prior installation of any programming language is required. To run the notebook on any platform, except for Colab or Kaggle, set up your Python environment, as explained in the [installation guide](https://tia-toolbox.readthedocs.io/en/latest/installation.html).

Here, the structure of example directory and a brief description of the included notebooks are explained. The example directory contains a subfolder named `pipelines`  which includes high-level use cases of TIAToolbox (such as patient survival prediction and MSI status prediction from H&E whole slide images) and the main directory includes general example notebooks explaining different functinoalities/modules of incorporated in the TIAToolbox.

## Example notebooks on functinalities
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

## 3- Extracting tissue mask (tissue region) from whole slide images ([03-tissue-masking.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/03-tissue-masking.ipynb))
Apart from tissue regions, WSIs usually show large blank (glass) background areas that contain no information. Therefore it is essential to detect the informative (tissue) region in the WSI before any action (like patch extraction and classification). We call this step, "tissue masking" which is the focus of this example notebook. This notebook shows how you can extract tissue region from a WSI with the help of TIAToolbox and  a single line of Python code.

![image](https://user-images.githubusercontent.com/74412979/145227864-6df6b12c-8d15-4ac6-bc46-19677bce1f8e.png)

## 4- Extracting patches from whole slide images ([04-patch-extraction.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/04-patch-extraction.ipynb))
In this example we will show how you can use TIAToolbox to extract patches from a large histology image. Tiatoolbox can extract patches based on point annotatations or fixed-size sliding window (patch extraction from a WSI with overlap). Also, patch extraction module of TIAToolbox supports mask-based patch extraction which means you can extract (overlapping) patches from a certain region of WSI (like tissue region).

![image](https://user-images.githubusercontent.com/74412979/145229244-933fba8b-aa9e-4e88-a9d0-713996e4874a.png)

## 5- Patch prediction in whole slide images ([05-patch-prediction.ipynb]((https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/05-patch-prediction.ipynb))
In this example, we will show how to use TIAToolbox for patch-level prediction using a range of deep learning models. TIAToolbox can be used to make predictions on pre-extracted image patches or on larger image tiles / whole-slide images (WSIs), where image patches are extracted on the fly. There are various state-of-the-art deep learning models implemented in the TIAToolbox pretrained on datasets related different cancer types. These models can be used out of the box to predict type of the patches in a WSI with just 2 lines of Python code. For example, in colorectal cancer, TIAToolbox can classify whole slide image regions into 9 different categories (Background (empty glass region), Lymphocytes, Normal colon mucosa, Debris, Smooth muscle, Cancer-associated stroma, Adipose, Mucus, Colorectal adenocarcinoma epithelium).

![image](https://user-images.githubusercontent.com/74412979/145231194-03d10b24-d7b6-40f7-84fc-32b093ae57e2.png)

## 6- Semantic segmentation of whole slide images ([06-semantic-segmentation.ipynb]((https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/06-semantic-segmentation.ipynb))
Semantic segmentation of tissue regions in histology images plays an important role in developing algorithms for cancer diagnosis and prognosis as it can help measure tissue attributes in an objective and reproducible fashion. In this example notebook, we show how you can use pretrained models to automatically segment different tissue region types in a set of input images or WSIs. We first focus on a pretrained model incorporated in the TIAToolbox to achieve semantic annotation of tissue region in histology images of breast cancer, which can be done as simple as writing two lines of codes. After that, we will explain how you can use your pretrained model in the TIAToolbox model inference pipeline to do prediction on a set of WSIs.

![image](https://user-images.githubusercontent.com/74412979/145233254-cd5ae68b-42b9-4627-bfb5-8ac395d904cc.png)


