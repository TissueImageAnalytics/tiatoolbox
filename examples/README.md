![TIA Centre Logo Light@3x](https://user-images.githubusercontent.com/74412979/145181206-1dc0a0cf-ef6d-47ff-8d1d-3bd662b5fdeb.png)
# TIAToolbox example notebooks

In this directory, you will find some example use cases of the TIAToolbox functionalities in the form of Jupyter Notebooks. All of these example notebooks are designed and maintained to run on Colab and Kaggle platforms (unless otherwise stated) but you can run them on your system as well. In the first cell of each example notebook, there are two Colab ![colab badge](https://colab.research.google.com/assets/colab-badge.svg) and Kaggle ![kaggle badge](https://kaggle.com/static/images/open-in-kaggle.svg) badges that allow you to open that example notebook in the preferred platform, simply by clicking on the badge. Each notebook contains all the information you need to run the example on any computer with a standard browser and no prior installation of any programming language is required. To run the notebook on any platform, except for Colab or Kaggle, set up your Python environment, as explained in the [installation guide](https://tia-toolbox.readthedocs.io/en/latest/installation.html).

Here, the structure of the example directory and a brief description of the included notebooks are explained. The example directory contains a subfolder named `pipelines`  which includes high-level use cases of TIAToolbox (such as patient survival prediction and MSI status prediction from H&E whole slide images) and the main directory includes general example notebooks explaining different functionalities/modules incorporated in the TIAToolbox.

## A) Examples of TIAToolbox functionalities
List of Jupyter notebooks that are dedicated to explaining the essential functionalities of TIAToolbox.

### 1- Reading Whole Slide Images ([01-wsi-reading.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/01-wsi-reading.ipynb))
This notebook shows you how you can use TIAToolbox to read different kinds of WSIs. TIAToolbox supports various WSI formats acquired by different sources a list of which is available [here](https://tia-toolbox.readthedocs.io/en/latest/usage.html?highlight=wsiread#tiatoolbox.wsicore.wsireader.get_wsireader). In this example, you will also learn about a couple of well-known techniques for WSI mask and patch extraction.

![image](https://user-images.githubusercontent.com/74412979/145223963-f5cc3efc-5762-43c1-b040-c1f738a98e1b.png) ![image](https://user-images.githubusercontent.com/74412979/145224002-b61eb074-5b55-45c9-a45c-9b527437be2c.png)

## 2- Stain normalisation of histology images ([02-stain-normalisation.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/02-stain-normalisation.ipynb))
Stain normalisation is a common pre-processing step in computational pathology, whose objective is to reduce the colour variation in histology images (caused by using scanners or different staining protocols) to a minimum that has no significant impact on clinical/computational workflow. It has been shown in many studies that stain normalisation can make the algorithm more robust against the domain shift problem seen in histology images taken under different circumstances. TIAToolbox makes a few different stain-normalisation algorithms available to the user, and we demonstrate how to use them in this notebook. The implemented stain normalisation methods in TIAToolbox are:
- Reinhard stain normalisation
- Ruifork 
- Macenko
- Vahadane

but you can use a custom stain matrix for stain normalisation as well.

![image](https://user-images.githubusercontent.com/74412979/145396514-4f84bcf3-35f1-4474-81d9-2c30be8ac353.png)

## 3- Extracting tissue mask (tissue region) from whole slide images ([03-tissue-masking.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/03-tissue-masking.ipynb))
Apart from tissue regions, WSIs usually show large blank (glass) background areas that contain no information. Therefore it is essential to detect the informative (tissue) region in the WSI before any action (like patch extraction and classification). We call this step, "tissue masking" which is the focus of this example notebook. This notebook shows how you can extract tissue regions from a WSI with the help of TIAToolbox and a single line of Python code.

![image](https://user-images.githubusercontent.com/74412979/145227864-6df6b12c-8d15-4ac6-bc46-19677bce1f8e.png)

## 4- Extracting patches from whole slide images ([04-patch-extraction.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/04-patch-extraction.ipynb))
In this example, we will show how you can use TIAToolbox to extract patches from a large histology image. Tiatoolbox can extract patches based on point annotations or a fixed-size sliding window (patch extraction from a WSI with overlap). Also, the patch extraction module of TIAToolbox supports mask-based patch extraction which means you can extract (overlapping) patches from a certain region of WSI (like tissue region).

![image](https://user-images.githubusercontent.com/74412979/145229244-933fba8b-aa9e-4e88-a9d0-713996e4874a.png)

## 5- Patch prediction in whole slide images ([05-patch-prediction.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/05-patch-prediction.ipynb))
In this example, we will show how to use TIAToolbox for patch-level prediction using a range of deep learning models. TIAToolbox can be used to make predictions on pre-extracted image patches or on larger image tiles / whole-slide images (WSIs), where image patches are extracted on the fly. There are various state-of-the-art deep learning models implemented in the TIAToolbox pretrained on datasets related to different cancer types. These models can be used out of the box to predict the type of patches in a WSI with just 2 lines of Python code. For example, in colorectal cancer, TIAToolbox can classify whole slide image regions into 9 different categories (Background (empty glass region), Lymphocytes, Normal colon mucosa, Debris, Smooth muscle, Cancer-associated stroma, Adipose, Mucus, Colorectal adenocarcinoma epithelium).

![image](https://user-images.githubusercontent.com/74412979/145231194-03d10b24-d7b6-40f7-84fc-32b093ae57e2.png)

## 6- Semantic segmentation of whole slide images ([06-semantic-segmentation.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/06-semantic-segmentation.ipynb))
Semantic segmentation of tissue regions in histology images plays an important role in developing algorithms for cancer diagnosis and prognosis as it can help measure tissue attributes in an objective and reproducible fashion. In this example notebook, we show how you can use pretrained models to automatically segment different tissue region types in a set of input images or WSIs. We first focus on a pretrained model incorporated in the TIAToolbox to achieve semantic annotation of tissue region in histology images of breast cancer, which can be done as simple as writing two lines of codes. After that, we will explain how you can use your pretrained model in the TIAToolbox model inference pipeline to do prediction on a set of WSIs.

![image](https://user-images.githubusercontent.com/74412979/145233254-cd5ae68b-42b9-4627-bfb5-8ac395d904cc.png)

## 7- Advanced model techniques ([07-advanced-modeling.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/07-advanced-modeling.ipynb))
This notebook demonstrates advanced techniques on how to use TIAToolbox models with your current workflow and how you can integrate your solutions into the TIAToolbox model framework. By doing so, you will be able to utilize extensively tested TIAToolbox tools in your experiments and speed up your computational pathology research. Notice, in this notebook, we assume that you are an advanced user of TIAToolbox who is familiar with object-oriented programming concepts in Python and the TIAToolbox models framework.

![image](https://user-images.githubusercontent.com/74412979/145396619-e33c1544-c45d-47f3-b070-89dc293c6517.png)

## 8- Nucleus instance segmentatino in whole slide images using HoVer-Net model ([08-nucleus-instance-segmentation.ipynb](https://github.com/TissueImageAnalytics/tiatoolbox/blob/master/examples/08-nucleus-instance-segmentation.ipynb))
Each WSI can contain up to million nuclei of various types, which can be further analysed systematically and used for predicting clinical outcomes. To use nuclear features for downstream analysis within computational pathology, nucleus segmentation and classification must be carried out as an initial step. In this example, we will demonstrate how you can use the TIAToolbox implementation of [HoVer-Net model](https://www.sciencedirect.com/science/article/pii/S1361841519301045) to solve the problem of nuclei instance segmentation and classification within histology image tiles or WSIs.

![image](https://user-images.githubusercontent.com/74412979/145235642-3f4f99b9-e583-4cbc-81a6-5a9c733746b4.png)

## B) Examples of high-level analysis (pipelines) using TIAToolbox
List of Jupyter notebooks that demonstrate how you can use TIAToolbox to simplify high-level analysis in computational pathology.

### 1- Prediction of molecular pathways and key mutations in colorectal cancer from whole slide images
Prediction of molecular pathways and key mutations directly from Haematoxylin and Eosin stained histology images can help bypass additional genetic (e.g., polymerase chain reaction or PCR) or immunohistochemistry (IHC) testing, which can therefore save both money and time. In this example notebook, we show how you can use TIAToolbox's pretrained models to do reproduce the inference results obtained by IDaRS pipeline introduced in [Bilal et al](https://bit.ly/3IwL6vv). In TIAToolbox, we include models that are capable of predicting the following entities in whole slide images:
- Microsatellite instability (MSI)
- Hypermutation density
- Chromosomal instability
- CpG island methylator phenotype (CIMP)-high prediction
- BRAF mutation 
- TP53 mutation

![image](https://user-images.githubusercontent.com/74412979/145396818-883ef9af-ae78-4f9d-bdb8-a0926ec807a4.png)

### 2- Prediction of HER2 status in breast cancer from H&E stained whole slide images
This example notebook demonstrates how the functionalities available in TIAToolbox can be used to reproduce the ["SlideGraph+ method" ("SlideGraph+: Whole Slide Image-Level Graphs to Predict HER2Status in Breast Cancer" by Lu et al. (2021))](https://arxiv.org/abs/2110.06042) to predict HER2 status of breast cancer samples from H&E stained whole slide images. As a brief overview, this method involves several steps to generating a graph that represents a whole slide image (WSI) and then directly feeding it into a special convolutional graph network, called SlideGraph, to predict whether that WSI is HER2 negative or positive.

![image](https://user-images.githubusercontent.com/74412979/145244421-ad2f28fe-1361-44b8-a82f-707fd72b0a28.png)
