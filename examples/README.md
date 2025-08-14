# About the Example Notebooks

In this directory, you will find example use cases of TIAToolbox functionalities in the form of Jupyter Notebooks. These notebooks are designed to run on Google Colab (unless otherwise stated), but you can also run them on your local system. Each example notebook includes links to open the notebook in Google Colab and GitHub. Simply click on one of these links to open the notebook on the selected platform. If the left click does not work, you can right-click on "Open in Colab" and select "Open in new tab".

## Running Notebooks Locally

To run the notebooks on your local machine, set up your Python environment as explained in the [installation guide](../docs/installation.rst).

## Running Notebooks on [Google Colab](https://colab.research.google.com/)

Each notebook contains all the information needed to run the example remotely on Colab. All you need is a local computer with a standard browser connected to the Internet. The local computer does not need to have any programming languages installed—all necessary resources are provided at the remote site, free of charge, thanks to Google Colaboratory. Ensure that "colab" appears in the address bar. Familiarize yourself with the drop-down menus near the top of the window. You can edit the notebook during the session, for example, by substituting your own image files for those used in the demo. Experiment by changing the parameters of functions. You cannot permanently change this version of the notebook on GitHub or Colab, so you cannot inadvertently mess it up. Use the notebook's File Menu if you wish to save your own (changed) version.

### GPU or CPU Runtime

Processes in the notebooks can be accelerated by using a GPU. Whether you are running the notebook on your system or Colab, you need to check and specify if you are using GPU or CPU hardware acceleration. In Colab, ensure that the runtime type is set to GPU in the *"Runtime → Change runtime type → Hardware accelerator"*. If you are *not* using GPU, consider changing the `device` variable to `cpu`, otherwise, some errors may occur when running the following cells.

> **IMPORTANT**: If you are using Colab and install TIAToolbox, please note that you need to restart the runtime after installation before proceeding (menu: *"Runtime → Restart runtime"*). This is necessary to load the latest versions of prerequisite packages installed with TIAToolbox. After restarting, you should be able to run all the remaining cells together (*"Runtime → Run after"* from the next cell) or one by one.

## Structure of the Examples Directory

The examples directory includes general notebooks explaining different functionalities/modules incorporated in TIAToolbox. Most of these notebooks are written with less advanced users in mind—some familiarity with Python is assumed—but the capabilities they demonstrate are also useful to more advanced users.

The examples directory contains two subdirectories: `full-pipelines` and `inference-pipelines`, which include examples of using TIAToolbox for training neural networks or inference of WSIs for high-level computational pathology applications, such as patient survival prediction and MSI status prediction from H&E whole slide images.

## A) Examples of TIAToolbox Functionalities

Below is a list of our Jupyter notebooks, with brief descriptions of the TIAToolbox functionalities each notebook provides.

### 1. Reading Whole Slide Images ([01-wsi-reading](./01-wsi-reading.ipynb))

This notebook shows how to use TIAToolbox to read different kinds of WSIs. TIAToolbox provides a uniform interface to various WSI formats. To see what formats are dealt with, click [here](https://tia-toolbox.readthedocs.io/en/latest/usage.html?highlight=wsiread#tiatoolbox.wsicore.wsireader.WSIReader) and then search for _format_. In this notebook, you will learn some well-known techniques for WSI masking and patch extraction.

[![image](../docs/images/wsi-reading.png)](./01-wsi-reading.ipynb)

### 2. Stain Normalization of Histology Images ([02-stain-normalization](./02-stain-normalization.ipynb))

Stain normalization is a common pre-processing step in computational pathology to reduce color variation that has no clinical significance. This variation may be caused by using different scanners, different staining protocols and practices, staining agents that have been left on the laboratory shelf for different lengths of time, different settings when using the scanner, etc. It has been shown in many studies that stain normalization can make an algorithm more robust against such differences. TIAToolbox offers several stain-normalization algorithms, including:

- Reinhard stain normalization
- Ruifork
- Macenko
- Vahadane.

Alternatively, if you prefer, you can use your own stain matrix for stain normalization. In the images below, the object of an algorithm is to change to source image to make its colours similar to those in the target image.

[![image](../docs/images/stain-norm-example.png)](./02-stain-normalization.ipynb))

### 3. Extracting Tissue Mask from Whole Slide Images ([03-tissue-masking](./03-tissue-masking.ipynb))

WSIs often show large blank (glass) background areas that contain no information. Therefore, it is essential to detect the informative (tissue) region in the WSI before taking any action (like patch extraction and classification). We call this step, "tissue masking" which is the focus of this example notebook. This notebook shows how to extract tissue regions from a WSI using TIAToolbox with a single line of Python code.

[![image](../docs/images/tissue-mask.png)](./03-tissue-masking.ipynb)

### 4. Extracting Patches from Whole Slide Images ([04-patch-extraction](./04-patch-extraction.ipynb))

Learn how to use TIAToolbox to extract patches from large histology images based on point annotations or using a fixed-size sliding window. The patch extraction module of TIAToolbox supports mask-based patch extraction which means you can extract (overlapping, if you prefer) patches from a certain region of WSI (for example a region consisting of a particular type of tissue).

[![image](../docs/images/patch-extraction.png)](./04-patch-extraction.ipynb)

### 5. Patch Prediction in Whole Slide Images ([05-patch-prediction](./05-patch-prediction.ipynb))

Use TIAToolbox for patch-level prediction with a range of deep learning models. Predict the type of patches in a WSI with just two lines of Python code. TIAToolbox can be used to make predictions on pre-extracted image patches or on larger image tiles / whole-slide images (WSIs), where image patches are extracted on the fly. For example, in colorectal cancer, TIAToolbox can classify whole slide image regions into nine different categories (Empty glass, Lymphocytes, Normal colon mucosa, Debris, Smooth muscle, Cancer-associated stroma, Adipose, Mucus, Colorectal adenocarcinoma epithelium).

[![image](../docs/images/patch-prediction.png)](./05-patch-prediction.ipynb)

### 6. Semantic Segmentation of Whole Slide Images ([06-semantic-segmentation](./06-semantic-segmentation.ipynb))

Use pretrained models to automatically segment different tissue region types in WSIs. _Semantic segmentation_ groups together similar parts of an image that belong to the same class, as in the image immediately above and in the image below. Semantic segmentation of tissue regions plays an important role in developing algorithms for cancer diagnosis and prognosis, as it can help measure tissue attributes in an objective and reproducible fashion.

[![image](../docs/images/sematic-segment.png)](./06-semantic-segmentation.ipynb)

### 7. Advanced Model Techniques ([07-advanced-modeling](./07-advanced-modeling.ipynb))

This notebook is aimed at advanced users familiar with object-oriented programming concepts in Python and the TIAToolbox models framework. We demonstrate the use of TIAToolbox models with your current workflow and how you can integrate your solutions into the TIAToolbox model framework. By doing so, you will be able to utilize extensively tested TIAToolbox tools in your experiments and speed up your computational pathology research.

[![image](../docs/images/advanced-techniques.png)](./07-advanced-modeling.ipynb)

### 8. Nucleus Instance Segmentation in Whole Slide Images Using the HoVer-Net Model ([08-nucleus-instance-segmentation](./08-nucleus-instance-segmentation.ipynb))

Demonstrates the use of the TIAToolbox implementation of the [HoVer-Net model](https://www.sciencedirect.com/science/article/pii/S1361841519301045) for nucleus instance segmentation and classification. Each WSI can contain up to a million nuclei of various types. These can analysed systematically and used for predicting clinical outcomes. Nucleus segmentation and classification must be carried out before using nuclear features in downstream analysis.

[![image](../docs/images/hovernet.png)](./08-nucleus-instance-segmentation.ipynb)

### 9. Multi-task Segmentation in Whole Slide Images Using the HoVer-Net+ Model ([09-multi-task-segmentation](./09-multi-task-segmentation.ipynb))

Demonstrates the use of the TIAToolbox implementation of the [HoVer-Net+ model](https://openaccess.thecvf.com/content/ICCV2021W/CDPath/papers/Shephard_Simultaneous_Nuclear_Instance_and_Layer_Segmentation_in_Oral_Epithelial_Dysplasia_ICCVW_2021_paper.pdf) for nucleus instance segmentation/classification and semantic segmentation of intra-epithelial layers. Each WSI consists of a multitude of different tissue types, each containing many nuclei of varying types. In computational pathology, it is often important to generate tissue specific morphological features for downstream analyses. It can therefore be beneficial to perform multiple tasks such as semantic segmentation of tissue regions and nuclear instance segmentation/classification simultaneously in order to exploit useful information learnt from each task to further advance both tasks.

[![image](../docs/images/hovernetplus.png)](./09-multi-task-segmentation.ipynb)

### 10. Image Alignment ([10-wsi_registration](./10-wsi-registration.ipynb))

Shows how to use TIAToolbox for registration of an image pair using [Deep Feature Based Registration (DFBR)](https://arxiv.org/pdf/2202.09971.pdf) followed by non-rigid alignment using [SimpleITK](https://simpleitk.readthedocs.io/en/master/registrationOverview.html). The registration tool in the TIAToolbox also comprises a pre-alignment step, a pre-requisite to DFBR. In this example, the affine transformation is computed using thumbnails of the fixed and moving images. The estimated transformation is then used to extract corresponding tiles from both fixed and moving images at a higher magnification level. The non-rigid deformation between the two tiles is then dealt with using the SimpleITK.

[1] Awan, Ruqayya, et al. "Deep Feature based Cross-slide Registration." arXiv preprint arXiv:2202.09971 (2022).

[![image](../docs/images/wsi-registration.png)](./10-wsi-registration.ipynb)

### 11. Feature Extraction Using Foundation Models ([11-import-foundation-models](./11-import-foundation-models.ipynb))

Explains how to extract features from WSIs using pre-trained models from the `timm` library. The notebook guides users through selecting appropriate model architectures, visualizing the extracted features using `UMAP` feature embedding, and verifying the model's performance by checking if different tissue types are correctly identified and separated in the feature map.

[![image](../docs/images/feature_extraction.png)](./11-import-foundation-models.ipynb)

## B) Examples of High-Level Analysis (Pipelines) Using TIAToolbox

List of Jupyter notebooks demonstrating how to use TIAToolbox for high-level analysis in computational pathology.

### 1. Prediction of Molecular Pathways and Key Mutations in Colorectal Cancer from Whole Slide Images (idars)

Prediction of molecular pathways and key mutations directly from Haematoxylin and Eosin stained histology images can help bypass additional genetic (e.g., polymerase chain reaction or PCR) or immunohistochemistry (IHC) testing, with a view to saving both money and time. In this notebook, we use TIAToolbox's pretrained models to reproduce the inference results obtained by the IDaRS pipeline due to <a href="https://www.thelancet.com/journals/landig/article/PIIS2589-7500(2100180-1/fulltext">Bilal et al</a>. In TIAToolbox, we include models that are capable of predicting the following in whole slide images:

- Microsatellite instability (MSI)
- Hypermutation density
- Chromosomal instability
- CpG island methylator phenotype (CIMP)-high prediction
- BRAF mutation
- TP53 mutation

[![image](../docs/images/idars.png)](./inference-pipelines/idars.ipynb)

### 2. Prediction of HER2 Status in Breast Cancer from H&E Stained Whole Slide Images

Demonstrates how to use TIAToolbox to reproduce the ["SlideGraph+ method" ("SlideGraph+: Whole Slide Image-Level Graphs to Predict HER2Status in Breast Cancer" by Lu et al. (2021))](https://arxiv.org/abs/2110.06042) to predict HER2 status of breast cancer samples from H&E stained WSIs. As a brief overview, this method involves several steps generating a graph that represents a WSI. The graph is then fed into a special convolutional graph network, called SlideGraph, to predict whether that WSI is HER2 negative or positive.

- Example notebook on training the SlideGraph model: [slide-graph for training](./full-pipelines/slide-graph.ipynb)
- Example notebook on using SlideGraph model for WSI inference: [slide-graph for inference](./inference-pipelines/slide-graph.ipynb)

[![image](../docs/images/her2-prediction-example.png)](./inference-pipelines/slide-graph.ipynb)
