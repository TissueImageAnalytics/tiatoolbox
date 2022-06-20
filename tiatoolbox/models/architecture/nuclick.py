import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from tiatoolbox.models.abc import ModelABC
from tiatoolbox.utils import misc

from skimage.morphology import remove_small_objects, remove_small_holes, reconstruction, disk
import warnings
import numpy as np
import matplotlib.pyplot as plt

bn_axis = 1


class Conv_Bn_Relu(nn.Module):
    """Convolution -> Batch Normalization -> ReLu/Sigmoid

    Args:
        num_input_channels (int): Number of channels in input.
        num_output_channels (int): Number of channels in output.
        kernelSize (int): Size of the kernel in the convolution layer.
        strds (int): Size of the stride in the convolution layer.
        useBias (bool): Whether to use bias in the convolution layer.
        dilatationRate (int): Dilatation rate in the convolution layer.
        actv (str): Name of the activation function to use.
        doBatchNorm (bool): Whether to do batch normalization after the convolution layer.

    Returns:
        model (torch.nn.Module): a pytorch model.

    """
    def __init__(self, num_input_channels, num_output_channels=32, 
        kernelSize=(3,3), strds=(1,1),
        useBias=False, dilatationRate=(1,1), 
        actv='relu', doBatchNorm=True
    ):

        super().__init__()
        if isinstance(kernelSize, int):
            kernelSize = (kernelSize, kernelSize)
        if isinstance(strds, int):
            strds = (strds, strds)

        self.conv_bn_relu = self.get_block(num_input_channels, num_output_channels, kernelSize,
            strds, useBias, dilatationRate, actv, doBatchNorm
        )

    def forward(self, input):
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            input (torch.Tensor): Input, the tensor is of the shape NCHW.

        Returns:
            output (torch.Tensor): The inference output. 

        """
        return self.conv_bn_relu(input)


    def get_block(self, in_channels, out_channels, 
        kernelSize, strds,
        useBias, dilatationRate, 
        actv, doBatchNorm
    ):

        layers = []

        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernelSize, 
                stride=strds, dilation=dilatationRate, bias=useBias, padding='same', padding_mode='zeros'
            )

        torch.nn.init.xavier_uniform_(conv1.weight)

        layers.append(conv1)

        if doBatchNorm:
            layers.append(nn.BatchNorm2d(num_features=out_channels,eps=1.001e-5))

        if actv == 'relu':
            layers.append(nn.ReLU())
        elif actv == 'sigmoid':
            layers.append(nn.Sigmoid())

        block = nn.Sequential(*layers)
        return block



class Multiscale_Conv_Block(nn.Module):
    """Multiscale convolution block

    Defines four convolution layers. 

    Args:
        num_input_channels (int): Number of channels in input.
        num_output_channels (int): Number of channels in output.
        kernelSizes (list): Size of the kernel in each convolution layer.
        strds (int): Size of stride in the convolution layer.
        useBias (bool): Whether to use bias in the convolution layer.
        dilatationRates (list): Dilation rate for each convolution layer.
        actv (str): Name of the activation function to use.

    Returns:
        model (torch.nn.Module): a pytorch model.
        
    """
    def __init__(self, num_input_channels, kernelSizes, 
        dilatationRates, num_output_channels=32, strds=(1,1),
        actv='relu', useBias=False
    ):

        super().__init__()

        self.conv_block_1 = Conv_Bn_Relu(num_input_channels=num_input_channels, num_output_channels=num_output_channels, kernelSize=kernelSizes[0],
                strds=strds, actv=actv, useBias=useBias, dilatationRate=(dilatationRates[0], dilatationRates[0]))
            
        self.conv_block_2 = Conv_Bn_Relu(num_input_channels=num_input_channels, num_output_channels=num_output_channels, kernelSize=kernelSizes[1],
                strds=strds, actv=actv, useBias=useBias, dilatationRate=(dilatationRates[1], dilatationRates[1]))

        self.conv_block_3 = Conv_Bn_Relu(num_input_channels=num_input_channels, num_output_channels=num_output_channels, kernelSize=kernelSizes[2],
                strds=strds, actv=actv, useBias=useBias, dilatationRate=(dilatationRates[2], dilatationRates[2]))

        self.conv_block_4 = Conv_Bn_Relu(num_input_channels=num_input_channels, num_output_channels=num_output_channels, kernelSize=kernelSizes[3],
                strds=strds, actv=actv, useBias=useBias, dilatationRate=(dilatationRates[3], dilatationRates[3]))


    def forward(self, input_map):
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            input (torch.Tensor): Input, the tensor is of the shape NCHW.

        Returns:
            output (torch.Tensor): The inference output. 

        """

        conv0 = input_map

        conv1 = self.conv_block_1(conv0)
        conv2 = self.conv_block_2(conv0)
        conv3 = self.conv_block_3(conv0)
        conv4 = self.conv_block_4(conv0)

        output_map = torch.cat([conv1, conv2, conv3, conv4], dim=bn_axis)

        return output_map



class Residual_Conv(nn.Module):
    """Residual Convolution block

    Args:
        num_input_channels (int): Number of channels in input.
        num_output_channels (int): Number of channels in output.
        kernelSize (int): Size of the kernel in all convolution layers.
        strds (int): Size of the stride in all convolution layers.
        useBias (bool): Whether to use bias in the convolution layers.
        dilatationRate (int): Dilation rate in all convolution layers.
        actv (str): Name of the activation function to use.

    Returns:
        model (torch.nn.Module): a pytorch model.
    
    """
    def __init__(self, num_input_channels, num_output_channels=32, 
        kernelSize=(3,3), strds=(1,1), actv='relu', 
        useBias=False, dilatationRate=(1,1)
    ):
        super().__init__()



        self.conv_block_1 = Conv_Bn_Relu(num_input_channels, num_output_channels, kernelSize=kernelSize, strds=strds, 
            actv='None', useBias=useBias, dilatationRate=dilatationRate, doBatchNorm=True
        )
        self.conv_block_2 = Conv_Bn_Relu(num_output_channels, num_output_channels, kernelSize=kernelSize, strds=strds, 
            actv='None', useBias=useBias, dilatationRate=dilatationRate, doBatchNorm=True
        )

        if actv == 'relu':
            self.activation = nn.ReLU()
        elif actv == 'sigmoid':
            self.activation = nn.Sigmoid()


    def forward(self, input):
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            input (torch.Tensor): Input, the tensor is of the shape NCHW.

        Returns:
            output (torch.Tensor): The inference output. 

        """

        conv1 = self.conv_block_1(input)
        conv2 = self.conv_block_2(conv1)

        out = torch.add(conv1, conv2)
        out = self.activation(out)
        return out



class NuClick(ModelABC):
    """NuClick Architecture.

    NuClick is used for interactive segmentation. 
    NuClick takes an RGB image patch along with an inclusion and an exclusion map.

    Args:
        num_input_channels (int): Number of channels in input.
        num_output_channels (int): Number of channels in output.

    Returns:
        model (torch.nn.Module): a pytorch model.

    Examples:
        >>> # instantiate a NuClick model for interactive nucleus segmentation. 
        >>> NuClick(num_input_channels = 5, num_output_channels = 1)

    """
    def __init__(self, num_input_channels, num_output_channels):
        super().__init__()
        self.net_name = 'NuClick'

        self.n_channels = num_input_channels
        self.n_classes = num_output_channels

        #-------------Conv_Bn_Relu blocks------------
        self.conv_block_1 = nn.Sequential(
            Conv_Bn_Relu(num_input_channels=self.n_channels, num_output_channels=64, kernelSize=7),
            Conv_Bn_Relu(num_input_channels=64, num_output_channels=32, kernelSize=5),
            Conv_Bn_Relu(num_input_channels=32, num_output_channels=32, kernelSize=3)
        )

        self.conv_block_2 = nn.Sequential(
            Conv_Bn_Relu(num_input_channels=64, num_output_channels=64),
            Conv_Bn_Relu(num_input_channels=64, num_output_channels=32),
            Conv_Bn_Relu(num_input_channels=32, num_output_channels=32)
        )

        self.conv_block_3 = Conv_Bn_Relu(num_input_channels=32, num_output_channels=self.n_classes,
            kernelSize=(1,1), actv=None, useBias=True, doBatchNorm=False)

        #-------------Residual_Conv blocks------------
        self.residual_block_1 = nn.Sequential(
            Residual_Conv(num_input_channels=32, num_output_channels=64),
            Residual_Conv(num_input_channels=64, num_output_channels=64)
        )

        self.residual_block_2 = Residual_Conv(num_input_channels=64, num_output_channels=128)

        self.residual_block_3 = Residual_Conv(num_input_channels=128, num_output_channels=128)

        self.residual_block_4 = nn.Sequential(
            Residual_Conv(num_input_channels=128, num_output_channels=256),
            Residual_Conv(num_input_channels=256, num_output_channels=256),
            Residual_Conv(num_input_channels=256, num_output_channels=256)
        )

        self.residual_block_5 = nn.Sequential(
            Residual_Conv(num_input_channels=256, num_output_channels=512),
            Residual_Conv(num_input_channels=512, num_output_channels=512),
            Residual_Conv(num_input_channels=512, num_output_channels=512)
        )

        self.residual_block_6 = nn.Sequential(
            Residual_Conv(num_input_channels=512, num_output_channels=1024),
            Residual_Conv(num_input_channels=1024, num_output_channels=1024)
        )

        self.residual_block_7 = nn.Sequential(
            Residual_Conv(num_input_channels=1024, num_output_channels=512),
            Residual_Conv(num_input_channels=512, num_output_channels=256)
        )

        self.residual_block_8 = Residual_Conv(num_input_channels=512, num_output_channels=256)

        self.residual_block_9 = Residual_Conv(num_input_channels=256, num_output_channels=256)

        self.residual_block_10 = nn.Sequential(
            Residual_Conv(num_input_channels=256, num_output_channels=128),
            Residual_Conv(num_input_channels=128, num_output_channels=128)
        )

        self.residual_block_11 = Residual_Conv(num_input_channels=128, num_output_channels=64)

        self.residual_block_12 = Residual_Conv(num_input_channels=64, num_output_channels=64)


        #-------------Multiscale_Conv_Block blocks------------
        self.multiscale_block_1 = Multiscale_Conv_Block(num_input_channels=128, num_output_channels=32,
            kernelSizes=[3,3,5,5], dilatationRates=[1,3,3,6]
        )

        self.multiscale_block_2 = Multiscale_Conv_Block(num_input_channels=256, num_output_channels=64,
            kernelSizes=[3,3,5,5], dilatationRates=[1,3,2,3]
        )

        self.multiscale_block_3 = Multiscale_Conv_Block(num_input_channels=64, num_output_channels=16,
            kernelSizes=[3,3,5,7], dilatationRates=[1,3,2,6]
        )
            
        #-------------MaxPool2d blocks------------
        self.pool_block_1 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool_block_2 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool_block_3 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool_block_4 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool_block_5 = nn.MaxPool2d(kernel_size=(2,2))

        #-------------ConvTranspose2d blocks------------
        self.conv_transpose_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
            kernel_size=2, stride=(2,2),
        )

        self.conv_transpose_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
            kernel_size=2, stride=(2,2),
        )

        self.conv_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
            kernel_size=2, stride=(2,2),
        )

        self.conv_transpose_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
            kernel_size=2, stride=(2,2),
        )

        self.conv_transpose_5 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
            kernel_size=2, stride=(2,2),
        )

    def forward(self, imgs: torch.Tensor):
        """Logic for using layers defined in init.

        This method defines how layers are used in forward operation.

        Args:
            imgs (torch.Tensor): Input images, the tensor is of the shape NCHW.

        Returns:
            output (torch.Tensor): The inference output. 

        """
        conv1 = self.conv_block_1(imgs)    
        pool1 = self.pool_block_1(conv1)     

        conv2 = self.residual_block_1(pool1) 
        pool2 = self.pool_block_2(conv2)    

        conv3 = self.residual_block_2(pool2)
        conv3 = self.multiscale_block_1(conv3)  
        conv3 = self.residual_block_3(conv3)    
        pool3 = self.pool_block_3(conv3)    

        conv4 = self.residual_block_4(pool3)    
        pool4 = self.pool_block_4(conv4)    

        conv5 = self.residual_block_5(pool4) 
        pool5 = self.pool_block_5(conv5)    

        conv51 = self.residual_block_6(pool5) 

        up61 = torch.cat([self.conv_transpose_1(conv51),conv5], dim=1)  
        conv61 = self.residual_block_7(up61)    
        
        up6 = torch.cat([self.conv_transpose_2(conv61), conv4], dim=1)  
        conv6 = self.residual_block_8(up6) 
        conv6 = self.multiscale_block_2(conv6)  
        conv6 = self.residual_block_9(conv6)    

        up7 = torch.cat([self.conv_transpose_3(conv6), conv3], dim=1)   
        conv7 = self.residual_block_10(up7)     

        up8 = torch.cat([self.conv_transpose_4(conv7), conv2], dim=1)   
        conv8 = self.residual_block_11(up8)     
        conv8 = self.multiscale_block_3(conv8)  
        conv8 = self.residual_block_12(conv8)   

        up9 = torch.cat([self.conv_transpose_5(conv8), conv1], dim=1)   
        conv9 = self.conv_block_2(up9)  

        conv10 = self.conv_block_3(conv9)   
        
        return conv10


    @staticmethod
    def generate_inst_dict(pred_mask, bounding_boxes):
        """To collect instance information and store it within a dictionary.

        Args:
            pred_mask: A list of (binary) prediction masks, shape(no.patch, h, w)
            bounding_boxes: ndarray, A list of bounding boxes. 
                bounding box: `[start_x, start_y, end_x, end_y]`.

        Returns:
            inst_info_dict (dict): A dictionary containing a mapping of each instance
                    within `pred_mask` instance information. It has following form

                    inst_info = {
                            box: number[],
                            centroids: number[],
                            contour: number[][],
                    }
                    inst_info_dict = {[inst_uid: number] : inst_info}

                    and `inst_uid` is an integer corresponds to the instance
                    having the same pixel value within `pred_inst`.

        """
        inst_info_dict = {}
        for i in range(len(pred_mask)):
            patch = pred_mask[i]
            patch = patch.astype(np.uint8)
            inst_moment = cv2.moments(patch)
            inst_contour = cv2.findContours(
                patch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            print(inst_contour)
            inst_contour = inst_contour[0][0].astype(np.int32)
            inst_contour = np.squeeze(inst_contour)

            # < 3 points does not make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small
            if inst_contour.shape[0] < 3:  # pragma: no cover
                continue
            # ! check for trickery shape
            if len(inst_contour.shape) != 2:  # pragma: no cover
                continue

            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_box = bounding_boxes[i]
            inst_box_tl = inst_box[:2]
            inst_contour += inst_box_tl[None]
            inst_centroid += inst_box_tl  # X
            inst_info_dict[i+1] = {  # inst_id should start at 1
                "box": inst_box,
                "centroid": inst_centroid,
                "contour": inst_contour,
            }

        return inst_info_dict

            


    
    @staticmethod
    def postproc(preds, thresh=0.33, minSize=10, minHole=30, doReconstruction=False, nucPoints=None):
        """Post processing.

        Args:
            preds (ndarray): list of prediction output of each patch and
                assumed to be in the order of (no.patch, h, w) (match with the output
                of `infer_batch`).
            thresh (float): Threshold value. If a pixel has a predicted value larger than the threshold, it will be classified as nuclei.
            minSize (int): The smallest allowable object size.
            minHole (int):  The maximum area, in pixels, of a contiguous hole that will be filled.
            doReconstruction (bool): Whether to perform a morphological reconstruction of an image.
            nucPoints (ndarray): In the order of (no.patch, h, w). 
                In each patch, The pixel that has been 'clicked' is set to 1 and the rest pixels are set to 0.

        Returns:
            masks (ndarray): pixel-wise nuclei instance segmentation
                prediction, shape:(no.patch, h, w).
        """
        masks = preds > thresh
        
        masks = remove_small_objects(masks, min_size=minSize)
        masks = remove_small_holes(masks, area_threshold=minHole)
        if doReconstruction:
            for i in range(len(masks)):
                thisMask = masks[i, :, :]
                thisMarker = nucPoints[i, :, :] > 0
                
                try:
                    thisMask = reconstruction(thisMarker, thisMask, selem=disk(1))
                    masks[i] = np.array([thisMask])
                except Exception as e:
                    print(e)
                    warnings.warn('Nuclei reconstruction error #' + str(i))
        return masks   


    @staticmethod
    def infer_batch(model, batch_data, on_gpu):
        """Run inference on an input batch.

        This contains logic for forward operation as well as batch i/o
        aggregation.

        Args:
            model (nn.Module): PyTorch defined model.
            batch_data (ndarray): a batch of data generated by
                torch.utils.data.DataLoader.
            on_gpu (bool): Whether to run inference on a GPU.

        Returns:
            Pixel-wise nuclei prediction for each patch, shape: (no.patch, h, w).

        """
        model.eval()
        device = misc.select_device(on_gpu)

        #Assume batch_data is NCHW
        batch_data= batch_data.to(device).type(torch.float32)

        with torch.inference_mode():
            output = model(batch_data)
            output = torch.sigmoid(output)
            output = torch.squeeze(output, 1)

        return output.cpu().numpy()