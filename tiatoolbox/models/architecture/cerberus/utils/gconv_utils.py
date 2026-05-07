import math

import numpy as np
import torch


def get_basis_info(ksize):
    """Get the filter info for a given kernel size.

    Args:
        ksize (int): input kernel size

    Returns:
        freq_list: list of frequencies
        radius_list: list of radius values
        bandlimit_list: used to bandlimit high frequency filters in get_basis_filters()

    """
    if ksize == 5:
        freq_list = [0, 1, 2]
        radius_list = [0, 1, 2]
        bandlimit_list = [0, 2, 2]
    elif ksize == 7:
        freq_list = [0, 1, 2, 3]
        radius_list = [0, 1, 2, 3]
        bandlimit_list = [0, 2, 3, 2]
    elif ksize == 9:
        freq_list = [0, 1, 2, 3, 4]
        radius_list = [0, 1, 2, 3, 4]
        bandlimit_list = [0, 3, 4, 4, 3]

    return freq_list, radius_list, bandlimit_list


def get_basis_filters(freq_list, radius_list, bandlimit_list, ksize, eps=1e-8):
    """Gets the atomic basis filters.

    Args:
        freq_list: list of frequencies for basis filters
        radius_list: list of radius values for the basis filters
        bandlimt_list: bandlimit list to reduce aliasing of basis filters
        ksize (int): kernel size of basis filters
        eps=1e-8: epsilon used to prevent division by 0

    Returns:
        filter_list_bl: list of filters, with bandlimiting (bl) to reduce aliasing
        freq_list_bl: corresponding list of frequencies used in bandlimited filters
        radius_list_bl: corresponding list of radius values used in bandlimited filters

    """
    filter_list = []
    used_frequencies = []
    for radius in radius_list:
        for freq in freq_list:
            if freq <= bandlimit_list[radius]:
                his = ksize // 2  # half image size
                y_index, x_index = np.mgrid[-his : (his + 1), -his : (his + 1)]
                y_index *= -1
                z_index = x_index + 1j * y_index

                # convert z to natural coordinates and add epsilon to avoid division by zero
                z = z_index + eps
                r = np.abs(z)

                if radius == radius_list[-1]:
                    sigma = 0.4
                else:
                    sigma = 0.6

                rad_prof = np.exp(-((r - radius) ** 2) / (2 * (sigma**2)))
                c_image = rad_prof * (z / r) ** freq
                c_image_norm = (math.sqrt(2) * c_image) / np.linalg.norm(c_image)

                # add basis filter to list
                filter_list.append(c_image_norm)
                # add corresponding frequency of filter to list (info needed for phase manipulation)
                used_frequencies.append(freq)

    filter_array = np.array(filter_list)

    filter_array = np.reshape(
        filter_array,
        [filter_array.shape[0], filter_array.shape[1], filter_array.shape[2]],
    )

    return filter_array, used_frequencies


def get_rot_info(nr_orients, freq_list):
    """Generate rotation info for phase manipulation of steerable filters.
    Rotation is dependent on the frequency of the filter.

    Args:
        nr_orients: number of filter rotations
        freq_list: list of frequencies

    Returns:
        rot_info used to rotate steerable filters

    """
    # Generate rotation matrix for phase manipulation of steerable function
    rot_list = []
    for i in range(len(freq_list)):
        list_tmp = []
        for j in range(nr_orients):
            # Rotation is dependent on the frequency of the basis filter
            angle = (2 * np.math.pi / nr_orients) * j
            list_tmp.append(np.exp(-1j * freq_list[i] * angle))
        rot_list.append(list_tmp)
    rot_info = np.array(rot_list)

    # Reshape to enable matrix multiplication
    rot_info = np.reshape(rot_info, [rot_info.shape[0], 1, nr_orients])
    return rot_info


def get_rotated_basis_filters(ksize, nr_orients):
    """Generate basis filters rotated by angles of 2*pi / nr_orients.

    Args:
        ksize_list: list of kernel sizes used in the model
        nr_orients: number of orientations of the filters

    Returns:
        list of rotated basis filters - each element of the list is a Tensor of rotated
        basis filters for a particular kernel size

    """
    freq_list, radius_list, bandlimit_list = get_basis_info(ksize)
    basis_filters, used_frequencies = get_basis_filters(
        freq_list, radius_list, bandlimit_list, ksize
    )
    rot_info = get_rot_info(nr_orients, used_frequencies)

    rot_info = np.expand_dims(np.transpose(rot_info, [2, 0, 1]), -1)
    basis_filters = np.repeat(np.expand_dims(basis_filters, 0), nr_orients, axis=0)
    rotated_basis_filters = rot_info * basis_filters

    # separate real and imaginary parts -> pytorch doesn't have complex number functionality
    rotated_basis_filters_real = np.expand_dims(rotated_basis_filters.real, -1)
    rotated_basis_filters_imag = np.expand_dims(rotated_basis_filters.imag, -1)
    rotated_basis_filters = np.stack(
        [rotated_basis_filters_real, rotated_basis_filters_imag]
    )
    rotated_basis_filters = rotated_basis_filters.astype(np.float32)
    rotated_basis_filters = torch.tensor(rotated_basis_filters, requires_grad=False)
    return rotated_basis_filters


def cycle_channels(filters, shape_list):
    """Perform cyclic permutation of the orientation channels for kernels on the group G.

    Args:
        filters: input filters
        shape_list:  [nr_orients_out, ksize, ksize,
                      nr_orients_in, in_ch, out_ch]

    Returns:
        tensor of filters with channels permuted

    """
    nr_orients_out = shape_list[0]
    rotated_filters = [None] * nr_orients_out
    # TODO Parallel processing - add decorator or vectorise?
    for orientation in range(nr_orients_out):
        # [K, K, nr_orients_in, in_ch, out_ch]
        filters_tmp = filters[orientation]
        # [K, K, in_ch, out_ch, nr_orients]
        filters_tmp = filters_tmp.permute(0, 1, 3, 4, 2)
        # [K * K * in_ch * out_ch, nr_orients_in]
        filters_tmp = filters_tmp.reshape(
            shape_list[1] * shape_list[2] * shape_list[4] * shape_list[5], shape_list[3]
        )
        # Cycle along the orientation axis
        roll_matrix = (
            torch.Tensor(torch.roll(torch.eye(shape_list[3]), orientation, dims=1))
            .to("cuda")
            .type(torch.float32)
        )
        filters_tmp = torch.mm(filters_tmp, roll_matrix)
        filters_tmp = filters_tmp.view(
            shape_list[1], shape_list[2], shape_list[4], shape_list[5], shape_list[3]
        )
        filters_tmp = filters_tmp.permute(0, 1, 4, 2, 3)
        rotated_filters[orientation] = filters_tmp

    return torch.stack(rotated_filters)


def get_rotated_filters(weight, nr_orients_out, rotated_basis_filters, cycle_filter):
    """Generate the rotated filters either by phase manipulation or direct rotation
    of planar filter. Cyclic permutation of channels is performed for kernels on the group G.

    Args:
        weight: coefficients used to perform a linear combination of basis filters
        domain: domain of the operation - either `Z2` or `G`
        nr_orients_out: number of output filter orientations
        rotated_basis_filters: rotated atomic basis filters

    Returns:
        rot_filters: rotated steerable filters, with
                     cyclic permutation if not the first layer

    """
    # Linear combination of basis filters, taking only the real part
    rotated_basis_filters = rotated_basis_filters.unsqueeze(-1).unsqueeze(-1)
    combined_basis_filters = (
        weight[0] * rotated_basis_filters[0] - weight[1] * rotated_basis_filters[1]
    )
    # [nr_orients_out, K, K, nr_orients_in, in_ch, out_ch]
    rotated_steerable_filters = torch.sum(combined_basis_filters, dim=1)
    # Do not cycle filter for input convolution f: Z2 -> G
    if cycle_filter:
        shape_list = rotated_steerable_filters.size()
        # cycle channels - [nr_orients_out, K, K, nr_orients_in, in_ch, out_ch]
        rotated_steerable_filters = cycle_channels(
            rotated_steerable_filters, shape_list
        )

    return rotated_steerable_filters


def group_concat(x, y, nr_orients):
    """Concatenate G-feature maps by not concatenating along
    orientation axis.

    Args:
        x: feature map 1
        y: feature map 2
        nr_orients: number of orientations considered in the G-feature map

    """
    shape1 = x.size()
    chans1 = shape1[1]
    c1 = int(chans1 / nr_orients)
    x = x.reshape(-1, nr_orients, c1, shape1[2], shape1[3])

    shape2 = y.size()
    chans2 = shape2[1]
    c2 = int(chans2 / nr_orients)
    y = y.reshape(-1, nr_orients, c2, shape2[2], shape2[3])

    z = torch.cat((x, y), dim=2)

    return z.reshape(-1, nr_orients * (c1 + c2), shape1[2], shape1[3])
