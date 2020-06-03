# Copyright University College London 2020
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# Author: Eric Einspänner
# For internal research only.


import os
import sys
import distutils.util
import re
import numpy as np
import scipy.optimize
import nibabel

import sirf.Reg as reg

import parser


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(string):
    return int(string) if string.isdigit() else string


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def human_sorting(string):
    return [atoi(c) for c in re.split(r'(\d+)', string)]


def warp_image_forward(resampler, static_image):
    return resampler.forward(static_image).as_array().astype(np.double)


def warp_image_adjoint(resampler, dynamic_image):
    return resampler.adjoint(dynamic_image).as_array().astype(np.double)


def gradient_function(optimise_array, static_image, dynamic_path, dvf_path, weighted_normalise, dynamic_data_magnitude):
    static_image.fill(np.reshape(optimise_array, static_image.as_array().astype(np.double).shape))

    gradient_value = static_image.clone()
    gradient_value.fill(0.0)

    adjoint_image = static_image.clone()

    for i in range(len(dynamic_path)):
        dynamic_image = reg.NiftiImageData(dynamic_path[i])
        dvf_image = reg.NiftiImageData3DDeformation(dvf_path[i])

        resampler = reg.NiftyResample()

        resampler.set_reference_image(static_image)
        resampler.set_floating_image(dynamic_image)
        resampler.add_transformation(dvf_image)

        resampler.set_interpolation_type_to_cubic_spline()

        adjoint_image.fill(((np.nansum(dynamic_image.as_array().astype(np.double), dtype=np.double) / dynamic_data_magnitude) * warp_image_forward(resampler, static_image)) - dynamic_image.as_array().astype(np.double))
        gradient_value.fill((gradient_value.as_array().astype(np.double) + (warp_image_adjoint(resampler, adjoint_image) * weighted_normalise[i])))

        # gradient_value.write("{0}/gradient.nii".format(output_path))

    print("Max gradient value: {0}, Mean gradient value: {1}, Gradient norm: {2}".format(
        str(np.amax(gradient_value.as_array().astype(np.double))),
        str(np.nanmean(np.abs(gradient_value.as_array().astype(np.double), dtype=np.double))),
        str(np.linalg.norm(gradient_value.as_array().astype(np.double)))))

    return np.ravel(gradient_value.as_array().astype(np.double)).astype(np.double)


def objective_function(optimise_array, static_image, dynamic_path, dvf_path, weighted_normalise, dynamic_data_magnitude):
    static_image.fill(np.reshape(optimise_array, static_image.as_array().astype(np.double).shape))

    objective_value = 0.0

    for i in range(len(dynamic_path)):
        dynamic_image = reg.NiftiImageData(dynamic_path[i])
        dvf_image = reg.NiftiImageData3DDeformation(dvf_path[i])

        resampler = reg.NiftyResample()

        resampler.set_reference_image(static_image)
        resampler.set_floating_image(dynamic_image)
        resampler.add_transformation(dvf_image)

        resampler.set_interpolation_type_to_cubic_spline()

        objective_value = objective_value + (np.nansum(np.square(dynamic_image.as_array().astype(np.double) - ((np.nansum(dynamic_image.as_array().astype(np.double), dtype=np.double) / dynamic_data_magnitude) * warp_image_forward(resampler, static_image)), dtype=np.double), dtype=np.double) * weighted_normalise[i])

    print("Objective function value: {0}".format(str(objective_value)))

    return objective_value


def suv_objective_function(multiple, optimise_array):
    objective_value = np.square(1.0 - np.nanmean(optimise_array.astype(np.double) * multiple[0]), dtype=np.double)

    print("Objective function value: {0}".format(str(objective_value)))

    return objective_value


def output_input(static_image, dynamic_path, dvf_path, output_path):
    static_image.write("{0}/static_image.nii".format(output_path))

    for i in range(len(dynamic_path)):
        dynamic_image = reg.NiftiImageData(dynamic_path[i])
        dvf_image = reg.NiftiImageData3DDeformation(dvf_path[i])

        dynamic_image.write("{0}/dynamic_image_{1}.nii".format(output_path, str(i)))
        dvf_image.write("{0}/dvf_image_{1}.nii".format(output_path, str(i)))

    return True


def test_for_adj(static_image, dvf_path, output_path):
    for i in range(len(dvf_path)):
        dvf_image = reg.NiftiImageData3DDeformation(dvf_path[i])

        resampler = reg.NiftyResample()

        resampler.set_reference_image(static_image)
        resampler.set_floating_image(static_image)
        resampler.add_transformation(dvf_image)

        resampler.set_interpolation_type_to_cubic_spline()

        warp = warp_image_forward(resampler, static_image)

        warped_image = static_image.clone()
        warped_image.fill(warp)

        warped_image.write("{0}/warp_forward_{1}.nii".format(output_path, str(i)))

        difference = static_image.as_array().astype(np.double) - warp

        difference_image = static_image.clone()
        difference_image.fill(difference)

        difference_image.write("{0}/warp_forward_difference_{1}.nii".format(output_path, str(i)))

        warp = warp_image_adjoint(resampler, static_image)

        warped_image = static_image.clone()
        warped_image.fill(warp)

        warped_image.write("{0}/warp_adjoint_{1}.nii".format(output_path, str(i)))

        difference = static_image.as_array().astype(np.double) - warp

        difference_image = static_image.clone()
        difference_image.fill(difference)

        difference_image.write("{0}/warp_adjoint_difference_{1}.nii".format(output_path, str(i)))

    return True


def get_resamplers(static_image, dynamic_array, dvf_array):
    resamplers = []

    for j in range(len(dynamic_array)):
        dynamic_image = dynamic_array[j]
        dvf_image = dvf_array[j]

        resampler = reg.NiftyResample()

        resampler.set_reference_image(static_image)
        resampler.set_floating_image(dynamic_image)
        resampler.add_transformation(dvf_image)

        resampler.set_interpolation_type_to_cubic_spline()

        resamplers.append(resampler)

    return resamplers


def edit_header(data, output_path):
    new_data_array = []

    for i in range(len(data)):
        current_data = nibabel.load(data[i])

        current_data_data = current_data.get_data()
        current_data_affine = current_data.affine
        current_data_header = current_data.header

        current_data_header["intent_code"] = 1007

        new_data = nibabel.Nifti1Image(current_data_data, current_data_affine, current_data_header)

        new_data_path = "{0}/new_dvf_{1}.nii".format(output_path, str(i))

        if os.path.exists(new_data_path):
            os.remove(new_data_path)

        nibabel.save(new_data, new_data_path)

        new_data_array.append(new_data_path)

    return new_data_array


def register_data(static_path, dynamic_path, output_path):
    path_new_displacement_fields = "{0}/new_displacement_fields/".format(output_path)

    if not os.path.exists(path_new_displacement_fields):
        os.makedirs(path_new_displacement_fields, mode=0o770)

    path_new_deformation_fields = "{0}/new_deformation_fields/".format(output_path)

    if not os.path.exists(path_new_deformation_fields):
        os.makedirs(path_new_deformation_fields, mode=0o770)

    path_new_tm = "{0}/new_tm/".format(output_path)

    if not os.path.exists(path_new_tm):
        os.makedirs(path_new_tm, mode=0o770)

    algo = reg.NiftyAladinSym()

    dvf_path = []

    for i in range(len(dynamic_path)):
        ref = reg.ImageData(dynamic_path[i])
        flo = reg.ImageData(static_path)

        algo.set_reference_image(ref)
        algo.set_floating_image(flo)

        algo.process()

        displacement_field = algo.get_displacement_field_forward()
        displacement_field.write("{0}/new_displacement_field_{1}.nii".format(path_new_displacement_fields, str(i)))

        dvf_path.append("{0}/new_DVF_field_{1}.nii".format(path_new_deformation_fields, str(i)))

        deformation_field = algo.get_deformation_field_forward()
        deformation_field.write(dvf_path[i])

        tm = algo.get_transformation_matrix_forward()
        tm.write("{0}/new_tm_{1}.nii".format(path_new_tm, str(i)))

    return dvf_path


def op_test(static_image, output_path):
    temp_at = reg.AffineTransformation()

    temp_at_array = temp_at.as_array().astype(np.double)
    temp_at_array[0][0] = 1.25
    temp_at_array[1][1] = 1.25
    temp_at_array[2][2] = 1.25
    temp_at_array[3][3] = 1.25

    temp_at = reg.AffineTransformation(temp_at_array)

    resampler = reg.NiftyResample()

    resampler.set_reference_image(static_image)
    resampler.set_floating_image(static_image)
    resampler.add_transformation(temp_at)

    resampler.set_interpolation_type_to_cubic_spline()

    warp = warp_image_forward(resampler, static_image)

    warped_image = static_image.clone()
    warped_image.fill(warp)

    warped_image.write("{0}/op_test_warp_forward.nii".format(output_path))

    difference = static_image.as_array().astype(np.double) - warp

    difference_image = static_image.clone()
    difference_image.fill(difference)

    difference_image.write("{0}/op_test_warp_forward_difference.nii".format(output_path))

    warp = warp_image_adjoint(resampler, static_image)

    warped_image = static_image.clone()
    warped_image.fill(warp)

    warped_image.write("{0}/op_test_warp_adjoint.nii".format(output_path))

    difference = static_image.as_array().astype(np.double) - warp

    difference_image = static_image.clone()
    difference_image.fill(difference)

    difference_image.write("{0}/warp_adjoint_difference.nii".format(output_path))

    return True


def get_dvf_path(input_dvf_path, dvf_split):
    all_dvf_path = os.listdir(input_dvf_path)
    dvf_path = []

    for i in range(len(all_dvf_path)):
        current_dvf_path = all_dvf_path[i].rstrip()

        if len(current_dvf_path.split(".nii")) > 1 and len(current_dvf_path.split(dvf_split)) > 1:
            dvf_path.append("{0}/{1}".format(input_dvf_path, current_dvf_path))

    dvf_path.sort(key=human_sorting)

    return dvf_path


def get_dynamic_data_magnitude(dynamic_path):
    dynamic_data_magnitude = 0.0

    for i in range(len(dynamic_path)):
        dynamic_data_magnitude = dynamic_data_magnitude + np.nansum(
            reg.NiftiImageData(dynamic_path[i]).as_array().astype(np.double), dtype=np.double)

    return dynamic_data_magnitude


def get_data_path(input_dynamic_path, dynamic_split):
    all_dynamic_path = os.listdir(input_dynamic_path)
    dynamic_path = []

    for i in range(len(all_dynamic_path)):
        current_dynamic_path = all_dynamic_path[i].rstrip()

        if len(current_dynamic_path.split(".nii")) > 1 and len(current_dynamic_path.split(dynamic_split)) > 1:
            dynamic_path.append("{0}/{1}".format(input_dynamic_path, current_dynamic_path))

    dynamic_path.sort(key=human_sorting)

    return dynamic_path


def back_warp(static_path, dvf_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o770)

    for i in range(len(dvf_path)):
        static_image = reg.NiftiImageData(static_path)
        dvf_image = reg.NiftiImageData3DDeformation(dvf_path[i])

        resampler = reg.NiftyResample()

        resampler.set_reference_image(static_image)
        resampler.set_floating_image(static_image)
        resampler.add_transformation(dvf_image)

        resampler.set_interpolation_type_to_cubic_spline()

        warped_static_image = warp_image_forward(resampler, static_image)

        static_image.fill(warped_static_image)

        static_image.write("{0}/back_warped_{1}.nii".format(output_path, str(i)))

        return True


def optimise(input_data_path, data_split, weighted_normalise_path, input_dvf_path, dvf_split, output_path, do_op_test,
             do_reg, do_test_for_adj, do_blind_start, do_opt, do_back_warp, prefix):
    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o770)

    new_dvf_path = "{0}/new_dvfs/".format(output_path)

    if not os.path.exists(new_dvf_path):
        os.makedirs(new_dvf_path, mode=0o770)

    # get static and dynamic paths
    dynamic_path = get_data_path(input_data_path, data_split)

    dynamic_data_magnitude = get_dynamic_data_magnitude(dynamic_path)

    static_path = "{0}/static_image.nii".format(output_path)

    # load static object for dvf registration
    static_image = reg.NiftiImageData(dynamic_path[0])
    static_image.write(static_path)

    if do_op_test:
        op_test(static_image, output_path)

    dvf_path = None

    if do_test_for_adj or do_opt or do_back_warp:
        # if do reg the calc dvf if not load
        if do_reg:
            dvf_path = register_data(static_path, dynamic_path, output_path)
        else:
            dvf_path = get_dvf_path(input_dvf_path, dvf_split)

        # fix dvf header and load dvf objects
        dvf_path = edit_header(dvf_path, new_dvf_path)

    # sum the dynamic data into the static data
    for i in range(1, len(dynamic_path)):
        static_image.fill(static_image.as_array().astype(np.double) + reg.NiftiImageData(dynamic_path[i]).as_array().astype(np.double))

    static_image.write(static_path)

    # test for adj
    if do_test_for_adj:
        test_for_adj(static_image, dvf_path, output_path)
        output_input(static_image, dynamic_path, dvf_path, output_path)

    # initial static image
    initial_static_image = static_image.clone()

    if do_blind_start:
        initial_static_image.fill(1.0)

    initial_static_image.write("{0}/initial_static_image_{1}.nii".format(output_path, prefix))

    # array to optimise
    optimise_array = initial_static_image.as_array().astype(np.double)

    # array bounds
    bounds = []

    for j in range(len(np.ravel(optimise_array))):
        bounds.append((0.01, 10.0))

    tol = 0.000000000009

    if do_opt:
        weighted_normalise = parser.parser(weighted_normalise_path, "weighted_normalise:=")

        if weighted_normalise is None:
            weighted_normalise = parser.parser(weighted_normalise_path, "normalise_array:=")

        for i in range(len(weighted_normalise)):
            weighted_normalise[i] = float(weighted_normalise[i])

        # optimise
        optimise_array = np.reshape(scipy.optimize.minimize(objective_function, np.ravel(optimise_array), args=(
            static_image, dynamic_path, dvf_path, weighted_normalise, dynamic_data_magnitude),
                                                            method="L-BFGS-B", jac=gradient_function, bounds=bounds,
                                                            tol=tol, options={"disp": True}).x, optimise_array.shape)

    # output
    static_image.fill(optimise_array)
    static_image.write("{0}/optimiser_output_{1}.nii".format(output_path, prefix))

    difference = static_image.as_array().astype(np.double) - initial_static_image.as_array().astype(np.double)

    difference_image = initial_static_image.clone()
    difference_image.fill(difference)

    static_image.write("{0}/optimiser_output_difference_{1}.nii".format(output_path, prefix))

    if do_back_warp:
        back_warp(static_path, dvf_path, "{0}/back_warp/".format(output_path))

    multiple = 1.0

    nan_optimise_array = optimise_array
    nan_optimise_array[nan_optimise_array < 0.01] = np.nan
    nan_optimise_array = nan_optimise_array - np.nanmin(nan_optimise_array)

    # array bounds
    bounds = [(0.01, 10.0)]

    # optimise
    multiple = scipy.optimize.minimize(suv_objective_function, np.asarray(multiple), args=(nan_optimise_array),
                                       method="L-BFGS-B", tol=tol, bounds=bounds, options={"disp": True}).x[0]

    # output
    nan_optimise_array = nan_optimise_array - np.nanmin(nan_optimise_array)
    nan_optimise_array = np.nan_to_num(nan_optimise_array)
    nan_optimise_array[nan_optimise_array < 0.01] = 0.0
    nan_optimise_array = nan_optimise_array * multiple

    static_image.fill(nan_optimise_array)
    static_image.write("{0}/suv_optimiser_output_{1}.nii".format(output_path, prefix))

    naive_suv_optimise_array = optimise_array / 0.25
    static_image.fill(naive_suv_optimise_array)
    static_image.write("{0}/naive_suv_optimiser_output_{1}.nii".format(output_path, prefix))


def main():
    # file paths to data
    input_data_path = parser.parser(sys.argv[1], "data_path:=")
    data_split = parser.parser(sys.argv[1], "data_split:=")
    weighted_normalise_path = parser.parser(sys.argv[1], "weighted_normalise_path:=")
    input_dvf_path = parser.parser(sys.argv[1], "dvf_path:=")
    dvf_split = parser.parser(sys.argv[1], "dvf_split:=")
    output_path = parser.parser(sys.argv[1], "output_path:=")
    do_op_test = parser.parser(sys.argv[1], "do_op_test:=")
    do_reg = parser.parser(sys.argv[1], "do_reg:=")
    do_test_for_adj = parser.parser(sys.argv[1], "do_test_for_adj:=")
    do_blind_start = parser.parser(sys.argv[1], "do_blind_start:=")
    do_opt = parser.parser(sys.argv[1], "do_opt:=")
    do_back_warp = parser.parser(sys.argv[1], "do_back_warp:=")

    for i in range(len(input_data_path)):
        optimise(input_data_path[i], data_split[i], weighted_normalise_path[i], input_dvf_path[i], dvf_split[i],
                 output_path[i], bool(distutils.util.strtobool(do_op_test[i])),
                 bool(distutils.util.strtobool(do_reg[i])), bool(distutils.util.strtobool(do_test_for_adj[i])),
                 bool(distutils.util.strtobool(do_blind_start[i])), bool(distutils.util.strtobool(do_opt[i])),
                 bool(distutils.util.strtobool(do_back_warp[i])), str(i))


main()
