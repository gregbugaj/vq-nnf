import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sahi.slicing import slice_image
from sewar import rmse
from skimage import metrics


def similarity_scoreXX(template, query, metric) -> float:
    """
    Compute similarity score between template and query image
    """

    p1 = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    p2 = cv2.cvtColor(query, cv2.COLOR_RGB2GRAY)

    s0 = rmse(p1, p2)  # value between 0 and 255
    s0 = 1 - s0 / 255  # normalize rmse to 1

    s1 = metrics.structural_similarity(
        p1, p2, full=True, data_range=1
    )[0]

    print(f"RMSE: {s0}")
    print(f"SSIM: {s1}")
    score = (s0 + s1) / 2
    return score


idx = 0


def slicer(image, patch_size_h, patch_size_w) -> list:
    slice_image_result = slice_image(
        image=image,
        output_file_name="patch_",  # ADDED OUTPUT FILE NAME TO (OPTIONALLY) SAVE SLICES
        output_dir="/tmp/dim/patches",  # ADDED INTERIM DIRECTORY TO (OPTIONALLY) SAVE SLICES
        slice_height=patch_size_h,
        slice_width=patch_size_w,
        overlap_height_ratio=0,
        overlap_width_ratio=0,
        auto_slice_resolution=True,
    )

    image_list = []
    for idx, slice_result in enumerate(slice_image_result):
        patch = slice_result["image"]
        starting_pixel = slice_result["starting_pixel"]
        image_list.append(patch)
        # shift_amount_list.append(starting_pixel)
    return image_list


def similarity_score(template, query, metric) -> float:
    image1_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    image1_gray = crop_to_content(image1_gray, content_aware=True)
    image2_gray = crop_to_content(image2_gray, content_aware=True)

    # find the max dimension of the two images and resize the other image to match it
    # we add 16 pixels to the max dimension to ensure that the image does not touch the border
    max_y = int(max(image1_gray.shape[0], image2_gray.shape[0])) + 16
    max_x = int(max(image1_gray.shape[1], image2_gray.shape[1])) + 16

    image1_gray, coord = resize_image(
        image1_gray,
        desired_size=(max_y, max_x),
        color=(255, 255, 255),
        keep_max_size=False,
    )

    image2_gray, coord = resize_image(
        image2_gray,
        desired_size=(max_y, max_x),
        color=(255, 255, 255),
        keep_max_size=False,
    )

    # ensure that the shapes are the same for
    if image1_gray.shape != image2_gray.shape:
        raise ValueError(
            f"Template and prediction snippet have different shapes: {image1_gray.shape} vs {image2_gray.shape}"
        )

    global idx
    idx += 1

    # save for debugging
    if True:
        stacked = np.hstack((image1_gray, image2_gray))
        cv2.imwrite(f"/tmp/dim/stacked_{idx}.png", stacked)

    patch_size_h = image1_gray.shape[0] // 4
    patch_size_w = image1_gray.shape[1] // 6
    # ensure that the patch size is at least 7x7
    if patch_size_h < 7:
        patch_size_h = 7
    if patch_size_w < 7:
        patch_size_w = 7

    # ensure that the patch is not larger than the image
    patch_size_h = min(patch_size_h, image1_gray.shape[0])
    patch_size_w = min(patch_size_w, image1_gray.shape[1])

    patches_t = slicer(image1_gray, patch_size_h, patch_size_w)
    patches_m = slicer(image2_gray, patch_size_h, patch_size_w)


    ix = 1
    s0_total = 0
    s1_total = 0
    slices_len = len(patches_t)
    for p1, p2 in zip(patches_t, patches_m):
        s0 = rmse(p1, p2)  # value between 0 and 255
        s0 = 1 - s0 / 255  # normalize rmse to 1

        s1 = metrics.structural_similarity(
            p1, p2, full=True, data_range=1
        )[0]

        s0_total += max(s0, 0)
        s1_total += max(s1, 0)
        ix += 1

    s0_total_norm = s0_total / slices_len
    s1_total_norm = s1_total / slices_len
    score = (s0_total_norm + s1_total_norm) / 2

    if False:
        print("score", score)
        print("s0_total", s0_total)
        print("s1_total", s1_total)
        print("s0_total_norm", s0_total_norm)
        print("s1_total_norm", s1_total_norm)

    return score


def crop_to_content(frame: np.ndarray, content_aware=True) -> np.ndarray:
    """
    Crop given image to content
    No content is defined as first non background(white) pixel.

    @param frame: the image frame to process
    @param content_aware: if enabled we will apply more aggressive crop method
    @return: new cropped frame
    """

    start = time.time()
    # conversion required, or we will get 'Failure to use adaptiveThreshold: CV_8UC1 in function adaptiveThreshold'
    # frame = np.random.choice([0, 255], size=(32, 32), p=[0.01, 0.99]).astype("uint8")
    cv2.imwrite("/tmp/fragments/frame-src.png", frame)

    # Transform source image to gray if it is not already
    # check if the image is already in grayscale
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    if content_aware:
        # apply division normalization to preprocess the image
        blur = cv2.GaussianBlur(gray, (5, 5), sigmaX=0, sigmaY=0)
        # divide
        divide = cv2.divide(gray, blur, scale=255)
        thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        #
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        op_frame = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    else:
        op_frame = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    indices = np.array(np.where(op_frame == [0]))
    img_w = op_frame.shape[1]
    img_h = op_frame.shape[0]
    min_x_pad = 16  # img_w // 16
    min_y_pad = img_h // 4

    if len(indices[0]) == 0 or len(indices[1]) == 0:
        print("No content found")
        return frame

    # indices are in y,X format
    if content_aware:
        x = max(0, indices[1].min() - min_x_pad)
        y = 0  # indices[0].min()
        h = img_h  # indices[0].max() - y
        w = min(img_w, indices[1].max() - x + min_x_pad)
    else:
        x = indices[1].min()
        y = indices[0].min()
        h = indices[0].max() - y
        w = indices[1].max() - x

    cropped = frame[y: y + h + 1, x: x + w + 1].copy()
    # cv2.imwrite("/tmp/fragments/cropped.png", cropped)

    dt = time.time() - start
    return cropped


def resize_image(
        image, desired_size, color=(255, 255, 255), keep_max_size=False
) -> tuple:
    """Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------

    image: np.array
        The image to be resized.

    desired_size: (int, int)
        The (height, width) of the resized image

    Return
    ------

    image: np.array
        The image of size = desired_size
    """

    # check if the image is already of the desired size
    if image.shape[0] == desired_size[0] and image.shape[1] == desired_size[1]:
        return image, (0, 0, image.shape[1], image.shape[0])

    size = image.shape[:2]
    if keep_max_size:
        # size = (max(size[0], desired_size[0]), max(size[1], desired_size[1]))
        h = size[0]
        w = size[1]
        dh = desired_size[0]
        dw = desired_size[1]

        if w > dw and h < dh:
            delta_h = max(0, desired_size[0] - size[0])
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left = 40
            right = 40
            image = cv2.copyMakeBorder(
                image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
            )
            size = image.shape[:2]
            # cv2.imwrite("/tmp/marie/box_framed_keep_max_size.png", image)
            return image, (left, top, size[1], size[0])

    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(
            image, (new_size[1], new_size[0]), interpolation=cv2.INTER_CUBIC
        )
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    # convert top, bottom, left, right to x, y, w, h
    x, y, w, h = left, top, size[1], size[0]

    # cv2.imwrite("/tmp/dim/box_framed.png", image)
    return image, (x, y, w, h)


def viz_patches(patches, filename):
    plt.figure(figsize=(9, 9))
    square_x = patches.shape[1]
    square_y = patches.shape[0]

    ix = 1
    for i in range(square_y):
        for j in range(square_x):
            # specify subplot and turn of axis
            ax = plt.subplot(square_y, square_x, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot
            plt.imshow(patches[i, j, :, :], cmap="gray")
            ix += 1
    # show the figure
    # plt.show()
    plt.savefig(filename)
    plt.close()
