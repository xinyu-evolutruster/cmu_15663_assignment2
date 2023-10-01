import skimage
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

from cp_hw2 import writeHDR, read_colorchecker_gm, lRGB2XYZ, XYZ2lRGB


def get_mask(z, z_min, z_max):
    mask = (z >= z_min)
    mask2 = (z <= z_max)
    mask = (mask & mask2)
    return mask


def w_uniform(z, z_min=0.05, z_max=0.95):
    if isinstance(z, np.ndarray):
        mask = get_mask(z, z_min, z_max)
        w = np.zeros_like(z)
        w[mask] = 1.0
        return w
    else:
        if z_min <= z <= z_max:
            return 1
        else:
            return 0


def w_tent(z, z_min=0.05, z_max=0.95):
    if isinstance(z, np.ndarray):
        mask = get_mask(z, z_min, z_max)
        w = np.minimum(z, 1 - z)
        w[~mask] = 0.0
        return w
    else:
        if z_min <= z <= z_max:
            return min(z, 1 - z)
        else:
            return 0


def w_gaussian(z, z_min=0.05, z_max=0.95):
    if isinstance(z, np.ndarray):
        mask = get_mask(z, z_min, z_max)
        w = np.exp(-4 * (z - 0.5)**2 / 0.5**2)
        w[~mask] = 0.0
        return w
    else:
        if z_min <= z <= z_max:
            return np.exp(-4 * (z - 0.5)**2 / 0.5**2)
        else:
            return 0


def w_photon(z, tk, z_min=0.05, z_max=0.95):
    if isinstance(z, np.ndarray):
        mask = get_mask(z, z_min, z_max)
        w = np.ones_like(z) * tk
        w[~mask] = 0.0
        return w
    else:
        if z_min <= z <= z_max:
            return tk
        else:
            return 0


def w_test(z, z_min=0.05, z_max=0.95):
    if isinstance(z, np.ndarray):
        return np.ones_like(z, dtype=np.float64) * 0.01
    else:
        return 0.01


def w_optimal(z, tk, z_min=0.05, z_max=0.95, g=39.9357, var=16287.0401):
    if isinstance(z, np.ndarray):
        mask = get_mask(z, z_min, z_max)
        num = np.ones_like(z) * (tk**2)
        den = z * g + var
        w = num / den
        w[~mask] = 0.0
        return w
    else:
        if z_min <= z <= z_max:
            return (tk**2) / (z * g + var)
        else:
            return 0


def calculate_tk(num_images=16):
    tks = []
    for k in range(1, num_images + 1):
        tks.append(1 / 2048 * np.power(2, k - 1))

    return tks


def read_images(data_dir, img_type='tiff'):
    image_paths = ['../data/door_stack/exposure' +
                   str(i) + '.' + img_type for i in range(1, 17)]

    print(image_paths)

    images = []
    for path in image_paths:
        img = skimage.io.imread(path)
        images.append(img)

    images = np.array(images)
    return images


def read_self_images(data_dir, img_type='tiff'):
    image_paths = ['../data/king_ohger/kingohger' +
                   str(i) + '.' + img_type for i in range(1, 12)]

    print(image_paths)

    images = []
    for path in image_paths:
        img = skimage.io.imread(path)
        images.append(img)

    images = np.array(images)
    return images


def compress_images(imgs, compress_rate=200):
    # imgs: [num_images, height, width, channel]
    return imgs[:, ::compress_rate, ::compress_rate, :]


def calculate_g(images, w_scheme=w_uniform, use_w_photon=False, z_min=0.05, z_max=0.95):
    compressed_imgs = compress_images(images, 200)

    num_images = compressed_imgs.shape[0]
    images = compressed_imgs.reshape(num_images, -1).swapaxes(0, 1)

    num_g = 256  # [0-255]
    num_pixels = images.shape[0]

    # The row of A represents a single pixel in an image
    # The last 256 rows are for the regularization term
    # The column of A corresponds to values g and L_ij
    A = np.zeros((num_images * num_pixels + num_g, num_g + num_pixels))
    b = np.zeros(num_images * num_pixels + num_g)

    # Calculate tk
    tks = calculate_tk(num_images)

    curr_row = 0
    for i in range(num_pixels):
        for j in range(num_images):
            pixel_val = images[i, j]
            weight = 0.0
            if use_w_photon:
                weight = w_scheme(
                    pixel_val / 255, tks[j], z_min, z_max)
            else:
                weight = w_scheme(pixel_val / 255, z_min, z_max)
            # set coefficient for g
            A[curr_row, pixel_val] = weight
            # set coefficient for Log(I_ij)
            A[curr_row, num_g + i] = -weight
            # set b
            b[curr_row] = weight * np.log(tks[j])
            curr_row += 1

    # Set regularization terms
    lamb = 100.0
    sqrt_lamb = np.sqrt(lamb)
    for z in range(0, 256):
        weight = sqrt_lamb
        if not use_w_photon:
            weight = w_scheme(z / 255, z_min, z_max) * sqrt_lamb
        A[curr_row, z] = -2 * weight
        A[curr_row, z - 1] = weight
        A[curr_row, z + 1] = weight
        curr_row += 1

    v = np.linalg.lstsq(A, b, rcond=None)
    g = v[0][:num_g]

    # x = np.arange(256)
    # plt.plot(x, g)
    # plt.show()
    # plt.savefig('g_plot.png')
    return g


def process_imgs(imgs, type='tiff', w_scheme=w_uniform, use_w_photon=False):
    if type == 'tiff':
        ldr_imgs = np.array([img / (2**16 - 1) for img in imgs])
        linear_imgs = ldr_imgs
    else:  # type == 'jpg'
        ldr_imgs = np.array([img / 255 for img in imgs])
        g = calculate_g(imgs, w_scheme=w_scheme, use_w_photon=use_w_photon)
        linear_imgs = np.array([np.exp(g[img]) for img in imgs])

    return ldr_imgs, linear_imgs


def optimal_process_imgs(imgs, dark_img, tks):
    ldr_imgs = []
    tnc = 1 / 4000
    for i in range(len(tks)):
        img = imgs[i] - dark_img * tks[i] / tnc
        img /= (2**16 - 1)
        ldr_imgs.append(img)

    return np.array(ldr_imgs)


def linear_merging(imgs, img_type='tiff', use_w_photon=False, w_scheme=w_uniform):
    ldr_imgs, linear_imgs = process_imgs(
        imgs, img_type, w_scheme, use_w_photon)

    tks = calculate_tk()

    # Calculate the denominator
    if use_w_photon:
        num_images = ldr_imgs.shape[0]
        print("num_images = ", num_images)
        hdr_denom = np.array([w_scheme(ldr_imgs[i], tks[i])
                             for i in range(num_images)]).sum(axis=0)
    else:
        hdr_denom = np.array([w_scheme(img) for img in ldr_imgs]).sum(axis=0)

    # Calculate the numerator
    hdr_num = np.zeros_like(imgs[0], dtype=np.float64)
    for k in range(len(imgs)):
        if use_w_photon:
            hdr_num += w_scheme(ldr_imgs[k], tks[k]) * linear_imgs[k] / tks[k]
        else:
            hdr_num += w_scheme(ldr_imgs[k]) * linear_imgs[k] / tks[k]

    hdr_img = (hdr_num / hdr_denom)

    # Set pixels where the denominator value is zero to the maximum valid pixel value
    mask = (hdr_denom == 0)
    hdr_img[mask] = 0.0

    return hdr_img


def logarithmic_merging(imgs, img_type='tiff', use_w_photon=False, w_scheme=w_uniform, eps=1e-5):
    ldr_imgs, linear_imgs = process_imgs(
        imgs, img_type, w_scheme, use_w_photon)

    tks = calculate_tk()

    # Calculate the denominator
    if use_w_photon:
        num_images = ldr_imgs.shape[0]
        hdr_denom = np.array([w_scheme(ldr_imgs[i], tks[i])
                             for i in range(num_images)]).sum(axis=0)
    else:
        hdr_denom = np.array([w_scheme(img) for img in ldr_imgs]).sum(axis=0)

    # Calculate the numerator
    hdr_num = np.zeros_like(imgs[0], dtype=np.float64)
    for k in range(len(imgs)):
        if use_w_photon:
            hdr_num += w_scheme(ldr_imgs[k], tks[k]) * \
                (np.log(linear_imgs[k] + eps) - np.log(tks[k]))
        else:
            hdr_num += w_scheme(ldr_imgs[k]) * \
                (np.log(linear_imgs[k] + eps) - np.log(tks[k]))

    hdr_img = np.exp(hdr_num / hdr_denom)

    # Set pixels where the denominator value is zero to the maximum valid pixel value
    mask = (hdr_denom == 0)
    hdr_img[mask] = 0.0

    return hdr_img


def optimal_weights_merging(imgs):
    tks = calculate_tk()
    dark_img = np.load('./dark_img.npy')

    ldr_imgs = optimal_process_imgs(imgs, dark_img, tks)

    # Calculate the denominator
    num_images = ldr_imgs.shape[0]
    hdr_denom = np.array([w_optimal(ldr_imgs[i], tks[i])
                          for i in range(num_images)]).sum(axis=0)

    # Calculate the numerator
    hdr_num = np.zeros_like(imgs[0], dtype=np.float64)
    for k in range(len(imgs)):
        hdr_num += w_optimal(ldr_imgs[k], tks[k]) * ldr_imgs[k] / tks[k]

    hdr_img = (hdr_num / hdr_denom)

    # Set pixels where the denominator value is zero to the maximum valid pixel value
    mask = (hdr_denom == 0)
    hdr_img[mask] = 0.0

    print(hdr_img[:10])

    return hdr_img


def create_hdr_images(data_dir="../data/door_stack/", img_type='tiff', use_w_photon=False, w_scheme=w_uniform):
    images = read_images(data_dir, img_type)
    # images = compress_images(images, 20)

    linear_hdr_img = linear_merging(
        images, img_type, use_w_photon, w_scheme)
    log_hdr_img = logarithmic_merging(
        images, img_type, use_w_photon, w_scheme)

    if img_type == 'tiff':
        writeHDR('linear_hdr_tiff.hdr', linear_hdr_img)
        writeHDR('log_hdr_tiff.hdr', log_hdr_img)
    else:  # img_type == jpg
        writeHDR('linear_hdr_jpg.hdr', linear_hdr_img)
        writeHDR('log_hdr_jpg.hdr', log_hdr_img)


def create_optimal_images(data_dir="../data/door_stack/"):
    images = read_self_images(data_dir, 'tiff')

    optimal_hdr_img = optimal_weights_merging(images)
    writeHDR('optimal_tiff.hdr', optimal_hdr_img)


def create_self_images(data_dir="../data/door_stack/", img_type='tiff', use_w_photon=False, w_scheme=w_uniform):
    images = read_self_images(data_dir, img_type)

    linear_hdr_img = linear_merging(
        images, img_type, use_w_photon, w_scheme)
    log_hdr_img = logarithmic_merging(
        images, img_type, use_w_photon, w_scheme)

    if img_type == 'tiff':
        writeHDR('self_linear_hdr_tiff.hdr', linear_hdr_img)
        writeHDR('self_log_hdr_tiff.hdr', log_hdr_img)
    else:  # img_type == jpg
        writeHDR('self_linear_hdr_jpg.hdr', linear_hdr_img)
        writeHDR('self_log_hdr_jpg.hdr', log_hdr_img)


def hand_pick_crops(img, save_file='avg_color.npy'):
    plt.imshow(img)
    input_coords = plt.ginput(n=24, timeout=0)

    np.save('input_coords.npy', input_coords)

    avg_colors = []
    for i in range(0, 24):
        x0, y0 = input_coords[i]
        x0, y0 = int(x0), int(y0)
        img_crop = img[y0-2:y0+2, x0-2:x0+2].reshape(-1, 3)
        avg_color = np.average(img_crop, axis=0)
        homo_color = np.append(avg_color, 1.0)
        avg_colors.append(homo_color)

    avg_colors = np.array(avg_colors)
    np.save(save_file, avg_colors)


def white_balancing(img, img_crop):
    r_mean, g_mean, b_mean = img_crop[:3]
    new_img = np.zeros(img.shape, dtype=np.float64)
    new_img[:, :, 0] = img[:, :, 0] * g_mean / r_mean
    new_img[:, :, 1] = img[:, :, 1]
    new_img[:, :, 2] = img[:, :, 2] * g_mean / b_mean

    return new_img


def color_correction(img, gt_colors, hdr_colors):

    print(img[0])

    A = np.zeros((3 * 24, 12), dtype=np.float64)
    b = np.zeros((3 * 24), dtype=np.float64)
    for i in range(24):
        row_start = i * 3
        for j in range(0, 3):
            b[row_start + j] = gt_colors[i, j]
            for k in range(0, 4):
                A[row_start + j, j * 4 + k] = hdr_colors[i, k]

    x = np.linalg.lstsq(A, b, rcond=None)
    affine = x[0].reshape(3, 4)

    h, w = img.shape[0], img.shape[1]
    homo = np.ones((h, w, 1), dtype=np.float64)
    img_homo = np.concatenate((img, homo), axis=2)

    # [h, w, 4] -> [h, w, 4, 1]
    img_homo = img_homo[..., np.newaxis]
    transformed_img = np.matmul(affine, img_homo)
    # clip negative value to zero
    transformed_img = np.clip(
        transformed_img, a_min=0.0, a_max=10000).squeeze(axis=3)

    print(transformed_img[0])

    return transformed_img


def color_correction_and_white_balancing(hdr_file):
    # load the hdr image
    hdr_img = cv2.imread(hdr_file, flags=cv2.IMREAD_ANYDEPTH)
    # BGR->RGB
    hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
    hdr_img = hdr_img.astype(np.float64)

    input_coords = np.load('input_coords.npy')
    # print(input_coords.shape)

    avg_colors = []
    for i in range(0, 24):
        x0, y0 = input_coords[i]
        x0, y0 = int(x0), int(y0)
        img_crop = hdr_img[y0-2:y0+2, x0-2:x0+2].reshape(-1, 3)
        avg_color = np.average(img_crop, axis=0)
        homo_color = np.append(avg_color, 1.0)
        avg_colors.append(homo_color)

    hdr_colors = np.array(avg_colors)

    # get the gt color coordinatess
    r, g, b = read_colorchecker_gm()
    gt_colors = np.stack([r, g, b], axis=2).reshape(-1, 3)

    corrected_img = color_correction(hdr_img, gt_colors, hdr_colors)

    # additional white balancing
    # use patch 4
    img_crop = hdr_colors[3]
    wb_img = white_balancing(corrected_img, img_crop)

    # show image for testing
    # writeHDR('corrected.hdr', wb_img)
    # plt.imshow(wb_img)
    # plt.show()


def tone_mapping_rgb_color_channels(hdr_img, eps=1e-5, K=0.15, B=0.95):
    h, w, _ = hdr_img.shape

    # Method 1: apply tone mapping to all color channels
    N = h * w  # number of pixels
    hdr_m = np.exp(1 / N * np.sum((np.log(hdr_img + eps)), axis=None))
    hdr_tilde_img = K / hdr_m * hdr_img
    img_white = B * np.max(hdr_tilde_img)
    tm_img = hdr_tilde_img * (1 + hdr_tilde_img /
                              (img_white**2)) / (1 + hdr_tilde_img)

    return tm_img


def tone_mapping_luminance_channel(hdr_img, eps=1e-8, K=0.15, B=0.95):
    h, w, _ = hdr_img.shape

    # Method 2: apply tone mapping only to the luminance channel Y
    # Convert the HDR image from RGB to XYZ
    xyz_img = lRGB2XYZ(hdr_img)
    # Convert XYZ to xyY
    # Ref: http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_xyY.html
    xyY_img = np.ones_like(xyz_img, dtype=np.float64)
    xyz_sum = np.sum(xyz_img, axis=2)
    xyY_img[:, :, 0] = xyz_img[:, :, 0] / xyz_sum
    xyY_img[:, :, 1] = xyz_img[:, :, 1] / xyz_sum
    xyY_img[:, :, 2] = xyz_img[:, :, 1]

    # Set pixels where X=Y=Z=0 to chromaticity white
    white_mask = (xyz_sum == 0)
    xyY_img[white_mask, 0] = 1.0
    xyY_img[white_mask, 1] = 1.0

    # Tonemap the luminance Y
    Y = xyY_img[:, :, 2]
    N = h * w
    hdr_m = np.exp(1 / N * np.sum(np.log(Y + eps), axis=None))
    hdr_tilde_img = K / hdr_m * Y
    img_white = B * np.max(hdr_tilde_img)
    tm_img = hdr_tilde_img * (1 + hdr_tilde_img /
                              (img_white**2)) / (1 + hdr_tilde_img)

    xyY_img[:, :, 2] = tm_img

    # xyY to XYZ
    # Ref: http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html
    x, y, Y = xyY_img[:, :, 0], xyY_img[:, :, 1], xyY_img[:, :, 2]
    xyz_img = np.zeros_like(xyY_img, dtype=np.float64)
    xyz_img[:, :, 0] = x * Y / y
    xyz_img[:, :, 1] = Y
    xyz_img[:, :, 2] = (1.0 - x - y) * Y / y

    # Invert the color transform to go back to RGB
    rgb_img = XYZ2lRGB(xyz_img)

    return rgb_img


def tone_mapping(hdr_img, eps=1e-5, K=0.15, B=0.95, mode='rgb'):
    if mode == 'rgb':
        return tone_mapping_rgb_color_channels(hdr_img, eps, K, B)
    else:  # mode == 'luminance'
        return tone_mapping_luminance_channel(hdr_img, eps, K, B)


def get_dark_img():
    data_dir = '../data/dark_frames'
    dark_img = None
    num_img = 0
    for f in os.listdir(data_dir):
        file_path = os.path.join(data_dir, f)
        print("load ", file_path)
        img = skimage.io.imread(file_path).astype(np.float64)
        if dark_img is None:
            dark_img = img
        else:
            dark_img += img
        num_img += 1

    dark_img /= num_img

    np.save('dark_img.npy', dark_img)


def plot_histograms(dark_img, imgs):
    # randomly pick 10 pixels
    h, w = dark_img.shape[:2]
    num_pixels = 10
    num_bins = 10
    xs = np.random.randint(low=0, high=h, size=num_pixels)
    ys = np.random.randint(low=0, high=w, size=num_pixels)

    for i in range(num_pixels):
        hist = []
        for img in imgs:
            hist.append(img[xs[i], ys[i]])
        hist = np.array(hist)
        # plt.hist(hist, bins=num_bins, rwidth=0.8)
        # plt.savefig('./hist_{}_x{}_y{}.png'.format(i, xs[i], ys[i]))
        # plt.cla()
        # plt.show()


def noise_calibration():
    data_dir = '../data/ramp_frames'
    dark_img = np.load('./dark_img.npy')

    # rgb to gray image
    dark_img = np.average(dark_img, axis=2)

    imgs = []
    cnt = 0
    for f in os.listdir(data_dir):
        file_path = os.path.join(data_dir, f)
        print("load ", file_path)
        img = skimage.io.imread(file_path).astype(np.float64)

        # rgb to gray image
        img = np.average(img, axis=2) - dark_img

        imgs.append(img)

    # plot_histograms(dark_img, imgs)

    # Compute the mean value for each pixel
    imgs = np.array(imgs)
    num_img = imgs.shape[0]
    mu = np.average(imgs, axis=0)
    # Round the mean to the nearest integer
    mu = np.rint(mu)

    # Compute the variance for each pixel
    total_diff = np.zeros_like(mu, dtype=np.float64)
    for i in range(num_img):
        diff = imgs[i] - mu
        diff2 = diff ** 2
        total_diff += diff2

    sigma = total_diff / (num_img - 1)

    avg_var = dict()
    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            if mu[i, j] not in avg_var:
                avg_var[mu[i, j]] = []
            avg_var[mu[i, j]].append(sigma[i, j])

    x = np.array(list(avg_var.keys()))
    y = np.array([sum(vars) / len(vars) for vars in avg_var.values()])
    plt.scatter(x, y, s=1)

    k, b = np.polyfit(x, y, deg=1)
    print("slope k = {}, intercept b = {}".format(k, b))

    xseq = np.linspace(0, x.max(), num=100)
    # plt.plot(xseq, k * xseq + b, color="red", lw=1.5)
    # plt.xlabel("mean")
    # plt.ylabel("variance")
    # plt.savefig('./slope.png')
    # plt.show()


def merge_optimal_weights():
    dark_img = np.load('./dark_img.npy')

    tks = calculate_tk()
    tnc = 1 / 4

    # load the RAW exposure stack and perform dark-frame subtraction
    data_dir = '../data/king_ohger'
    images = read_self_images(data_dir, img_type='tiff')

    num_img = images.shape[0]
    for i in range(num_img):
        images[i] = images[i] - dark_img * tks[i] / tnc

    hdr_img = linear_merging(images, w_scheme=w_optimal)


def gamma_encoding(img):
    mask = img <= 0.0031308
    new_img = np.zeros_like(img)
    new_img[mask] = 12.92 * img[mask]
    new_img[~mask] = (1 + 0.055) * np.power(img[~mask], 1 / 2.4) - 0.055
    return new_img


def main():
    # Section 1.2
    # data_dir = '../data/door_stack'
    # images = read_images(data_dir, img_type='jpg')

    # x = [i for i in range(256)]
    # g_uniform = calculate_g(images, w_scheme=w_uniform,
    #                         use_w_photon=False, z_min=0.05, z_max=0.95)
    # g_gaussian = calculate_g(images, w_scheme=w_gaussian,
    #                          use_w_photon=False, z_min=0.05, z_max=0.95)
    # g_tent = calculate_g(images, w_scheme=w_tent,
    #                      use_w_photon=False, z_min=0.05, z_max=0.95)
    # g_photon = calculate_g(images, w_scheme=w_photon,
    #                        use_w_photon=True, z_min=0.05, z_max=0.95)

    # Section 1.3
    # data_dir = '../data/door_stack'
    # create_hdr_images(data_dir, img_type='jpg',
    #                   use_w_photon=False, w_scheme=w_photon)
    # create_hdr_images(data_dir, img_type='tiff',
    #                   use_w_photon=False, w_scheme=w_uniform)

    # data_dir = '../data/king_ohger'
    # create_self_images(data_dir, img_type='jpg',
    #                    use_w_photon=False, w_scheme=w_uniform)
    # create_self_images(data_dir, img_type='tiff',
    #                    use_w_photon=False, w_scheme=w_uniform)

    # Section 2
    # hdr_file = '../result/hdr_images/w_uniform/linear_hdr_jpg.hdr'
    # # hdr_file = './rgb_gamma_encoded_img.png'
    # hdr_img = cv2.imread(hdr_file, flags=cv2.IMREAD_COLOR)
    # # BGR->RGB
    # hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
    # # hand_pick_crops(hdr_img)
    # color_correction_and_white_balancing(hdr_file=hdr_file)

    # Section 3
    # load the hdr image
    # hdr_file = '../result/hdr_images/w_uniform/linear_hdr_jpg.hdr'
    # hdr_file = './corrected.hdr'
    # hdr_img = cv2.imread(hdr_file, flags=cv2.IMREAD_ANYDEPTH)
    # # BGR->RGB
    # hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)

    # rgb_tm_img = tone_mapping(hdr_img, mode='rgb', K=0.05, B=1.0)
    # lum_tm_img = tone_mapping(hdr_img, mode='luminance', K=0.1, B=0.95)

    # gamma_encoded_img = gamma_encoding(rgb_tm_img)
    # lum_gamma_encoded_img = gamma_encoding(lum_tm_img)

    # Section 4: Create and tonemap your own HDR photo
    # data_dir = '../data/king_ohger/'
    # create_self_images(data_dir)

    # hdr_file = 'self_linear_hdr_tiff.hdr'
    # hdr_img = cv2.imread(hdr_file, flags=cv2.IMREAD_ANYDEPTH)
    # # BGR->RGB
    # hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)

    # rgb_tm_img = tone_mapping(hdr_img, mode='rgb', K=150.0, B=10.0)
    # lum_tm_img = tone_mapping(hdr_img, mode='luminance', K=0.8, B=0.95)

    # gamma_encoded_img = gamma_encoding(rgb_tm_img)
    # lum_gamma_encoded_img = gamma_encoding(lum_tm_img)

    # Section 5
    # noise_calibration()

    # Section 6
    # data_dir = '../data/king_ohger/'
    # create_optimal_images(data_dir)
    pass


if __name__ == '__main__':
    main()
