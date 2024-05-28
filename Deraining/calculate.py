import cv2
import numpy as np
import glob
from concurrent.futures import ProcessPoolExecutor
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * log10(255.0 / sqrt(mse))

def ssim(img1, img2):
    C1 = (0.02 * 255)**2
    C2 = (0.05 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # Valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def compute_ssim(img1, img2):
    if img1.ndim == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(img1, img2)



def process_images(image_path, gt_path):
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if input_image is None or gt_image is None:
        print(f"Error loading images. Check paths:\nInput: {image_path}\nGT: {gt_path}")
        return None, None  # 返回 None，以示错误

    psnr_val = compute_psnr(input_image, gt_image)
    ssim_val = compute_ssim(input_image, gt_image)
    return psnr_val, ssim_val

def main():
    datasets = ['Test100', 'Rain100H', 'Rain100L', 'Test2800', 'Test1200']
    num_set = len(datasets)
    psnr_alldatasets = 0
    ssim_alldatasets = 0

    for dataset in datasets:
        file_path = f'./results-Prompt-Mix/{dataset}/'
        gt_path = f'./dataset/test/{dataset}/target/'
        image_files = glob.glob(file_path + '*.png') + glob.glob(file_path + '*.jpg')
        gt_files = glob.glob(gt_path + '*.png') + glob.glob(gt_path + '*.jpg')
        img_num = len(image_files)

        total_psnr = 0
        total_ssim = 0
        valid_count = 0  # 计算有效结果的数量

        with ProcessPoolExecutor(max_workers=20) as executor:
            results = executor.map(process_images, image_files, gt_files)
            for result in results:
                if result[0] is not None and result[1] is not None:
                    psnr_val, ssim_val = result
                    total_psnr += psnr_val
                    total_ssim += ssim_val
                    valid_count += 1  # 只有有效的结果才计数

        if valid_count > 0:
            qm_psnr = total_psnr / valid_count
            qm_ssim = total_ssim / valid_count
        else:
            qm_psnr = 0
            qm_ssim = 0

        print(f'For {dataset} dataset PSNR: {qm_psnr:.6f} SSIM: {qm_ssim:.6f}')

        psnr_alldatasets += qm_psnr
        ssim_alldatasets += qm_ssim

    print(f'For all datasets PSNR: {psnr_alldatasets / num_set:.6f} SSIM: {ssim_alldatasets / num_set:.6f}')

if __name__ == '__main__':
    main()
