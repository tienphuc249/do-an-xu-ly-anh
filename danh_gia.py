import cv2
import os
from rapidfuzz import process
import numpy as np

def mse(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return np.mean((img1 - img2) ** 2)

def ssim(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # dùng ddepth = CV_32F để tránh mất dữ liệu
    mu1 = cv2.filter2D(img1, cv2.CV_32F, window)
    mu2 = cv2.filter2D(img2, cv2.CV_32F, window)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 * img1, cv2.CV_32F, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, cv2.CV_32F, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, cv2.CV_32F, window) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def compare_folders(folderA, folderB, output="result.txt"):
    filesA = sorted(os.listdir(folderA))
    filesB = sorted(os.listdir(folderB))

    namesB = {f: f for f in filesB}

    with open(output, "w", encoding="utf-8") as f:
        for fileA in filesA:
            match, score, _ = process.extractOne(fileA, namesB.keys())

            if score < 30:
                print(f"⚠ Không tìm được file phù hợp cho {fileA}")
                continue

            pathA = os.path.join(folderA, fileA)
            pathB = os.path.join(folderB, match)

            img1 = cv2.imread(pathA, 0)
            img2 = cv2.imread(pathB, 0)

            if img1 is None or img2 is None:
                continue

            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            M = mse(img1, img2)
            S = ssim(img1, img2)

            f.write(f"{fileA} ↔ {match} : MSE={M:.4f}, SSIM={S:.4f}\n")
            print(f" {fileA} ↔ {match} : MSE={M:.4f}, SSIM={S:.4f}")

    print("\n successful!", output)


compare_folders(
    r"D:\do an xu ly anh\code\outputs\processed",
    r"D:\do an xu ly anh\code\origin",
    output="result.txt"
)

