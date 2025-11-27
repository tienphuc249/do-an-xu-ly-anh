import cv2
import os
import numpy as np

# --- MSE ---
def mse(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return np.mean((img1 - img2) ** 2)

# --- SSIM (Gaussian window) ---
def ssim(img1, img2):
    # expects grayscale images (uint8 or float), returns scalar mean SSIM
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.filter2D(img1, cv2.CV_32F, window)
    mu2 = cv2.filter2D(img2, cv2.CV_32F, window)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 * img1, cv2.CV_32F, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 * img2, cv2.CV_32F, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, cv2.CV_32F, window) - mu1_mu2

    # avoid division by zero
    denom1 = (mu1_sq + mu2_sq + C1)
    denom2 = (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (denom1 * denom2 + 1e-12)

    return float(np.mean(ssim_map))

# --- Compare by index (MSE + SSIM) ---
def compare_folders_by_index_with_ssim(folderA, folderB, output="result.txt", verbose=True):
    filesA = sorted([f for f in os.listdir(folderA) if os.path.isfile(os.path.join(folderA, f))])
    filesB = sorted([f for f in os.listdir(folderB) if os.path.isfile(os.path.join(folderB, f))])

    total = min(len(filesA), len(filesB))
    if total == 0:
        print("❌ Một trong hai thư mục rỗng hoặc không có file ảnh.")
        return

    sum_mse = 0.0
    sum_ssim = 0.0
    count = 0

    with open(output, "w", encoding="utf-8") as f:
       # header = f"Comparing by index: {folderA} <-> {folderB}\nTotal pairs: {total}\n\n"
        #f.write(header)
        #if verbose:
        #    print(header.strip())

        for i in range(total):
            fileA = filesA[i]
            fileB = filesB[i]

            pathA = os.path.join(folderA, fileA)
            pathB = os.path.join(folderB, fileB)

            imgA = cv2.imread(pathA, cv2.IMREAD_GRAYSCALE)
            imgB = cv2.imread(pathB, cv2.IMREAD_GRAYSCALE)

            if imgA is None or imgB is None:
                line = f"{fileA} ↔ {fileB} : FAILED_TO_READ\n"
                f.write(line)
                if verbose:
                    print(line.strip())
                continue

            # resize B -> A size to compare
            if (imgB.shape[0] != imgA.shape[0]) or (imgB.shape[1] != imgA.shape[1]):
                imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]), interpolation=cv2.INTER_AREA)

            M = mse(imgA, imgB)
            S = ssim(imgA, imgB)

            sum_mse += M
            sum_ssim += S
            count += 1

            line = f"{fileA} ↔ {fileB} : MSE={M:.4f}, SSIM={S:.4f}\n"
            f.write(line)
            if verbose:
                print(line.strip())

        if count > 0:
            mean_mse = sum_mse / count
            mean_ssim = sum_ssim / count
        else:
            mean_mse = float('nan')
            mean_ssim = float('nan')

        summary = f"\nSUMMARY over {count} pairs: mean MSE = {mean_mse:.4f}, mean SSIM = {mean_ssim:.4f}\n"
        f.write(summary)
        if verbose:
            print(summary.strip())

    print("✔ Kết quả đã lưu vào", output)
    return {#"pairs": count,
            "mean_mse": mean_mse, "mean_ssim": mean_ssim}

# --- Example usage ---
if __name__ == "__main__":
    compare_folders_by_index_with_ssim(
        r"D:\do an xu ly anh\code\outputs\processed",
        r"D:\do an xu ly anh\code\origin",
        output="result.txt",
        verbose=True
    )
