import cv2
import math
import numpy as np
import os

def DarkChannel(im, sz):
    # im: float image in [0,1], shape HxWx3
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def AtmLight(im, dark):
    # im: float image in [0,1], dark: float in [0,1]
    h, w = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 500), 1))

    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    # indices of pixels sorted by dark channel (ascending), we want the brightest dark values => last elements
    indices = darkvec.argsort()
    top_inds = indices[-numpx:]  # top numpx indices

    # sum corresponding RGB pixels and average
    atmsum = np.zeros((3,), dtype=np.float64)
    for ind in top_inds:
        atmsum += imvec[ind]
    A = (atmsum / len(top_inds)).reshape((1, 3))
    return A

def TransmissionEstimate(im, A, sz, omega=0.95):
    # im: float [0,1], A: shape (1,3)
    im3 = np.empty_like(im, dtype=np.float64)
    for ind in range(3):
        # clamp division to avoid values > 1
        im3[:, :, ind] = np.minimum(im[:, :, ind] / (A[0, ind] + 1e-8), 1.0)
    transmission = 1 - omega * DarkChannel(im3, sz)
    transmission = np.clip(transmission, 0.0, 1.0)
    return transmission

def Guidedfilter(I, p, r, eps):
    # I - guidance image (grayscale float64), p - filtering input (float64)
    # Implementation of box-filter based guided filter
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b
    return q

def TransmissionRefine(im, et, r=40, eps=1e-3):
    # im: original BGR uint8, et: raw transmission (float [0,1])
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    et64 = et.astype(np.float64)
    t = Guidedfilter(gray, et64, r, eps)
    t = np.clip(t, 0.0, 1.0)
    return t

def Recover(im, t, A, tx=0.1):
    # im: float [0,1], t: float [0,1], A: (1,3)
    res = np.empty_like(im, dtype=np.float64)
    t_safe = np.maximum(t, tx)  # avoid division by very small values
    # expand t_safe to 3 channels
    if t_safe.ndim == 2:
        t3 = np.repeat(t_safe[:, :, np.newaxis], 3, axis=2)
    else:
        t3 = t_safe
    for ind in range(3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t3[:, :, ind] + A[0, ind]
    res = np.clip(res, 0.0, 1.0)
    return res

def process_folder(input_folder,
                   output_root="./outputs",
                   patch_size=15,
                   omega=0.95,
                   guided_r=20,
                   guided_eps=1e-2,
                   t0=0.2):
    # create output subfolders
    output_dirs = {
        "processed": os.path.join(output_root, "processed"),
        "dark": os.path.join(output_root, "dark"),
        "trans_raw": os.path.join(output_root, "transmission_raw"),
        "trans_refined": os.path.join(output_root, "transmission_refined"),
        "A_light": os.path.join(output_root, "A_light")
    }
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)

    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith(extensions):
            continue
        print(" Processing:", filename)
        path = os.path.join(input_folder, filename)
        src = cv2.imread(path)
        if src is None:
            print(" Cannot read:", filename)
            continue

        # convert to float [0,1]
        I = src.astype(np.float64) / 255.0

        # dark channel (on float)
        dark = DarkChannel(I, patch_size)
        # save dark (scale to 0-255)
        dark_out = (np.clip(dark, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dirs["dark"], filename), dark_out)

        # atmospheric light
        A = AtmLight(I, dark)  # shape (1,3), float in [0,1] roughly
        # save A visualization as constant image
        A_img = np.ones_like(src, dtype=np.uint8) * 0
        A_vis = (np.clip(A, 0, 1) * 255).astype(np.uint8).reshape((1,1,3))
        A_img[:] = A_vis[0,0,:]
        cv2.imwrite(os.path.join(output_dirs["A_light"], filename), A_img)

        # transmission estimate (raw)
        te = TransmissionEstimate(I, A, patch_size, omega=omega)
        te_out = (np.clip(te, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dirs["trans_raw"], filename), te_out)

        # refine transmission
        t = TransmissionRefine(src, te, r=guided_r, eps=guided_eps)
        t_out = (np.clip(t, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dirs["trans_refined"], filename), t_out)

        # recover
        J = Recover(I, t, A, tx=t0)
        J_out = (np.clip(J, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dirs["processed"], filename), J_out)

    print("\n Done. Outputs saved under:", output_root)
    for k, v in output_dirs.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    input_folder = "./image"
    process_folder(input_folder,
                   output_root="./outputs",
                   patch_size=15,
                   omega=0.95,
                   guided_r=60,       
                   guided_eps=1e-4,
                   t0=0.1)
