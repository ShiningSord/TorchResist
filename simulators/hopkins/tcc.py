import pickle
import argparse

import numpy as np
from sklearn.utils.extmath import randomized_svd

def getFreqs(pixel, canvas): 
    size = round(canvas / pixel)
    basic = np.fft.fftshift(np.fft.fftfreq(size, d=pixel))
    freqX = basic.reshape((1, -1)).repeat(size, axis=0)
    freqY = basic.reshape((-1, 1)).repeat(size, axis=1)
    assert freqX.shape[0] == size and freqX.shape[1] == size
    assert freqY.shape[0] == size and freqY.shape[1] == size
    return freqX, freqY

def srcPoint(pixel, canvas): 
    freqX, freqY = getFreqs(pixel, canvas)
    result = (freqX == 0) * (freqY == 0)
    return result.astype(np.double)

def funcPupil(pixel, canvas, na, lam, defocus=None, refract=None): 
    limit = na / lam
    freqX, freqY = getFreqs(pixel, canvas)
    result = np.sqrt(freqX**2 + freqY**2) < limit
    result = result.astype(np.double)
    if not defocus is None: 
        assert not refract is None
        print(f"{np.max(freqX), np.max(freqY)}")
        mask = result > 0
        opd = defocus * (refract - np.sqrt(refract**2 - lam**2 * ((freqX*mask)**2 + (freqY*mask)**2)))
        shift = np.exp(1j * (2 * np.pi / lam) * opd)
        result = result * shift
    return result

def genTCC(src, pupil, pixel, canvas, thresh=1.0e-6): 
    size = round(canvas / pixel)
    pupilFFT = np.fft.fftshift(np.fft.fft2(pupil)) # h
    pupilStar = pupilFFT.conj() # h*
    srcFFT = np.fft.fftshift(np.fft.fft2(src/np.sum(src))) # J
    print(f"Creating big matrix: {pupilStar.shape + pupilStar.shape}")
    w = np.zeros(pupilStar.shape + pupilStar.shape, dtype=np.complex64)
    for idx in range(pupilStar.shape[0]): 
        for jdx in range(pupilStar.shape[1]): 
            srcShifted = np.roll(srcFFT, shift=(idx, jdx), axis=(0, 1))
            srcShifted = np.flip(srcShifted, axis=(0, 1))
            w[idx, jdx] = srcShifted * pupilFFT[idx, jdx] * pupilStar / (np.prod(pupil.shape) * np.prod(src.shape))
    sizeAll = np.prod(pupilStar.shape)
    w = w.reshape(sizeAll, sizeAll)
    print(f"Running SVD for matrix {w.shape}")
    matU, matS, matVT = randomized_svd(w, n_components=64, n_iter=32, n_oversamples=16)
    print(f"SVD results: {matU.shape, matS.shape, matVT.shape}, weights = {matS}")
    phis = []
    weights = []
    for idx, weight in enumerate(matS): 
        if not thresh is None and weight >= thresh: 
            phis.append(matU[:, idx].reshape(size, size) * (size*size))
            weights.append(matS[idx])
    return phis, weights


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='TCC generation')

    parser.add_argument('-p', '--pixelsize', type=int, default=8)
    parser.add_argument('-c', '--canvas', type=int, default=512)
    parser.add_argument('-w', '--wavelength', type=int, default=193)
    parser.add_argument('-d', '--defocus', type=float, default=None)
    parser.add_argument('-o', '--output', type=str, required=True)

    args = parser.parse_args()

    PIXEL = args.pixelsize
    CANVAS = args.canvas
    DEFOCUS = args.defocus
    LAMBDA = args.wavelength # nm
    SIZE = round(CANVAS / PIXEL)
    REFRACT = 1.44 # 1.0 #
    SINMAX = 0.9375 # 0.7 #
    NA = REFRACT * SINMAX

    pupil = funcPupil(PIXEL, CANVAS, NA, LAMBDA, defocus=DEFOCUS, refract=REFRACT)
    circ = srcPoint(PIXEL, CANVAS)
    phis, weights = genTCC(circ, pupil, PIXEL, CANVAS)
    print(f"Get {len(weights)} TCC: {weights}")
    with open(args.output, "wb") as fout: 
        pickle.dump((phis, weights), fout)
