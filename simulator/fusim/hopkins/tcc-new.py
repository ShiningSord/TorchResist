"""
Author: Guojin Chen @ CUHK-CSE
Homepage: https://gjchen.me
Date: 2024-01-05 10:08:59
LastEditTime: 2024-01-10 07:19:43
Contact: cgjcuhk@gmail.com
Description:

NOTE:

torch.svd can not support complex numbers
torch.linalg.svd is slow  (25s for 2000 x 2000)
sci.linalg.svd is slow    (25s for 2000 x 2000)

so I use sci.sparse.linalg.svds for fast calculating. (1.5s for 2000 x 2000)

For results:

the sci.linalg.svd == torch.linalg.svd
the sci.sparse.linalg.svds != sci.linalg.svd


TODO:
If change the device to GPU, we need to compare the runtime performance again.
"""
import shelve
import time

import scipy as sci
import torch

from src import utils
from src.models.litho.lens import LensList
from src.models.litho.source import Source

log = utils.get_pylogger(__name__)


class TCC:
    """"""

    def __init__(self, source, lens):
        self.s = source
        self.s.update()
        self.s.ifft()

        self.lens = lens
        self.lens.update()
        self.lens.calPupil()
        self.lens.calPSF()

        self.order = 7
        self.psf = lens.data

    def calMutualIntensity(self):
        self.gnum, self.fnum = self.s.data.shape
        J = torch.zeros((self.gnum, self.fnum, self.gnum, self.fnum), dtype=torch.complex64)
        for ii in range(self.gnum):
            for jj in range(self.fnum):
                J[:, :, ii, jj] = self.s.spatMutualData.real[
                    (self.gnum - ii - 1) : (2 * self.gnum - ii - 1),
                    (self.fnum - jj - 1) : (2 * self.fnum - jj - 1),
                ]
        self.jsource = torch.reshape(J, (self.gnum * self.fnum, self.gnum * self.fnum))

    def calSpatTCC(self):
        H = torch.reshape(self.psf, (torch.prod(torch.tensor(self.psf.shape)), 1))
        self.tcc2d = self.jsource * torch.matmul(H, H.t()) / self.s.detaf / self.s.detag

    def svd(self):
        self.spat_part = torch.zeros(
            (self.gnum, self.fnum, self.gnum, self.fnum), dtype=torch.complex64
        )

        self.freq_part = torch.zeros(
            (self.gnum, self.fnum, self.gnum, self.fnum), dtype=torch.complex64
        )

        tcc4d = self.tcc2d.reshape((self.gnum, self.fnum, self.gnum, self.fnum))
        self.spat_part[:] = torch.fft.ifftshift(tcc4d)

        self.freq_part = torch.fft.fftn(self.spat_part)
        tcc4df = torch.fft.fftshift(self.freq_part)
        tcc2df = tcc4df.reshape((self.gnum * self.fnum, self.gnum * self.fnum))

        tic = time.time()
        U, S, V = sci.sparse.linalg.svds(tcc2df.numpy(), self.order)  # faster than torch svd
        U = torch.from_numpy(U.copy())
        S = torch.from_numpy(S.copy())
        log.info(f"sci.sparse.linalg.svds taking {(time.time() - tic):.3f} seconds")

        self.coefs = S[0 : self.order]
        self.kernels = torch.zeros((self.gnum, self.fnum, self.order), dtype=torch.complex64)
        for ii in range(self.order):
            self.kernels[:, :, ii] = torch.reshape(U[:, ii], (self.gnum, self.fnum))


class TCCList(TCC):
    def __init__(self, source: Source, lensList: LensList, order: int = 7):
        self.s = source
        self.s.update()
        self.s.ifft()

        self.lensList = lensList
        self.lensList.calculate()
        self.PSFList = lensList.sDataList
        self.order = order
        self.focusList = lensList.focusList
        self.focusCoef = lensList.focusCoef
        self.kernelList = []
        self.coefList = []

        """
        Process calculation
        """
        self.calculate()

    def calculate(self):
        self.calMutualIntensity()
        for ii in self.PSFList:
            self.psf = ii
            self.calSpatTCC()
            self.svd()
            self.coefList.append(self.coefs)
            self.kernelList.append(self.kernels)


class TCCDB:
    def __init__(self, dbPath):
        self.s = None
        self.PSFList = []
        self.order = None
        self.focusList = []
        self.focusCoef = []
        self.kernelList = []
        self.coefList = []
        self.dbPath = dbPath

    def save_db(self, tcclist):
        self.s = tcclist.s
        self.PSFList = tcclist.PSFList
        self.order = tcclist.order
        self.focusList = tcclist.focusList
        self.focusCoef = tcclist.focusCoef
        self.kernelList = tcclist.kernelList
        self.coefList = tcclist.coefList

        db = shelve.open(f"{self.dbPath}")
        db["TCCList"] = self
        db.close()
        print(f"tcc list saved to {self.dbPath}")

    def load_db(self):
        db = shelve.open(f"{self.dbPath}")
        tcclist = db["TCCList"]
        self.s = tcclist.s
        self.PSFList = tcclist.PSFList
        self.order = tcclist.order
        self.focusList = tcclist.focusList
        self.focusCoef = tcclist.focusCoef
        self.kernelList = tcclist.kernelList
        self.coefList = tcclist.coefList
        print(f"Load tcc list from {self.dbPath}")
