import numpy as np
import scipy.signal as ss
import pylab as py


def cwt(x, MinF, MaxF, Fs, w=7.0, df=1.0, plot=False):
    """
    w - parametr falki Morleta, wiąże się z jej częstościa centralną i skalą w następujacy sposób:
    f = 2*a*w / T
    gdzie: a-skala,  T-długość sygnału w sek.
    """
    T = len(x) / Fs
    M = len(x)
    t = np.arange(0, T, 1. / Fs)
    freqs = np.arange(MinF, MaxF, df)
    P = np.zeros((len(freqs), M))
    X = np.fft.fft(x)  # transformacja sygnału do dziedziny czestosci
    for i, f in enumerate(freqs):  # petla po kolejnych czestosciach
        a = T * f / (2 * w)  # obliczenie skali dla danej czestosci
        psi = np.fft.fft(ss.morlet(M, w=w, s=a,
                                   complete=True))  # transformacja falki Morleta do dziedziny czestosci. W bardziej wydajnym kodzie moznaby zastosowac analityczna postac tej falki w dziedzinie czestosci.
        psi /= np.sqrt(np.sum(psi * psi.conj()))  # normalizacja energii falki
        CWT = np.fft.fftshift(np.fft.ifft(X * psi))
        P[i, :] = (CWT * CWT.conj()).real

    if plot:
        py.imshow(P, aspect='auto', origin='lower', extent=(0, T, MinF, MaxF))
        py.show()
    return P, f, t


def time_freq_scipy(signal):
    powers = np.zeros((19, 40, 600))
    for ch in range(19):

        P_all = 0
        for e_i in range(len(signal)):
            _s = signal[e_i][ch, :]

            cwtmatr, f, t = cwt(_s, 20, 80, 1000, 10)

            P_all += np.abs(cwtmatr)
        P_all = np.log(P_all / len(signal))
        powers[ch, :, :] = P_all

        print("channel: ", ch + 1)

    return powers