from pyteomics import ms1
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.stats as stats
import scipy.signal as sig
import argparse
import pathlib
from scipy.optimize import curve_fit,least_squares
import os




def load_ms1(path):
    """Read all MS1 spectra from a file into a list."""
    spectra = []
    with open(path, "r") as f:
        for spect in ms1.read(f):
            spectra.append(spect)
    return spectra


def is_in_region(spectrum, min_mz, max_mz):
    """ return true or false , if m/z inside region. select only m/z , that inside the region [min_mz, max_mz]"""
    mz_array = spectrum['m/z array']
    return (mz_array>= min_mz) & (mz_array <= max_mz)


def get_intensities_of_region(spectrum, min_mz, max_mz):
    """Return all intensities of only m/z , that inside the region [min_mz, max_mz]."""
    intensity_array = spectrum["intensity array"]
    mask = is_in_region(spectrum, min_mz, max_mz)
    return intensity_array[mask]

def get_final_eic_intensities(spectra, min_mz, max_mz):
    """Return final intensities array of our EIC by summing intensity in the m/z window for each spectrum."""

    #  array of sum of intensities in the region per spectrum
    final_intensities = []

    for spect in spectra:

        # array of intensities in one region
        intensities_per_region = get_intensities_of_region(spect, min_mz, max_mz)

        # sum of intensities in each region
        sum_of_intensities_per_region = np.sum(intensities_per_region)

        # final EIC Intensities
        final_intensities.append(sum_of_intensities_per_region)

    return np.array(final_intensities)

def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / ( sigma**2))
def fit_curve(y):
    x=np.arange(1,len(y)+1)
    mu=np.mean(x)
    sigma=np.sqrt(5)
    initial_guess = [max(y), mu, sigma]
    params, error = curve_fit(gaus, x, y,p0=initial_guess,method='trf',nan_policy='omit')
    fit=gaus(x,*params)
    return fit
def dip_detect_correct(y):
    peak, _ = sig.find_peaks(y, height=np.mean(y))
    max_peak=max(y[peak])
    minPoint = max_peak + (max_peak - min(y[peak[0]:peak[-1]]))
    pointsX=np.array([peak[0] - 1,
                      peak[-1] - ((peak[-1] - peak[0]) / 2),
                      peak[-1] + 1])
    pointsY=np.array([y[peak[0] - 1]
                         , minPoint
                         ,y[peak[-1] + 1]])
    predict = interpolate.interp1d(pointsX,pointsY,kind='quadratic')
    interpolated = np.concatenate((y[:(peak[0] - 1)], predict(np.arange(1,len(y))[peak[0] - 1:peak[len(peak) - 1] + 1]),
                                           y[peak[len(peak) - 1] + 1:]))
    return interpolated
def try_flip(y):
    peak,_=sig.find_peaks(y,height=np.mean(y))
    max_peak=max(y[peak])
    result=np.concatenate((y[:peak[0]],[x+((y[peak[0]]-x)*2) for x in y[peak[0]:(peak[-1]-int((peak[-1]-peak[0])/2))]],[x+((y[peak[-1]]-x)*2) for x in y[peak[-1]-int((peak[-1]-peak[0])/2):peak[-1]]],y[peak[-1]:]))
    return result
def main():

    parser = argparse.ArgumentParser(
        description="Extract EICs from an MS1 file for one or many m/z regions."
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path to the .ms1 file"
    )

    parser.add_argument(
        "--smooth","-S",
        nargs=2,
        action="store",
        metavar=('Window Length', 'Poly order'),
        required=True,
        help="Savitzky-Golay Filter Parameters, ex. --smooth 10 3,! 1st param>=2nd param"
    )

    parser.add_argument(
        "--fit","-F",
        required=False,
        action="store_true",
        default=False,
        help="set this flag if a gaussian fitting should be done"
    )
    parser.add_argument(
        "--interpolate","-I",
        required=False,
        action="store_true",
        default=False,
        help="set this flag if an interpolation should be done"
    )

    parser.add_argument(
        "--region","-R",
        nargs=2,
        action="append",
        metavar=('MIN_MZ', 'MAX_MZ'),
        required=True,
        help="m/z region, example: --region 604 605   (can be repeated)"
    )

    args = parser.parse_args()

    path = pathlib.Path(args.path)

    if not path.exists() or path.suffix.lower() != ".ms1":
        print("ERROR: The input file must exist and must be .ms1 format.")
        return

    # Convert region strings to floats
    regions = [(float(a), float(b)) for a, b in args.region]
    # Load spectra
    print(f"Loading MS1 file: {path}")
    spectra = load_ms1(path)
    print(f"Loaded {len(spectra)} spectra.")

    # Extract EICs for each region

    for (i,(min_mz, max_mz)) in enumerate(regions):
        #
        final_intensities = get_final_eic_intensities(spectra, min_mz, max_mz)



        plt.figure(figsize=(10, 6))
        seconds = np.arange(1, len(final_intensities) + 1)
        plt.plot(seconds, final_intensities, label=f"EIC of Region [{min_mz}, {max_mz}]")

        smoothed_intensities=sig.savgol_filter(final_intensities, int(args.smooth[0]), int(args.smooth[1]))
        plt.plot(seconds, smoothed_intensities, label=f"EIC Smoothed")



        if args.interpolate:
            dip_corrected_intensities=dip_detect_correct(smoothed_intensities)
            plt.plot(seconds,dip_corrected_intensities, label=f"EIC's Dip Correction")

        if args.fit:
            if args.interpolate:
                fitted_intensities=fit_curve(dip_corrected_intensities)
            else:
                fitted_intensities=fit_curve(smoothed_intensities)
            plt.plot(seconds, fitted_intensities,'--',label=f"EIC's Gaussian Fit")

        plt.xlabel("Seconds")
        plt.ylabel("Total intensity")
        plt.title(f"Extracted Ion Chromatograms (EIC) of regions [{min_mz}, {max_mz}]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()


    plt.show()


if __name__ == "__main__":
    main()
