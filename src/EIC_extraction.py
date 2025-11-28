import scipy.optimize
from pyteomics import ms1
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.stats as stats
import scipy.signal as sig
import scipy.optimize as opt
import argparse
import pathlib
from scipy.optimize import curve_fit,least_squares
import lmfit
from sklearn.metrics import r2_score
import os
from BaselineRemoval import BaselineRemoval




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

def get_final_eic_intensities(spectra, protein_mz, protein_sampling_range)->np.ndarray:
    """Return final intensities array of our EIC by summing intensity in the m/z window for each spectrum."""

    #  array of sum of intensities in the region per spectrum
    final_intensities = []

    for spect in spectra:

        # array of intensities in one region
        intensities_per_region = get_intensities_of_region(spect, protein_mz-protein_sampling_range, protein_mz+protein_sampling_range)

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

    initial_guess = [np.max(y), mu, sigma]

    params, error = curve_fit(gaus, x, y,p0=initial_guess)
    fit=gaus(x,*params)

    return fit

#finds the peaks of the smoothed graph
#finds the smallest value between first and last peak: min
#then sets the mid point of interpolation to double of: biggest peak - min
#interpolates between first peak, mid point, last peak quadratically
def dip_detect_correct(y):
    """
    finds the peaks of the smoothed graph
    finds the smallest value between first and last peak: min
    then sets the mid point of interpolation to double of: biggest peak - min
    interpolates between first peak, mid point, last peak quadratically
    y: smoothed values of the y-axis
    Returns graph with dip replaced with interpolated values"""

    #get middle point by adding the highest peak to the difference between the highest peek and the lowest point
    peak, _ = sig.find_peaks(y, height=np.mean(y))
    max_peak=max(y[peak])
    minPoint = max_peak + (max_peak - min(y[peak[0]:peak[-1]]))

    pointsX=np.array([peak[0] - 1,
                      peak[-1] - ((peak[-1] - peak[0]) / 2),
                      peak[-1] + 1])
    pointsY=np.array([y[peak[0] - 1]
                         , minPoint
                         ,y[peak[-1] + 1]])
    #interpolate using first and last peak in x-axis and middle-point
    predict = interpolate.interp1d(pointsX,pointsY,kind='quadratic')

    #connect the interpolated values to the tails on either side of the left and right peak
    interpolated = np.concatenate((y[:(peak[0] - 1)], predict(np.arange(1,len(y))[peak[0] - 1:peak[len(peak) - 1] + 1]),
                                           y[peak[len(peak) - 1] + 1:]))

    return interpolated


def try_flip(y):
    peak,_=sig.find_peaks(y,height=np.mean(y))
    max_peak=max(y[peak])
    result=np.concatenate((y[:peak[0]],[x+((y[peak[0]]-x)*2) for x in y[peak[0]:(peak[-1]-int((peak[-1]-peak[0])/2))]],[x+((y[peak[-1]]-x)*2) for x in y[peak[-1]-int((peak[-1]-peak[0])/2):peak[-1]]],y[peak[-1]:]))
    return result


def get_2_peaks(y:np.ndarray[float],y_nonsmoothed)->np.ndarray[float]:
    """
    Returns y-values with the found noise created by ion suppression removed(replaced by np.nan)
    """
    peaks,_=sig.find_peaks(y,height=y.mean())
    x_c=peaks[0]+np.argmin(y[peaks[0]:peaks[-1]])
    right_peak=np.argmax(y_nonsmoothed[x_c:])
    left_peak=np.argmax(y_nonsmoothed[:x_c])
    actual_right=-1
    actual_left=-1
    right_side=y[right_peak+x_c:]
    left_side=y[:left_peak]
    done=False
    boundL,boundR=0.94,0.96
    while not done:
        for (i,x) in enumerate(right_side):
            if boundL*right_side[right_peak]<= x <=boundR*right_side[right_peak]:
                actual_right=x_c+i
                break
        if actual_right==-1:
            boundL-=0.01
        else:
            done=True
    done=False
    boundL,boundR=0.94,0.96
    while not done:
        for (i,x) in enumerate(left_side):
            if boundL*left_side[left_peak-1]<= x <=boundR*left_side[left_peak-1]:
                actual_left=i
                break
        if actual_left==-1:
            boundL-=0.01
        else:
            done=True
    new_y=np.concatenate((y_nonsmoothed[:actual_left],[np.nan]*len(y[actual_left:actual_right]),y_nonsmoothed[actual_right:]))
    return new_y,x_c

def different_approach_gaus(y:np.ndarray[float],x:np.ndarray[float]):
    """
    fits the curve of the tails created by ion suppression removal
    """
    mask=~np.isnan(y)
    x_fit=x[mask]
    y_fit=y[mask]
    params,_=curve_fit(gaus,x_fit,y_fit,maxfev=10000)

    return gaus(x,*params)

def gauss_constant_fit(y:np.ndarray[float],y_original:np.ndarray[float]):
    mod_gaus = lmfit.models.GaussianModel(nan_policy='propagate')
    mod_const=lmfit.models.ConstantModel(nan_policy='propagate')
    y_work=y[~np.isnan(y)]
    xdat=np.arange(1,len(y)+1)
    # pars = mod.make_params(c=np.mean(y_work),
    #                        center=xdat.mean(),
    #                        sigma=xdat.std(),
    #                        amplitude=xdat.std() * np.ptp(y_work)
    #                        )
    pars=mod_gaus.guess(y_work,x=xdat)
    mod=mod_gaus+mod_const
    out = mod.fit(y_work, pars, x=xdat)
    return out.best_fit
def gauss_from_jonathan_lmfit(y:np.ndarray[float],y_original:np.ndarray[float]):
    mod = lmfit.models.Model(new_gauss_from_jonathan,nan_policy='propagate')
    # mod = lmfit.models.GaussianModel(nan_policy='omit')
    y_work=y[~np.isnan(y)]
    xdat=np.arange(1,len(y)+1)
    pars=mod.make_params()
    # pars = mod.make_params(c=np.mean(y_work),
    #                        center=xdat.mean(),
    #                        sigma=xdat.std(),
    #                        amplitude=xdat.std() * np.ptp(y_work)
    #                        )
    out = mod.fit(y, pars, x=xdat)
    plt.figure(5, figsize=(8, 8))
    out.plot_fit()


def new_gauss_from_jonathan(y:np.ndarray[float],y_0,xc,w,A):
    return y_0 +((A/(w*np.sqrt(np.pi/2)))*np.exp((-2)*(((y-xc)/w)**2)))
def gaussian_fit_with_removed_dip(y,y_original,x_c):
    obj=BaselineRemoval(y_original)

    y_mask=~np.isnan(y)
    xdat=np.arange(1,len(y)+1)
    y_0=y_original[obj.ZhangFit().argmin()]
    w=np.sqrt(2)*np.std(y_original)
    A=xdat.std() * np.ptp(y_original)

    initial_p=[np.float64(y_0),np.float64(x_c),w,A]

    params,err=curve_fit(new_gauss_from_jonathan,xdat[y_mask],y[y_mask],p0=initial_p)

    return new_gauss_from_jonathan(xdat,*params)

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
        "--region","-R",
        nargs=2,
        action="append",
        metavar=('MIN_MZ', 'MAX_MZ'),
        required=True,
        help="-R <m/z value> <region size> , example: --region 604 4 (can be repeated)"
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
    plt.style.use('ggplot')

    # Extract EICs for each region
    for (i,(protein_mz, protein_sampling_range)) in enumerate(regions):

        plt.figure(figsize=(10, 6))

        #summation of intensities of mz and range
        final_intensities = get_final_eic_intensities(spectra, protein_mz, protein_sampling_range)
        seconds = np.arange(1, len(final_intensities) + 1)

        plt.scatter(seconds, final_intensities, label=f"EIC of Protein {protein_mz} with range {protein_sampling_range}")

        #smoothing
        smoothed_intensities=sig.savgol_filter(final_intensities, int(args.smooth[0]), int(args.smooth[1]))
        plt.plot(seconds,smoothed_intensities, label=f"smoothed EIC intensity of Protein {protein_mz}")

        if args.fit:
            print("Fitting EIC")
            #masking
            removed_dip,xc=get_2_peaks(smoothed_intensities,final_intensities)
            plt.plot(seconds,removed_dip,label="EIC after Masking",color='lavender')

            #fitting
            removed_dip_fitted=different_approach_gaus(removed_dip,seconds)
            r2=r2_score(final_intensities,removed_dip_fitted)
            plt.plot(seconds,removed_dip_fitted,'--',label=f"EIC with R^2 value of {r2}")

        plt.xlabel("Seconds")
        plt.ylabel("Total intensity")
        plt.title(f"Extracted Ion Chromatograms (EIC) of {protein_mz}m/z of range {protein_sampling_range}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
