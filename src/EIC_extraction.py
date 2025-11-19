from pyteomics import ms1
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
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
        "--region",
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

    for (min_mz, max_mz) in regions:
        #
        final_intensities = get_final_eic_intensities(spectra, min_mz, max_mz)
        plt.figure(figsize=(10, 6))
        seconds = np.arange(1, len(final_intensities) + 1)
        plt.plot(seconds, final_intensities, label=f"EIC of Region [{min_mz}, {max_mz}]")
        plt.xlabel("Seconds")
        plt.ylabel("Total intensity")
        plt.title(f"Extracted Ion Chromatograms (EIC) of regions [{min_mz}, {max_mz}]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()


    plt.show()


if __name__ == "__main__":
    main()
