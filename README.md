"NE2001 for extra-galactic astronomy"

This package is designed to calculate interesting/useful statistics related to scintillation and variability.
The assumption is that the objects of interest are extragalactic and thus we only need consider the total electron column.
Thus models like NE2001 are too simplistic, and so we use Hα as a proxy for electron column.

# Installing
Code: 
- There is no setup.py or install process
- Just run `git clone git@github.com:PaulHancock/RISS19.git` to get the code

Data: 
- The data in `data/` are too large (143Mb) for git, so you'll need `git-lfs`
- See [here](https://git-lfs.github.com/) for instructions on how to install `git-lfs`
- The above instructions are not super clear, but once you run the scripts the [install](https://packagecloud.io/github/git-lfs/install) page, you'll probably have to do something like `apt install git-lfs` to actually install the extension.
- Once installed on your system you should run `git lfs install` to add the appropriate git hooks
- Finally, you can run `git lfs pull` to download the two large data files in the `data/` directory

# Usage
```
usage: varcalc.py [-h] [--Halpha] [--xi] [--mod] [--sm] [--timescale] [--rms1y] [--theta] [--nuzero] [--fzero] [--dist] [--all]
                  [--in INFILE] [--incol COLS COLS] [--out OUTFILE] [--append] [--pos POS POS] [-g] [--debug]
                  [--freq FREQUENCY] [--dist_in DIST_IN] [--vel VELOCITY]

optional arguments:
  -h, --help         show this help message and exit

Output parameter selection:
  --Halpha           Calculate Hα intensity (Rayleighs)
  --xi               Calculate ξ (dimensionless)
  --mod              Calculate modulation index (fraction)
  --sm               Calculate scintillation measure (kpc m^{-20/3})
  --timescale        Calculate timescale of variability (years)
  --rms1y            Calculate rms variability over 1 year (fraction/year)
  --theta            Calculate the scattering disk size (deg)
  --nuzero           Calculate the transition frequency (GHz)
  --fzero            Calculate the Fresnel zone (deg)
  --dist             Calculate the model distance
  --all              Include all of the above parameter calculations

Input and output data:
  --in INFILE        Table of coordinates
  --incol COLS COLS  Column names to read from input. [ra,dec]
  --out OUTFILE      Table of results
  --append           Append the data to the input data (write a new file)
  --pos POS POS      Single coordinates in ra/dec degrees
  -g, --galactic     Interpret input coordinates as l/b instead of ra/dec (default False)
  --debug            Debug mode (default False)

Input parameter settings:
  --freq FREQUENCY   Frequency in MHz
  --dist_in DIST_IN  Distance to scattering screen in kpc
  --vel VELOCITY     Relative motion of screen and observer in km/s
```

# Citation
If you use this code please cite the following publicattion:

```
@ARTICLE{2019arXiv190708395H,
       author = {{Hancock}, P.~J. and {Charlton}, E.~G. and {Macquart}, J-P. and {Hurley-Walker}, N.},
        title = "{Refractive Interstellar Scintillation of Extra-galactic Radio Sources I: Expectations}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2019,
        month = jul,
          eid = {arXiv:1907.08395},
        pages = {arXiv:1907.08395},
archivePrefix = {arXiv},
       eprint = {1907.08395},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190708395H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```