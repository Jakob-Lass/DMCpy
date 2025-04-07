DMCpy
=====
DMCpy is a software package designed for reduction of neutron powder and single crystal diffraction data from [DMC](https://www.psi.ch/en/sinq/dmc) at the [Paul Scherrer Institute](https://psi.ch/). This instrument a combined powder and single-crystal diffractometer located at SINQ, Villigen, Switzerland. 
This software package covers conversion and data analysis both neutron powder measurements as well as single crystal experiments. Through DMCpy, users can carry out initial conversions and normalizations of the data and perform data analysis through cuts and integration methods. 


## Installation
For the installation, it is recommended to create either a virtual python environment (e.g. through [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)) and then install the package through the Python Package Index ([PyPI](https://pypi.org/project/DMCpy/)) by issuing 

```
pip install DMCpy
```
or 
```
python3 -m pip install DMCpy
```

This will install the software within the environment allowing data analysis of the DMC data structures. Further details are found in our [documentation](https://dmcpy.readthedocs.io/en/latest/introduction.html) 


> **_NOTE:_**  For neutron powder experiments, the pre-installed DMCpy version on the instrument computer is sufficient to convert, calibrate, and extract powder patterns to be used in crystallography software (like [FullProf](https://www.ill.eu/sites/fullprof/), [Jana](https://doi.org/10.1515/zkri-2023-0005), etc.)

## Documentation and Tutorials
A series of tutorials and explanations of features have been prepared for DMCpy and is available through our [ReadTheDocs page](https://dmcpy.readthedocs.io/en/latest/). Here, both an introduction to the neutron instrument as well as the most used analysis methods are presented.



## Contribute
To contribute or report bugs or suggestions, please visit the [issues](https://github.com/Jakob-Lass/DMCpy/issues) and/or [pull requests](https://github.com/Jakob-Lass/DMCpy/pulls). 
Before laying out a full-blown pull request, please contact the DMCpy maintainers for a expectation clarification discussion.


## Contact
For a direct communications means please send an E-Mail to [MJOLNIRPackage](mailto:MJOLNIRPackage@gmail.com) or contact the [instrument responsibles](https://www.psi.ch/en/sinq/dmc/people) for the DMC instrument. 



