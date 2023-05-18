# EvalOCL (Still under construction-- will be ready before May End)


This repository contains the code for the paper:

**Rapid Adaptation in Online Continual Learning: Are We Evaluating It Right?** 
[Hasan Abed Al Kader Hammoud](), [Ameya Prabhu](https://drimpossible.github.io), [Ser-Nam Lim](https://drimpossible.github.io), [Philip H.S. Torr](https://www.robots.ox.ac.uk/~phst/), [Adel Bibi](https://www.robots.ox.ac.uk/~phst/), [Bernard Ghanem](https://www.robots.ox.ac.uk/~phst/)
[[Arxiv](https://arxiv.org/abs/2305.09275)]
[[PDF](https://drimpossible.github.io/documents/EvalOCL.pdf)]
[[Bibtex](https://github.com/drimpossible/EvalOCL/#citation)]

## Installation and Dependencies

* Install all requirements required to run the code on a Python 3.9 environment by:
 ```	
# First, activate a new virtual environment
pip3 install -r requirements.txt
 ```
 
* Create three additional folders in the repository `data/`, `data_scripts/` and `logs/` which will store the datasets and logs of experiments. Point `--order_file_dir` and `--log_dir` in `src/opts.py` to locations of these folders.

## Downloading Data

* Follow instructions from here for downloading datasets.

## Usage

## Replicating Results



##### If you discover any bugs in the code please contact me, I will cross-check them with my nightmares.

## Citation

We hope Near-Future Accuracy is a reliable measure, and this codebase is useful for your cool CL work! We have tried to keep the codebase simple, readable but very compute/memory efficient. To cite our work:

```
@article{hammoud2023rapid,
      title={Rapid Adaptation in Online Continual Learning: Are We Evaluating It Right?}, 
      author={Hasan Abed Al Kader Hammoud and Ameya Prabhu and Ser-Nam Lim and Philip H. S. Torr and Adel Bibi and Bernard Ghanem},
      year={2023},
      journal={arXiv preprint arXiv:2305.09275},
}
```
