# binary classification curves - algorithms speed test

Test which variation of FPR computation is faster.

Tested with python version `3.10.12` and environment in `requirements.txt`.

# install

To install an environment with conda:

```bash
cd path/to/gist/folder
conda create -n "binclf-speed-test" python=3.10.12 -y
conda activate binclf-speed-test
pip install -r requirements.txt
```

# examples:

```bash
python binclf-speed-test.py --resolution 128 --num_images 100 --num_thresholds 1000 --seed 0 --algorithm numpy_numba --device cpu --mode set
```

```bash
python binclf-speed-test.py --resolution 128 --num_images 100 --num_thresholds 1000 --seed 0 --algorithm numpy_itertools --device cpu --mode perimg
```
