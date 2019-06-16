DGMA Implementation
===================

Requirements
-------------
python>=3.5
matplotlib==2.0.2
numpy==1.13.0

How to install
--------------

    1. Create a virtual environment
       (For Ubuntu or Linux system)
       virtualenv -p /usr/bin/python3 .env

    2. source .env/bin/activate

    3. pip install matplotlib==2.0.2 numpy==1.13.0

    4. Ensure that backend for matplotlib is set to TkAgg for dynamic rendering.
       TkAgg as a backend is recommended for Ubuntu or Linux systems.
       Check your matplotlibrc file. The location of it depends on your system.

Experiments
-----------

Experiment 1 Python files with different distance metrics:

    1. python main_euclidean_exp1.py rpg
    2. python main_max_norm_exp1.py rpg
    3. python main_manhattan_exp1.py rpg

Corresponding results are stored in euclidean.txt, maxnorm.txt and manhattan.txt

Experiment 2 Python files with different distance metrics:

    1. python main_euclidean_exp2.py rpg
    2. python main_max_norm_exp2.py rpg
    3. python main_manhattan_exp2.py rpg

Corresponding results are stored in :
    exp2_euclidean.txt, exp2_maxnorm.txt and exp2_manhattan.txt
