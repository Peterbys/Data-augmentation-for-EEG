# Data augmentation for EEG

This is the reposity of the master thesis project Data augmentation for EEG, containing the scripts used for the thesis.<br>
Several packages are needed to run the scripts, including pytorch and mne. In particular the mne 0.21 dev version was used

# Contents
The freesurfer directory contains the scripts to convert the surfces to an .obj format as described in the thesis.
helper_functions.py contains helper functions used<br>
neural_networks.py contain the neural network architectures and classes for training sets<br>
cnn_example.py contains an example of training a convolutional network<br>


# Data
The data can be found at https://gofile.io/d/i35R7A <br>
There are two data files, fwd_list.pkl containing all the forward models for the subject. The keys for this list is 01,02,...,16<br>
The other file is the data_dsfactor9.pkl, containing all the different epoched data. The data is organized as a nested list, where the keys in order are:<br>
1, subjects: 01,02...,16<br>
2, runs: 1,2,..,6<br>
3, type: normal,scrambled,events,cov<br>

normal are all the normal epochs for the run<br>
scrambled are all the scrambled epochs for the run<br>
events is an array containing information about all the events during recording<br>
cov is the covariance computed before any event occurs.

# Questions and issues
If there are any questions or issues, don't hesitate to contact me at s144045@student.dtu.dk