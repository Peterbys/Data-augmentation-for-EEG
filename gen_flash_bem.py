#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:55:14 2020

@author: peter
"""
import mne
subject = 'sub-03_free'
flash_path = 'Downloads/flashtest/ds117/sub003/anatomy/FLASH'
subjects_dir = '/home/peter/my_subjects'
mne.bem.make_flash_bem(subject, overwrite=False, show=True, subjects_dir=subjects_dir, flash_path=flash_path, copy=False, verbose=None)

