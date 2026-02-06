import os
import pandas as pd
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel
import torch
import h5py
import xdas
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.integrate import cumulative_trapezoid
import pickle
from joblib import Parallel, delayed
from itertools import product
from obspy.imaging.beachball import beachball, aux_plane
import sp_ratio_source_mech
import pick_arrivals_PhasenetDAS
import channels_takeoff_azim



def get_source_mech(files, dev, eqlon, eqlat, eqdepth, output_folder, pathToEqNet='/home/a/EQNet/predict.py', workers=-1, model=None, ignorePicksStartSec=5, ignorePicksEndSec=10, topN_median=30):
    
    """
    Implementirana metoda opisana u clanku Funabiki, Y., & 
    Miyazawa, M. (2025). Estimating focal mechanism of small
    earthquakes using S/P amplitude ratios of distributed
     acoustic sensing records.



    Unijeti datoteke (cak i kad je jedna treba biti lista). 
    Funkcija zapisuje parametre rasjeda u csv formatu i sliku 
    beachball-a u output folder. Zapisuje parametre s 
    najmanjim odstupanjem od odazenog zapisa te medijan 
    top n rezultata.


    Funkcija za odredivanje nastupnog vremena P i S vala 
    korisit PhaseNetDAS. Uzimaju se prvi pickovi P i S 
    vala za svaki kanal, ako postoje. Razmatraju se samo
    kanali za koji je odredjeno nastupno vrijeme P i S
    vala. Moze se postaviti da metoda ignorira pickove P 
    i S valova preblizu kraju i pocetku zapisa, bitno je 
    zbog velicine koristenog prozora.



    Parameters
    
    ----------------

    files : list
        Unijeti datoteke na kojima je zabiljezen potres. Bolje je 
        unijeti vise da nailasci P i S vala nisu blizu pocetka i 
        kraja zbog koristenog prozora. I kada je jedna treba biti 
        u listi.
    dev : str
        Uredjaj, 'febus' ili 'sintela'
    eqlon : float
        Longituda potresa, za racunanje azimuta, takeoff kuta
    eqlat : float
        Latituda
    eqdepth : float
        Dubina potresa u km.
    output_folder : str
        Folder u koji ce se spremati parametri rasjeda i slika
    pathToEqNet : str
        Put do skripte PhaseNetDAS-a.
    workers : int
        Broj procesora.
    model : str
        Put do TauPy modela koji ce se koristiti za odredjivanje
        teoretskih nastupa. Ako se nista ne stavi
        koristi se 1D model za koru juznog Jadrana "Ston"
    ignorePicksStartSec : int
        Ignoriraju se pickovi koji se unutar toliko sekundi od
         pocetka zapisa. Ponekad PhaseNetDAS pick-a nesto na
          pocetku i na kraju sto moze prouzrociti problem zbog
           prevelikog prozora.
    ignorePicksEndSec : int
        Ignoriraju se pickovi na kraju zapisa
    topN_median : int
        Broj najboljih rjesenja za izracunati parametre rasjeda.


    """
    if model == None:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        model = f'{SCRIPT_DIR}/merged_model_1D_Ston.npz'
    
    
    pick_arrivals_PhasenetDAS.pick_arrivals(files=files, dev=dev, pathToEqNet=pathToEqNet, workers=0)
    channels_takeoff_azim.get_takeoff_angles_azimuth(dev=dev, eqlon=eqlon, eqlat=eqlat, eqdepth=eqdepth, input_model=model, workers=workers)
    sp_ratio_source_mech.source_mech(files=files, dev=dev, output_folder=output_folder, ignorePicksStartSec=ignorePicksStartSec, ignorePicksEndSec=ignorePicksEndSec, workers=workers, topN_median=topN_median)