import xdas
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.integrate import cumulative_trapezoid
import pickle
from joblib import Parallel, delayed
from itertools import product
import os
from obspy.imaging.beachball import beachball, aux_plane


def _bandpass_filter(data, fs, lowcut, highcut, axis=1, order=4):

    """
    Funkcija implementira Butterworthov bandpass filter. Vraca filtrirane podatke.

    Parameters

    ---------------

    data: numpy array
        2D array
    fs: int
        Frekvencija uzorkovanja
    lowcut: float
        Donja granica filtra
    highcut: float
        Gornja granica filtra
    axis: int
        Dimenzija array-a koja se filtrira
    order: int
        Red filtra

    --------------

    Returns
    filtered, array, isti oblik kao i ulazni

    """


    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = butter(order, [low, high], btype='bandpass', output='sos')

    filtered = sosfiltfilt(sos, data, axis=axis)

    return filtered



def _longest_half_cycle(x, fs):

    """
    Funkcija uzima array x i frekvenciju uzorkovanja fs. Vraca duljinu najduljeg 
    poluperioda u sekundama.

    Parameters

    ------------

    x: array, 1D
    fs: int
        Frek uzorkovanja
    
    -----------

    Returns
    Trajanje u sekundama najduljeg poluperioda.

    """

    zero_crossings = np.where(np.diff(np.signbit(x)))[0]
    if len(zero_crossings) < 2:
        return 0
    half_cycles = np.diff(zero_crossings) / fs
    return np.max(half_cycles)


def _get_clean_channels(data, phase_picks, nSecBefore, nSecAfter):

    """
    Funkcija vraca "ciste" kanale prema radu Funabiki, Y., & 
    Miyazawa, M. (2025). Funkcija trazi vremenske indekse u intervalu od nSecBefore
    prije do nSecAfter nakon (defaultno od 0.5 s prije do 2.0 s poslije) pickanog
    nailaska S vala kada dolazi do skoka u fazi od -pi ili pi za svaki od kanala.
    Kanali u kojima dolazi i do pozitivnih (+pi) i do negativnih (-pi) skokova u 
    fazi oznacuju se kao potencijalno losi. Analiziraju se ti potencijalno losi 
    kanali i oznacuju se kao losi oni u kojima je minimalni interval izmedju pozitivnih i 
    negativnih skokova manji od najveceg poluperioda u najblizem kanali bez pozitivnih i 
    negativnih skokova u fazi.

    Parameters

    ----------------

    data: ndarray
        2D array s podacima
    phase_picks: pandas DataFrame
        DataFrame objekt s vremenskim indeksima nailaska vala za svaki od kanala
    nSecBefore: float
        Broj sekundi prije nailaska vala za uzeti u obzir u analizi
    nSecAfter: float
        Broj sekundi nakon nailaska vala za uzeti u obizr u analizi

    -----------------

    Returns
    cleanChannelsGlob, phase_picks_clean
    cleanChannelsGlob: array, Bool
        Bool array iste velicine kao i broj kanala u originalnim podacima, True ako je 
        kanal "cist", a False ako nije
    phase_picks_clean: DataFrame
        Objekt koji sadrzi indekse nailaska S vala samo za kanale koji su oznaceni kao
        "cisti"


    """

    # frekv uzorkovanja
    delta_t = pd.to_timedelta(data.time[1].data - data.time[0].data).total_seconds()
    fs = int(round(1/delta_t))
    
    # kanali za koje postoje pickove zadane faze (S defaultno) i indeksi indeksi nastupa pickanog vala
    channelsWithPicks = phase_picks.channel_index.to_numpy()
    picks = phase_picks.phase_index.to_numpy()

    # definiraju se indeksi oko nastupa vala koje treba obuhvatiti za svaki kanal te se uzimaju podaci 
    # iz originalnih za sve kanale za koje postoje pickovi u prije definiranom intervalu
    times = picks[:, None] + np.arange(int(-nSecBefore*fs), int(nSecAfter*fs))
    sel_data = data.data[times, channelsWithPicks[:, None]]
    nChannelsWithPicks = sel_data.shape[0]

    positive_skips = []
    negative_skips = []


    for ch_idx in range(nChannelsWithPicks):


        # Za svaki od promatranih kanala se racuna Hilbertov transform te se iz toga racuna faza u svakoj tocki.
        # np.angle definira kut od -pi do pi tako da ponekad dolazi do umjetnih skokova kada se racuna razlika. 
        # Stoga se razliku izmedju kuteva uzimaju kao najmanja kutna udaljenost na kruznici, moze biti pozitivna ili
        # negativna, pomocu formule deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi. Kao vremena pozitivnih 
        # skokova u fazi uzimaju se vremena s razlikama vecim od pi - eps, a kao vremena negativnih skokova uzimaju
        # se vremena s razlikama manjim od -pi + eps. Eps je definiran kao 1e-3.
        chdata = sel_data[ch_idx]
        phi = np.angle(hilbert(chdata))
        
        deltaPhi = phi[1:] - phi[:-1]

        deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi

        eps = 1e-3
        
        positive_times = times[ch_idx][1:][deltaPhi >= (np.pi - eps)]
        negative_times = times[ch_idx][1:][deltaPhi <= (-np.pi + eps)]

        positive_skips.append(positive_times)
        negative_skips.append(negative_times)


    # Potencijalno losi kanali su oni s pozitivnim i negativnim skokovima u fazi.

    candidateChannels = np.array([(len(positive_skips[ch]) > 0) and (len(negative_skips[ch]) > 0) for ch in range(nChannelsWithPicks)])


    # Definira se array cleanChannelsGlob, boolean, jesu li kanali "cisti" ili ne, globalni indeks, svi kanali. 
    # Definira se array cleanLocal, indeksi kanala koji nisu potencijalno losi u "lokalnim" koordinatama, tj. kada
    # se gledaju samo kanali za koje postoje pickovi.
    cleanChannelsGlob = np.zeros(data.shape[1], dtype=bool)
    cleanLocal = np.where(~candidateChannels)[0]
    cleanChannelsGlob[channelsWithPicks[cleanLocal]] = True
    global_clean = channelsWithPicks[cleanLocal]
    
    for ch_idx in np.where(candidateChannels)[0]:

        # Gleda se minimalna vremenska razlika izmedju pozitivnih i negativnih skokova u fazi u potencijalno losim
        # kanalima.

        dt_min = np.min(np.abs(positive_skips[ch_idx][:, None] - negative_skips[ch_idx][None, :])) / fs


        # Trazi se najblizi kanal koji nije potencijalno los

        global_ch = channelsWithPicks[ch_idx]

        nearest_clean = global_clean[np.argmin(np.abs(global_clean - global_ch))]
        nearest_idx = np.where(channelsWithPicks == nearest_clean)[0][0]

        # Racuna se najdulji poluperiod u najblizem "cistom" kanalu
        ref_data = sel_data[nearest_idx]
        T_half_max = _longest_half_cycle(ref_data, fs=fs)


        # Ako je izracunata razlika izmedju poz i neg skokova manja od maks poluperioda onda kanal nije los
        if dt_min < T_half_max:
            cleanChannelsGlob[global_ch] = True

    phase_picks_clean = phase_picks[phase_picks.channel_index.isin(np.where(cleanChannelsGlob)[0])]
    
    return cleanChannelsGlob, phase_picks_clean



def _Rp(phi, delta, lam, i, phi_source_rec):

    """
    Funkcija racuna teoretski Rp za dane velicine.

    Parameters

    ------------

    phi: float
        Strike
    delta: float
        Dip
    lam: float
        Rake
    i: array
        Kutevi emergencije P vala
    phi_source_rec: array
        Azimuti potresa i kanala.

    -----------

    Returns
    Rp, array
    """

    return (np.cos(lam)*np.sin(delta)*np.sin(i)**2*np.sin(2*(phi_source_rec - phi)) - 
            np.cos(lam)*np.cos(delta)*np.sin(2*i)*np.cos(phi_source_rec - phi) +
            np.sin(lam)*np.sin(2*delta)*(np.cos(i)**2 - np.sin(i)**2*np.sin(phi_source_rec - phi)**2) +
            np.sin(lam)*np.cos(2*delta)*np.sin(2*i)*np.sin(phi_source_rec - phi))

def _Rsv(phi, delta, lam, i, phi_source_rec):

    """
    Funkcija racuna teoretski Rsv za dane velicine.

    Parameters

    ------------

    phi: float
        Strike
    delta: float
        Dip
    lam: float
        Rake
    i: array
        Kutevi emergencije S vala
    phi_source_rec: array
        Azimuti potresa i kanala.

    -----------

    Returns
    Rsv, array
    """

    return (np.sin(lam)*np.cos(2*delta)*np.cos(2*i)*np.sin(phi_source_rec - phi) - 
            np.cos(lam)*np.cos(delta)*np.cos(2*i)*np.cos(phi_source_rec - phi) +
            0.5*np.cos(lam)*np.sin(delta)*np.sin(2*i)*np.sin(2*(phi_source_rec - phi)) - 
            0.5*np.sin(lam)*np.sin(2*delta)*np.sin(2*i)*(1 + np.sin(phi_source_rec - phi)**2))

def _Rsh(phi, delta, lam, i, phi_source_rec):

    """
    Funkcija racuna teoretski Rsh za dane velicine.

    Parameters

    ------------

    phi: float
        Strike
    delta: float
        Dip
    lam: float
        Rake
    i: array
        Kutevi emergencije S vala
    phi_source_rec: array
        Azimuti potresa i kanala.

    -----------

    Returns
    Rsh, array
    """
    return (np.cos(lam)*np.cos(delta)*np.cos(i)*np.sin(phi_source_rec - phi) +
            np.cos(lam)*np.sin(delta)*np.sin(i)*np.cos(2*(phi_source_rec - phi)) +
            np.sin(lam)*np.cos(2*delta)*np.cos(i)*np.cos(phi_source_rec - phi) -
            0.5*np.sin(lam)*np.sin(2*delta)*np.sin(i)*np.sin(2*(phi_source_rec - phi)))


def _coeff(spObs, phi, delta, lam, i_p, i_s, phi_source_rec):

    """
    Funkcija racuna pomocni koeficijent za racun teoretskih omjera amplituda S i P valova. 
    Co = median( abs(sp_ratio_observed / ( sqrt(Rsv**2 + Rsh**2)/Rp) ) ).

    Parameters

    -------------
    spObs: array
        Omjeri amplituda S i P vala za sve kanale
    phi: int / float
        strike
    delta: int / float
        dip
    lam: int / float
        rake
    i_p: array
        Kutevi emergencije za P val
    i_s: array
        Kutevi emergencije za S val
    phi_source_rec: array
        Azimuti potresa i kanala.

    ------------

    Returns
    coeff: float
    """

    return np.median(np.abs( spObs / ( np.sqrt(_Rsv(phi, delta, lam, i_s, phi_source_rec)**2 + _Rsh(phi, delta, lam, i_s, phi_source_rec)**2) / _Rp(phi, delta, lam, i_p, phi_source_rec) ) ) )

def _calc_SP(spObs, phi, delta, lam, i_p, i_s, phi_source_rec):

    """
    Funkcija racuna teoretski omjer amplituda S i P vala prema formuli Co * abs( sqrt(Rsv**2 + Rsh**2) / Rp), gdje je Co 
    dan formulom median( abs(sp_ratio_observed / ( sqrt(Rsv**2 + Rsh**2)/Rp) ) ) prema izvornom radu.

    Parameters

    -------------
    spObs: array
        Omjeri amplituda S i P vala za sve kanale
    phi: int / float
        strike
    delta: int / float
        dip
    lam: int / float
        rake
    i_p: array
        Kutevi emergencije za P val
    i_s: array
        Kutevi emergencije za S val
    phi_source_rec: array
        Azimuti potresa i kanala.

    ------------

    Returns
    Array, teoretski omjeri amplituda S i P vala.


    """
    phi = np.deg2rad(phi)
    delta = np.deg2rad(delta)
    lam = np.deg2rad(lam)
    i_p = np.deg2rad(i_p)
    i_s = np.deg2rad(i_s)
    phi_source_rec = np.deg2rad(phi_source_rec)
    
    co = _coeff(spObs=spObs, phi=phi, delta=delta, lam=lam, i_p=i_p, i_s=i_s, phi_source_rec=phi_source_rec)
    return ( co * np.abs(np.sqrt(_Rsv(phi, delta, lam, i_s, phi_source_rec)**2 + _Rsh(phi, delta, lam, i_s, phi_source_rec)**2) / _Rp(phi, delta, lam, i_p, phi_source_rec)) )


def _get_l1_norms(row, spObs, i_p, i_s, phi_source_rec, weights):

    """
    Funkcija racuna L1 normu prema formuli sum( abs( log10(sp_ratio_calculated / sp_ratio_observed) * weights ) ).
    Weights su tezina za svaki od kanala. Vraca tuple(norma, strike, dip, rake)

    Parameters

    ----------------

    row: tuple(phi, delta, lam)
    spObs: array
        Omjeri amplituda S i P vala za sve kanale
    i_p: array
        Kutevi emergencije za P val
    i_s: array
        Kutevi emergencije za S val
    phi_source_rec: array
        Azimuti potresa i kanala.
    weights: array
        Tezine za svaki kanal.

    -----------------

    Returns
    tuple(l1 norma, strike, dip, rake)
    
    
    """
    phi, delta, lam = row
    calc = _calc_SP(spObs=spObs, phi=phi, delta=delta, lam=lam,
            i_p=i_p, i_s=i_s, phi_source_rec=phi_source_rec)
    l1_norm = np.sum(np.abs(np.log10(calc/spObs)*weights))

    return (l1_norm, phi, delta, lam)


def source_mech(files, dev, output_folder, ignorePicksStartSec=5, ignorePicksEndSec=10, workers=-1, topN_median=30, rakeRangeMin=-90, rakeRangeMax=90):

    """
    Funkcija racuna zarisni mehanizam na temelju omjera amplituda P i S valova zabiljezenih 
    na DAS-u prema Funabiki, Y., & Miyazawa, M. (2025). Rezultate (strike, dip, rake) zapisuje
    u csv formatu u output folder zajedno s vizualiziranim beachball-om.



    Parameters

    ---------------------------------

    files: list
        Lista s putevima do datoteka (str)
    dev: str
        'febus' ili 'sintela'
    output_folder: str
        Put do output foldera
    ignorePicksStarSec: int
        Ne uzimaju se u obzir pickovi unutar toliko sekundi od pocetka zapisa
    ignorePicksEndSec: int
        Ne uzimaju se u obzir pickovi unutar toliko sekundi do kraja zapisa
    workers: int
        Broj procesora
    topN_median: int
        Koliko najboljih rjesenja uzeti za racunanje medijana
    rakeRangeMin: float
        Donja granica za pretrazivanje rake-a
    rakeRangeMax: float
        Gornja granica za pretrazivanje rake-a


    """


    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



    # Datoteka s izracunatim prvim teoretskim nastupima P i S vala, kutevima emergencije, udaljenosti svakog kanala od izvora u km te azimuta izvora i 
    # dijela kabela
    arrivals = f'{SCRIPT_DIR}/first_arrivals.pkl'

    # Definiranje intervala suma i signala, uzima se 0.5 s prije nailaska P vala (pickanog) kao sum, a 0.5 s nakon pickanog nailaska S vala kao signal
    nSecBeforeP = 0.5
    nSecAfterS = 0.5

    with open(arrivals, 'rb') as f:
        first_arrivals = pickle.load(f)

    # Ucitavanje podataka
    if dev == 'febus':
        data = xdas.open_mfdataarray(files, dim='time', tolerance=np.timedelta64(30, "ms"), engine='febus').to_xarray()
    elif dev == 'sintela':
        data = xdas.open_mfdataarray(files, dim='time', tolerance=np.timedelta64(30, "ms"), engine='sintela').to_xarray()
    else:
        raise Exception('Uredjaj!!!')

    # Delta t i frekvencija uzorkovanja DAS zapisa
    delta_t = pd.to_timedelta(data.time[1].data - data.time[0].data).total_seconds()
    fs = int(round(1/delta_t))

    if len(files) > 1:
        base, ext = os.path.splitext(os.path.basename(files[0]))
        base += '_extended'
    else:
        base, ext = os.path.splitext(os.path.basename(files[0]))

    # Ucitavanje pickova PhaseNetDAS-a
    picks = pd.read_csv(f'{SCRIPT_DIR}/results/picks_phasenet_das/' + base + '.csv')

    # Datoteka s pickovima moze imati vise pickova P i S valova za svaki kanal, aftershockovi, druge faze i sl. Ovo osigurava da se uzimaju prvi pickovi P i 
    # S valova za svaki kanal
    picks_P = picks[picks.phase_type == 'P']
    picks_P = picks_P.loc[picks_P.groupby('channel_index')['phase_index'].idxmin()].reset_index(drop=True)
    picks_P = picks_P[picks_P.channel_index < data.shape[1]]
    picks_P = picks_P.sort_values('channel_index').reset_index(drop=True)

    picks_S = picks[picks.phase_type == 'S']
    picks_S = picks_S.loc[picks_S.groupby('channel_index')['phase_index'].idxmin()].reset_index(drop=True)
    picks_S = picks_S[picks_S.channel_index < data.shape[1]]
    picks_S = picks_S.sort_values('channel_index').reset_index(drop=True)

    # Odbacuju se kanali s pickovima P ili S vala unutar definiranog broja sekundi nakon pocetka zapisa i prije kraja zapisa
    edgeSamplesStart = int(ignorePicksStartSec*fs)
    edgeSamplesEnd = int(ignorePicksEndSec*fs)
    nTimeSamples = data.shape[0]

    badChannelsP = picks_P[
        (picks_P.phase_index < edgeSamplesStart) | (picks_P.phase_index > nTimeSamples - edgeSamplesEnd)
    ].channel_index.unique()

    badChannelsS = picks_S[
        (picks_S.phase_index < edgeSamplesStart) | (picks_S.phase_index > nTimeSamples - edgeSamplesEnd)
    ].channel_index.unique()

    badChannels = np.union1d(badChannelsP, badChannelsS)

    picks_P = picks_P[~picks_P.channel_index.isin(badChannels)].reset_index(drop=True)
    picks_S = picks_S[~picks_S.channel_index.isin(badChannels)].reset_index(drop=True)

    # Uzimaju se samo kanali za koje postoje pickova i P i S vala
    common_channels = set(picks_P.channel_index).intersection(picks_S.channel_index)
    picks_P = picks_P[picks_P.channel_index.isin(common_channels)].reset_index(drop=True)
    picks_S = picks_S[picks_S.channel_index.isin(common_channels)].reset_index(drop=True)
    
    # Provodi se metoda za odredjivanje "cistih" kanala po izvornom radu, onih bez skokova u fazi. Razmatraju se samo kanali koji su prosli prethodnu
    #  analizu i za koje postoje pickovi i P i S vala. Prema izvornom radu, za odredjivanje skokova u fazi koristi se zapis od 0.5 s prije do 2.0 s 
    # nakon nailaska S vala za svaki kanal. Za vise informacija pogledati funkciju _get_clean_channels().
    cleanChannelsS, picks_S_clean = _get_clean_channels(data, phase_picks=picks_S, nSecBefore=0.5, nSecAfter=2.0)

    # Uzimaju se nailasci P vala samo za kanala koji su oznaceni kao "cisti"
    picks_P_sel = picks_P[picks_P.channel_index.isin(picks_S_clean.channel_index)]

    dataCleanChannelsS = data.T[cleanChannelsS]

    # U svrhu poboljsanja omjera signala i suma zbraja se 10 uzastopnih kanala na DAS-u
    dataCleanChannelsSStacked = (
        dataCleanChannelsS
        .coarsen(distance=10, boundary='trim')
        .mean('distance')
    )
    del dataCleanChannelsS

    # Zapis na DAS-u se filtrira od bandpass filterom od 2 do 10 Hz
    dataCleanChannelsSStackedFiltered = _bandpass_filter(data=dataCleanChannelsSStacked.data, fs=fs, lowcut=2.0, highcut=10.0, axis=1, order=2)
    del dataCleanChannelsSStacked

    # Racuna se srednje vrijeme nailaska P i S vala za 10 uzastopnih kanala
    picks_S_clean_stacked = np.array(picks_S_clean.phase_index)[:len(picks_S_clean)//10 * 10]
    picks_S_clean_stacked = picks_S_clean_stacked.reshape(-1, 10).mean(axis=-1)
    picks_S_clean_stacked = picks_S_clean_stacked.round().astype(int)

    picks_P_sel_stacked = np.array(picks_P_sel.phase_index)[:len(picks_P_sel)//10 * 10]
    picks_P_sel_stacked = picks_P_sel_stacked.reshape(-1, 10).mean(axis=-1)
    picks_P_sel_stacked = picks_P_sel_stacked.round().astype(int)

    # Uzima se sum i signal za svaki od kanala. Sum je prije definiran broj sekundi prije nailaska P vala dok je signal
    # prije definiran broj sekundi nakon nailaska S vala
    noise_ind = picks_P_sel_stacked[:, None] + np.arange(int(-nSecBeforeP*fs), 0, 1)
    signal_ind = picks_S_clean_stacked[:, None] + np.arange(1, int(nSecAfterS*fs) + 1, 1)

    noiseData = dataCleanChannelsSStackedFiltered[np.arange(dataCleanChannelsSStackedFiltered.shape[0])[:, None], noise_ind]
    signalData = dataCleanChannelsSStackedFiltered[np.arange(dataCleanChannelsSStackedFiltered.shape[0])[:, None], signal_ind]


    # Racuna se RMS za signal i sum
    signalData = signalData - np.mean(signalData, axis=1, keepdims=True)
    noiseData = noiseData - np.mean(noiseData, axis=1, keepdims=True)

    signalRMS = np.sqrt(np.mean(signalData**2, axis=1))
    noiseRMS = np.sqrt(np.mean(noiseData**2, axis=1))

    # Racuna se omjer signala i suma za svaki od promatranih kanala te se uzimaju samo kanali gdje je omjer signala
    # i suma veci od 3, prema izvornom radu
    snr = signalRMS / noiseRMS
    goodChannels = np.where(snr >= 3)[0]

    dataCleanChannelsSStackedFilteredGoodChannels = dataCleanChannelsSStackedFiltered[goodChannels]
    del dataCleanChannelsSStackedFiltered


    # Racuna se deformacija integracijom zabiljezene promjene deformacije u vremenu
    strain = cumulative_trapezoid(dataCleanChannelsSStackedFilteredGoodChannels, dx=delta_t, axis=1, initial=0)
    strain -= np.mean(strain, axis=1, keepdims=True)
    del dataCleanChannelsSStackedFilteredGoodChannels

    # Kao i za zapis, za pickove su uzeti samo kanali koji imaju snr => 3
    picks_P_sel_stacked = picks_P_sel_stacked[goodChannels]
    picks_S_clean_stacked = picks_S_clean_stacked[goodChannels]

    # Racunanje sredine izmedju picka P i S vala za svaki kanal, potrebno za definiranje prozora
    midpoint_P_S = ((picks_P_sel_stacked + picks_S_clean_stacked) / 2).round().astype(int)

    # Prozor oko P vala definiran od picka nailaska P vala do sredine izmedju nailazaka P vala i nailaska S vala za svaki kanal
    pWindowInd = [np.arange(start, end) for start, end in zip(picks_P_sel_stacked, midpoint_P_S)]

    # Duljina prozora oko S vala definirana kao duljina prozora za P val pomnozena s faktorom sqrt(3) za svaki od kanala
    sWindowLength = (np.array([len(window) for window in pWindowInd]) * np.sqrt(3)).round().astype(int)

    # Prozor oko S vala definiran od picka nailaska S vala, duljine jednake prije definiranom prozoru za S val
    sWindowInd = [np.arange(start, end) for start, end in zip(picks_S_clean_stacked, picks_S_clean_stacked + sWindowLength)]

    # Uzimanje zapisa deformacije za prozor P i S vala. Prozor pomaknut za 0.1 s unaprijed, kao u izvornom radu.
    strain_P_window = [strain[ch, np.clip(winInd-int(0.1*fs), 0, nTimeSamples-1)] for ch, winInd in enumerate(pWindowInd)]

    strain_S_window = [strain[ch, np.clip(winInd-int(0.1*fs), 0, nTimeSamples-1)] for ch, winInd in enumerate(sWindowInd)]


    # Racunanje omjera amplituda S i P vala kao omjer maksimalne apsolutne amplitude deformacije u prozoru S vala i prozoru P vala za
    # svaki od kanala
    spAmpRatio = np.array([np.max(np.abs(sWin)) / np.max(np.abs(pWin)) for sWin, pWin in zip(strain_S_window, strain_P_window)])

    
    # Udaljenost svakog kanala od pocetka kabela
    Ln = data.distance.data / 1000

    # Udaljenosti svakog kanala od potresa
    rn = np.array(first_arrivals['distance_from_source'])

    # Racunanje srednjih vrijednosti 10 uzastopnih kanala jer su i prije usrednjene sve vrijednosti za 10 uzastopnih kanala
    rn_stacked = rn[:len(rn)//10 * 10]
    rn_stacked = rn_stacked.reshape(-1, 10).mean(axis=-1)
    rn_stacked_sel = rn_stacked[goodChannels]

    Ln_stacked = Ln[:len(Ln)//10 * 10]
    Ln_stacked = Ln_stacked.reshape(-1, 10).mean(axis=-1)
    Ln_stacked_sel = Ln_stacked[goodChannels]


    # Definirane tezine za svaki od kanala, prema izvornom radu. 0.927 dolazi od atenuacije signala prema telekomunikacijskom standardu.
    weights = np.pow(0.972, Ln_stacked_sel) / rn_stacked_sel


    # Kutevi emergencije i azimuti od izvora do dijela kabela.
    takeoff_angles_p = np.array(first_arrivals['takeoff_angle_p'])
    takeoff_angles_s = np.array(first_arrivals['takeoff_angle_s'])
    src_rec_azimuth = np.array(first_arrivals['src_rec_azimuth'])

    takeoff_angles_p_stacked = takeoff_angles_p[:len(takeoff_angles_p)//10 * 10]
    takeoff_angles_p_stacked = takeoff_angles_p_stacked.reshape(-1, 10).mean(axis=-1)
    takeoff_angles_p_stacked_sel = takeoff_angles_p_stacked[goodChannels]

    takeoff_angles_s_stacked = takeoff_angles_s[:len(takeoff_angles_s)//10 * 10]
    takeoff_angles_s_stacked = takeoff_angles_s_stacked.reshape(-1, 10).mean(axis=-1)
    takeoff_angles_s_stacked_sel = takeoff_angles_s_stacked[goodChannels]

    src_rec_azimuth_stacked = src_rec_azimuth[:len(src_rec_azimuth)//10 * 10]
    src_rec_azimuth_stacked = src_rec_azimuth_stacked.reshape(-1, 10).mean(axis=-1)
    src_rec_azimuth_stacked_sel = src_rec_azimuth_stacked[goodChannels]

    # Raspon vrijednosti za grid search. Rake odredjen od -90 do 90 jer metoda gleda samo omjere apsolutnih vrijednosti amplituda deformacije,
    # nema informacija o polaritetu.
    phi_range = np.arange(0, 360)
    delta_range = np.arange(0, 91)
    lam_range = np.arange(-rakeRangeMin, rakeRangeMax + 1)

    phi_grid, delta_grid, lam_grid = np.meshgrid(phi_range, delta_range, lam_range, indexing='ij')

    grid_array = np.stack([phi_grid.ravel(), delta_grid.ravel(), lam_grid.ravel()], axis=1)

    del phi_grid, delta_grid, lam_grid

    # Racuna L1 normu logaritma omjera zabiljezenih i teoretskih omjera amplituda S i P valova. Koristena funkcija vraca L1 normu i
    # strike, dip i rake za svaku kombinaciju
    print('Performing grid search.......')
    results = Parallel(n_jobs=workers, verbose=5, backend='loky')(
        delayed(_get_l1_norms)(row=row, spObs=spAmpRatio, i_p=takeoff_angles_p_stacked_sel, i_s=takeoff_angles_s_stacked_sel,
        phi_source_rec=src_rec_azimuth_stacked_sel, weights=weights) for row in grid_array)


    # Rezultati se sortiraju po L1 normi, zapisuje se najbolji rezultat (najmanja norma) te medijan top n rjesenja.
    results = sorted(results, key=lambda x: x[0])

    l1_norm_min, phi_l1_min, delta_l1_min, lam_l1_min = results[0]

    l1_norm_top_n_median, phi_l1_top_n_median, delta_l1_top_n_median, lam_l1_top_n_median = np.median(results[:topN_median], axis=0)

    
    os.makedirs(output_folder, exist_ok=True)

    # Zapisuju se najbolji rezultati i medijan top n rezultata u output direktorij u csv formatu zajedno s beachball-om


    print('Beachball min')
    b = beachball([phi_l1_min, delta_l1_min, lam_l1_min], size=200, facecolor='k')
    b.savefig(str(output_folder) + '/' + base + '_mech_min.png', bbox_inches='tight')
    plt.show()

    print(f'Beachball top {topN_median} median')
    b2 = beachball([phi_l1_top_n_median, delta_l1_top_n_median, lam_l1_top_n_median], size=200, facecolor='k')
    b2.savefig(str(output_folder) + '/' + base + f'_mech_top_{topN_median}_median.png', bbox_inches='tight')
    plt.show()


    phi_l1_min_2, delta_l1_min_2, lam_l1_min_2 = aux_plane(phi_l1_min, delta_l1_min, lam_l1_min)
    phi_l1_top_n_median_2, delta_l1_top_n_median_2, lam_l1_top_n_median_2 = aux_plane(phi_l1_top_n_median, delta_l1_top_n_median, lam_l1_top_n_median)

    print('phi delta lambda min')
    print(phi_l1_min, delta_l1_min, lam_l1_min)
    print(phi_l1_min_2, delta_l1_min_2, lam_l1_min_2)
    print(f'phi delta lambda top {topN_median} median')
    print(phi_l1_top_n_median, delta_l1_top_n_median, lam_l1_top_n_median)
    print(phi_l1_top_n_median_2, delta_l1_top_n_median_2, lam_l1_top_n_median_2)

    data_to_write = {
        'phi1_min': [phi_l1_min],
        'delta1_min': [delta_l1_min],
        'lambda1_min': [lam_l1_min],
        'phi2_min': [phi_l1_min_2],
        'delta2_min': [delta_l1_min_2],
        'lambda2_min': [lam_l1_min_2],
        f'phi1_top_{topN_median}_median': [phi_l1_top_n_median],
        f'delta1_top_{topN_median}_median': [delta_l1_top_n_median],
        f'lambda1_top_{topN_median}_median': [lam_l1_top_n_median],
        f'phi2_top_{topN_median}_median': [phi_l1_top_n_median_2],
        f'delta2_top_{topN_median}_median': [delta_l1_top_n_median_2],
        f'lambda2_top_{topN_median}_median': [lam_l1_top_n_median_2]

    }

    df = pd.DataFrame(data_to_write)
    df.to_csv(f'{output_folder}/{base}_mech.csv', index=False)