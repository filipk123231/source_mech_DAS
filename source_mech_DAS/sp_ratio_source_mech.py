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

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = butter(order, [low, high], btype='bandpass', output='sos')

    filtered = sosfiltfilt(sos, data, axis=axis)

    return filtered



def _longest_half_cycle(x, fs):
    zero_crossings = np.where(np.diff(np.signbit(x)))[0]
    if len(zero_crossings) < 2:
        return 0
    half_cycles = np.diff(zero_crossings) / fs
    return np.max(half_cycles)


def _get_clean_channels(data, phase_picks, nSecBefore, nSecAfter):

    delta_t = pd.to_timedelta(data.time[1].data - data.time[0].data).total_seconds()
    fs = int(round(1/delta_t))
    
    channelsWithPicks = phase_picks.channel_index.to_numpy()
    picks = phase_picks.phase_index.to_numpy()

    times = picks[:, None] + np.arange(int(-nSecBefore*fs), int(nSecAfter*fs))
    sel_data = data.data[times, channelsWithPicks[:, None]]
    nChannelsWithPicks = sel_data.shape[0]

    positive_skips = []
    negative_skips = []


    for ch_idx in range(nChannelsWithPicks):

        chdata = sel_data[ch_idx]
        phi = np.angle(hilbert(chdata))
        
        deltaPhi = phi[1:] - phi[:-1]

        deltaPhi = (deltaPhi + np.pi) % (2 * np.pi) - np.pi

        eps = 1e-3
        
        positive_times = times[ch_idx][1:][deltaPhi >= (np.pi - eps)]
        negative_times = times[ch_idx][1:][deltaPhi <= (-np.pi + eps)]

        positive_skips.append(positive_times)
        negative_skips.append(negative_times)

    candidateChannels = np.array([(len(positive_skips[ch]) > 0) and (len(negative_skips[ch]) > 0) for ch in range(nChannelsWithPicks)])

    cleanChannelsGlob = np.zeros(data.shape[1], dtype=bool)
    cleanLocal = np.where(~candidateChannels)[0]
    cleanChannelsGlob[channelsWithPicks[cleanLocal]] = True
    
    for ch_idx in np.where(candidateChannels)[0]:

        dt_min = np.min(np.abs(positive_skips[ch_idx][:, None] - negative_skips[ch_idx][None, :])) / fs

        global_ch = channelsWithPicks[ch_idx]
        global_clean = channelsWithPicks[cleanLocal]

        nearest_clean = global_clean[np.argmin(np.abs(global_clean - global_ch))]
        nearest_idx = np.where(channelsWithPicks == nearest_clean)[0][0]

        ref_data = sel_data[nearest_idx]
        T_half_max = _longest_half_cycle(ref_data, fs=fs)

        if dt_min < T_half_max:
            cleanChannelsGlob[global_ch] = True

    phase_picks_clean = phase_picks[phase_picks.channel_index.isin(np.where(cleanChannelsGlob)[0])]
    
    return cleanChannelsGlob, phase_picks_clean



def _Rp(phi, delta, lam, i, phi_source_rec):
    return (np.cos(lam)*np.sin(delta)*np.sin(i)**2*np.sin(2*(phi_source_rec - phi)) - 
            np.cos(lam)*np.cos(delta)*np.sin(2*i)*np.cos(phi_source_rec - phi) +
            np.sin(lam)*np.sin(2*delta)*(np.cos(i)**2 - np.sin(i)**2*np.sin(phi_source_rec - phi)**2) +
            np.sin(lam)*np.cos(2*delta)*np.sin(2*i)*np.sin(phi_source_rec - phi))

def _Rsv(phi, delta, lam, i, phi_source_rec):
    return (np.sin(lam)*np.cos(2*delta)*np.cos(2*i)*np.sin(phi_source_rec - phi) - 
            np.cos(lam)*np.cos(delta)*np.cos(2*i)*np.cos(phi_source_rec - phi) +
            0.5*np.cos(lam)*np.sin(delta)*np.sin(2*i)*np.sin(2*(phi_source_rec - phi)) - 
            0.5*np.sin(lam)*np.sin(2*delta)*np.sin(2*i)*(1 + np.sin(phi_source_rec - phi)**2))

def _Rsh(phi, delta, lam, i, phi_source_rec):
    return (np.cos(lam)*np.cos(delta)*np.cos(i)*np.sin(phi_source_rec - phi) +
            np.cos(lam)*np.sin(delta)*np.sin(i)*np.cos(2*(phi_source_rec - phi)) +
            np.sin(lam)*np.cos(2*delta)*np.cos(i)*np.cos(phi_source_rec - phi) -
            0.5*np.sin(lam)*np.sin(2*delta)*np.sin(i)*np.sin(2*(phi_source_rec - phi)))


def _coeff(spObs, phi, delta, lam, i_p, i_s, phi_source_rec):
    return np.median(np.abs( spObs / ( np.sqrt(_Rsv(phi, delta, lam, i_s, phi_source_rec)**2 + _Rsh(phi, delta, lam, i_s, phi_source_rec)**2) / _Rp(phi, delta, lam, i_p, phi_source_rec) ) ) )

def _calc_SP(spObs, phi, delta, lam, i_p, i_s, phi_source_rec):
    phi = np.deg2rad(phi)
    delta = np.deg2rad(delta)
    lam = np.deg2rad(lam)
    i_p = np.deg2rad(i_p)
    i_s = np.deg2rad(i_s)
    phi_source_rec = np.deg2rad(phi_source_rec)
    
    co = _coeff(spObs=spObs, phi=phi, delta=delta, lam=lam, i_p=i_p, i_s=i_s, phi_source_rec=phi_source_rec)
    return ( co * np.abs(np.sqrt(_Rsv(phi, delta, lam, i_s, phi_source_rec)**2 + _Rsh(phi, delta, lam, i_s, phi_source_rec)**2) / _Rp(phi, delta, lam, i_p, phi_source_rec)) )


def _get_l1_norms(row, spObs, i_p, i_s, phi_source_rec, weights):
    phi, delta, lam = row
    calc = _calc_SP(spObs=spObs, phi=phi, delta=delta, lam=lam,
            i_p=i_p, i_s=i_s, phi_source_rec=phi_source_rec)
    l1_norm = np.sum(np.abs(np.log10(calc/spObs)*weights))

    return (l1_norm, phi, delta, lam)


def source_mech(files, dev, output_folder, ignorePicksStartSec=5, ignorePicksEndSec=10, workers=-1, topN_median=30):

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    arrivals = f'{SCRIPT_DIR}/first_arrivals.pkl'


    nSecBeforeP = 0.5
    nSecAfterS = 0.5

    with open(arrivals, 'rb') as f:
        first_arrivals = pickle.load(f)


    if dev == 'febus':
        data = xdas.open_mfdataarray(files, dim='time', tolerance=np.timedelta64(30, "ms"), engine='febus').to_xarray()
    elif dev == 'sintela':
        data = xdas.open_mfdataarray(files, dim='time', tolerance=np.timedelta64(30, "ms"), engine='sintela').to_xarray()
    else:
        raise Exception('Uredjaj!!!')

    delta_t = pd.to_timedelta(data.time[1].data - data.time[0].data).total_seconds()
    fs = int(round(1/delta_t))

    if len(files) > 1:
        base, ext = os.path.splitext(os.path.basename(files[0]))
        base += '_extended'
    else:
        base, ext = os.path.splitext(os.path.basename(files[0]))

    picks = pd.read_csv(f'{SCRIPT_DIR}/results/picks_phasenet_das/' + base + '.csv')

    picks_P = picks[picks.phase_type == 'P']
    picks_P = picks_P.loc[picks_P.groupby('channel_index')['phase_index'].idxmin()].reset_index(drop=True)
    picks_P = picks_P[picks_P.channel_index < data.shape[1]]
    picks_P = picks_P.sort_values('channel_index').reset_index(drop=True)

    picks_S = picks[picks.phase_type == 'S']
    picks_S = picks_S.loc[picks_S.groupby('channel_index')['phase_index'].idxmin()].reset_index(drop=True)
    picks_S = picks_S[picks_S.channel_index < data.shape[1]]
    picks_S = picks_S.sort_values('channel_index').reset_index(drop=True)


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

    common_channels = set(picks_P.channel_index).intersection(picks_S.channel_index)
    picks_P = picks_P[picks_P.channel_index.isin(common_channels)].reset_index(drop=True)
    picks_S = picks_S[picks_S.channel_index.isin(common_channels)].reset_index(drop=True)

    cleanChannelsS, picks_S_clean = _get_clean_channels(data, phase_picks=picks_S, nSecBefore=0.5, nSecAfter=2.0)

    picks_P_sel = picks_P[picks_P.channel_index.isin(picks_S_clean.channel_index)]

    dataCleanChannelsS = data.T[cleanChannelsS]

    dataCleanChannelsSStacked = (
        dataCleanChannelsS
        .coarsen(distance=10, boundary='trim')
        .mean('distance')
    )
    del dataCleanChannelsS

    dataCleanChannelsSStackedFiltered = _bandpass_filter(data=dataCleanChannelsSStacked.data, fs=fs, lowcut=2.0, highcut=10.0, axis=1, order=2)
    del dataCleanChannelsSStacked

    picks_S_clean_stacked = np.array(picks_S_clean.phase_index)[:len(picks_S_clean)//10 * 10]
    picks_S_clean_stacked = picks_S_clean_stacked.reshape(-1, 10).mean(axis=-1)
    picks_S_clean_stacked = picks_S_clean_stacked.round().astype(int)

    picks_P_sel_stacked = np.array(picks_P_sel.phase_index)[:len(picks_P_sel)//10 * 10]
    picks_P_sel_stacked = picks_P_sel_stacked.reshape(-1, 10).mean(axis=-1)
    picks_P_sel_stacked = picks_P_sel_stacked.round().astype(int)

    noise_ind = picks_P_sel_stacked[:, None] + np.arange(int(-nSecBeforeP*fs), 0, 1)
    signal_ind = picks_S_clean_stacked[:, None] + np.arange(1, int(nSecAfterS*fs) + 1, 1)

    noiseData = dataCleanChannelsSStackedFiltered[np.arange(dataCleanChannelsSStackedFiltered.shape[0])[:, None], noise_ind]
    signalData = dataCleanChannelsSStackedFiltered[np.arange(dataCleanChannelsSStackedFiltered.shape[0])[:, None], signal_ind]

    signalData = signalData - np.mean(signalData, axis=1, keepdims=True)
    noiseData = noiseData - np.mean(noiseData, axis=1, keepdims=True)

    signalRMS = np.sqrt(np.mean(signalData**2, axis=1))
    noiseRMS = np.sqrt(np.mean(noiseData**2, axis=1))

    snr = signalRMS / noiseRMS
    goodChannels = np.where(snr >= 3)[0]

    dataCleanChannelsSStackedFilteredGoodChannels = dataCleanChannelsSStackedFiltered[goodChannels]
    del dataCleanChannelsSStackedFiltered

    strain = cumulative_trapezoid(dataCleanChannelsSStackedFilteredGoodChannels, dx=delta_t, axis=1, initial=0)
    strain -= np.mean(strain, axis=1, keepdims=True)
    del dataCleanChannelsSStackedFilteredGoodChannels

    picks_P_sel_stacked = picks_P_sel_stacked[goodChannels]
    picks_S_clean_stacked = picks_S_clean_stacked[goodChannels]

    midpoint_P_S = ((picks_P_sel_stacked + picks_S_clean_stacked) / 2).round().astype(int)

    pWindowInd = [np.arange(start, end) for start, end in zip(picks_P_sel_stacked, midpoint_P_S)]

    sWindowLength = (np.array([len(window) for window in pWindowInd]) * np.sqrt(3)).round().astype(int)

    sWindowInd = [np.arange(start, end) for start, end in zip(picks_S_clean_stacked, picks_S_clean_stacked + sWindowLength)]

    strain_P_window = [strain[ch, np.clip(winInd-int(0.1*fs), 0, nTimeSamples-1)] for ch, winInd in enumerate(pWindowInd)]

    strain_S_window = [strain[ch, np.clip(winInd-int(0.1*fs), 0, nTimeSamples-1)] for ch, winInd in enumerate(sWindowInd)]

    spAmpRatio = np.array([np.max(np.abs(sWin)) / np.max(np.abs(pWin)) for sWin, pWin in zip(strain_S_window, strain_P_window)])

    
    Ln = data.distance.data / 1000
    rn = np.array(first_arrivals['distance_from_source'])

    rn_stacked = rn[:len(rn)//10 * 10]
    rn_stacked = rn_stacked.reshape(-1, 10).mean(axis=-1)
    rn_stacked_sel = rn_stacked[goodChannels]

    Ln_stacked = Ln[:len(Ln)//10 * 10]
    Ln_stacked = Ln_stacked.reshape(-1, 10).mean(axis=-1)
    Ln_stacked_sel = Ln_stacked[goodChannels]

    weights = np.pow(0.972, Ln_stacked_sel) / rn_stacked_sel

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


    phi_range = np.arange(0, 360)
    delta_range = np.arange(0, 91)
    lam_range = np.arange(-90, 91)

    phi_grid, delta_grid, lam_grid = np.meshgrid(phi_range, delta_range, lam_range, indexing='ij')

    grid_array = np.stack([phi_grid.ravel(), delta_grid.ravel(), lam_grid.ravel()], axis=1)

    del phi_grid, delta_grid, lam_grid

    print('Performing grid search.......')
    results = Parallel(n_jobs=workers, verbose=5, backend='loky')(
        delayed(_get_l1_norms)(row=row, spObs=spAmpRatio, i_p=takeoff_angles_p_stacked_sel, i_s=takeoff_angles_s_stacked_sel,
        phi_source_rec=src_rec_azimuth_stacked_sel, weights=weights) for row in grid_array)


    results = sorted(results, key=lambda x: x[0])

    l1_norm_min, phi_l1_min, delta_l1_min, lam_l1_min = results[0]

    l1_norm_top_n_median, phi_l1_top_n_median, delta_l1_top_n_median, lam_l1_top_n_median = np.median(results[:topN_median], axis=0)

    
    os.makedirs(output_folder, exist_ok=True)


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