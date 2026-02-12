import pandas as pd
import pickle
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel
import os
from joblib import Parallel, delayed




def _compute_arrivals(i, lon, lat, eqlon, eqlat, eqdepth, dist_deg, model):

    """
    Pomocna funkcija za racunanje prvih nailaska P i S valova te azimuta izvora
    i dijela kabela.

    Vraca index kanala i dict s teoretskim vremenom nailaska P i S vala, 
    kutom emergencije, udaljenosti dijela kabela od izvora u km te azimut
    izvora i dijela kabela.

    Parameters

    ----------------------

    i: int
        Indeks kanala
    lon: float
        Longituda kanala
    lat: float
        Latituda kanala
    eqlon: float
        Longituda potresa
    eqlat: float
        Latituda potresa
    eqdepth: float
        Dubina potresa u km
    dist_deg: list
        Lista s udaljenostima svakog kanala od potresa u stupnjevima
    model: TauPyModel
        TauPyModel

    ----------------------

    Returns
    channel_index, dict{'time_p', time_s', 'takeoff_angle_p', 'takeoff_angle_s',
    'distance_from_source', 'src_rec_azimuth'}
    """

    arrivals = model.get_ray_paths(
        source_depth_in_km=eqdepth,
        distance_in_degree=dist_deg[i],
        phase_list=['p', 'Pg', 'P', 's', 'Sg', 'S'],
        )
        
    first_p = next(ray for ray in arrivals if ray.name.upper().startswith('P'))
    first_s = next(ray for ray in arrivals if ray.name.upper().startswith('S'))

    distance_in_m, azimuth, back_azimuth = gps2dist_azimuth(lat1=eqlat, lon1=eqlon,
                                                            lat2=lat, lon2=lon)

    return i, {'time_p': first_p.time,
            'time_s': first_s.time,
            'takeoff_angle_p': first_p.takeoff_angle,
            'takeoff_angle_s': first_s.takeoff_angle,
            'distance_from_source': distance_in_m / 1000,
            'src_rec_azimuth': azimuth}






def get_takeoff_angles_azimuth(dev, eqlon, eqlat, eqdepth, input_model, nChannelsFebus=4000, workers=-1):


    """
    Funkcija koristi obspy za racunanje azimuta izmedju izvora i dijela kabela, udaljenost
    od izvora do dijela kabela i kut emergencije te ih zapisuje u pkl datoteku. Takodjer zapisuje
    teoretska vremena nailaska P i S vala.

    Parameters

    -------------------------

    dev: str
        'febus' ili 'sintela'
    eqlon: float
        Longituda potresa
    eqlat: float
        Latituda potresa
    eqdepth: float
        Dubina potresa u km
    input_model: str
        Naizv ili put od TauPy modela
    nChannelsFebus: int
        Broj kanala koji se koristi u slucaju febus uredjaja. Datoteka s koordinatama sadrzi
        oko 80 km kabela, DAS ne koristi sve
    workers: int
        Broj procesora
    """




    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


    if dev == 'febus':

        kanali = pd.read_csv(f'{SCRIPT_DIR}/febus_channels_elevation_200126.csv')
        kanali = kanali[kanali.channel <= nChannelsFebus]

    elif dev == 'sintela':
        
        kanali = pd.read_csv(f'{SCRIPT_DIR}/sintela_channels.csv')

    else:
        raise Exception('Uredjaj!!!!!')


    dist_deg = locations2degrees(lat1=eqlat, long1=eqlon, 
                                lat2=kanali.latitude, long2=kanali.longitude)

    model = TauPyModel(input_model, verbose=True)


    first_arrivals = {'time_p': [], 'time_s': [], 'takeoff_angle_p': [], 'takeoff_angle_s': [],
                     'distance_from_source': [], 'src_rec_azimuth': []}


    print('Calculating theoretical arrivals.........')

    results = Parallel(n_jobs=workers, verbose=5, backend='loky')(
        delayed(_compute_arrivals)(i=i, lon=lon, lat=lat, eqlon=eqlon, eqlat=eqlat, eqdepth=eqdepth, dist_deg=dist_deg, model=model)
        for i, (lon, lat) in enumerate(zip(kanali.longitude, kanali.latitude))
    )

    results.sort(key=lambda x: x[0])
    results = [r for _, r in results]

    for res in results:
        for key in first_arrivals:
            first_arrivals[key].append(res[key])

    with open(f'{SCRIPT_DIR}/first_arrivals.pkl', 'wb') as f:
        pickle.dump(first_arrivals, f)

    print('Done!')