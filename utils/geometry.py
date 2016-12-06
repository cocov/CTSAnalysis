import numpy as np
from ctapipe.io.camera import CameraGeometry
from ctapipe.io.camera import find_neighbor_pixels
from astropy import units as u

def generate_geometry(cts,availableBoard=None):
    '''
    Generate the SST-1M geometry from the CTS configuration
    :param cts: a CTS instance
    :param availableBoard:  which board per sector are available (dict)
    :return: the geometry for visualisation and the list of "good" pixels
    '''
    pix_x = []
    pix_y = []
    pix_id = []
    pix_goodid = []
    for pix in cts.camera.Pixels:
        if  not availableBoard or pix.fadc in availableBoard[pix.sector]:
            pix_x.append(pix.center[0])
            pix_y.append(pix.center[1])
            pix_id.append(pix.ID)
            pix_goodid.append(True)
        else :
            pix_x.append(pix.center[0])
            pix_y.append(pix.center[1])
            pix_id.append(pix.ID)
            pix_goodid.append(False)

    pix_goodid= np.array(pix_goodid)
    neighbors_pix = find_neighbor_pixels(pix_x, pix_y, 30.)
    geom = CameraGeometry(0, pix_id, pix_x * u.mm, pix_y * u.mm, np.ones((1296)) * 400., neighbors_pix, 'hexagonal')
    return geom, pix_goodid


def update_pixels_quality(badid,pix_goodid):
    '''
    Update the pixel quality array
    :param badid: list of bad pixels
    :param pix_goodid: pixel quality array
    :return:
    '''
    for pix in badid:
        pix_goodid[pix]=False