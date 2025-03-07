#!/usr/bin/env python
'''
Code to convert from Postel to full-disk Helioprojective cartesian. The header
of the Postel map should include the info of the World Coordinate System (WCS)
and the T_REC keyword (same as the one specified in the filename).
The output is a 4096x4096 map
'''

import sys
# Check number of arguments
if len(sys.argv) != 3:
    sys.tracebacklimit = 0
    raise TypeError('Usage: python postel2los.py INPUT[FITS] OUTPUT')

import numpy as np
from sunpy.map import Map, make_fitswcs_header
from drms import Client
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits


def get_hmi_keywords(t_rec):
    '''
    Given a string date in the format YYYY.mm.dd_HH:MM:SS, extract keywords to
    use them as template in the helioprojective header. It returns a drms query
    response containting metadata info for reprojection
    '''

    # Selects initial and final time. Here I choose Doppler data but it is
    # only used for reading the header
    trecstep = 45  # TRECSTEP for HMI
    qstr = f'hmi.v_{int(trecstep)}s[{t_rec}_TAI]'

    # Get coordinate metadata information
    client = Client()
    query = client.query(qstr, key='T_REC, CRPIX1, CRPIX2, CDELT1, CDELT2,'
                         'CUNIT1, CUNIT2, CRVAL1, CRVAL2, CROTA2, CRLN_OBS,'
                         'CRLT_OBS, TELESCOP, DSUN_OBS, RSUN_OBS, DATE-OBS,'
                         'CAR_ROT, WCSNAME', convert_numeric=True)
    return query


def update_map_header(map_out, keywords):
    '''
    Update the header of map_out with keywords from the Helioprojective
    template.

    Parameters:
    ----------
    map_out : sunpy.map.Map
        Reprojected solar map header to be updated.

    keywords : drms.QueryResponse
        Query response containing metadata keywords.
    '''

    # Update header with relevant keywords
    for key in keywords:
        map_out.meta[key] = keywords[key].values[0]

    # Keyword comments
    map_out.meta['keycomments'] = {
        'CDELT1': '[arcsec/pixel] image scale in the x direction',
        'CDELT2': '[arcsec/pixel] image scale in the y direction',
        'CRPIX1': '[pixel] CRPIX1: location of the Sun center',
        'CRPIX2': '[pixel] CRPIX2: location of the Sun center',
        'CRVAL1': '[arcsec] CRVAL1: x origin',
        'CRVAL2': '[arcsec] CRVAL2: y origin',
        'CRLN_OBS': '[deg] Carrington longitude of the observer',
        'CRLT_OBS': '[deg] Carrington latitude of the observer',
        'CROTA2': '[deg]',
        'CAR_ROT': 'Carrington rotation number of CRLN_OB',
        'DSUN_OBS': '[m] Distance from SDO to Sun center',
        'RSUN_OBS': '[arcsec] angular radius of Sun',
        'DATE-OBS': r'[ISO] Observation date {DATE__OBS}',
    }

    return None


def check_and_change_bitpix(map_input, out_hdu):
    '''
    Change the BITPIX keyword of the output map with BSCALE and BZERO
    according to the equation:

       physical_value = BSCALE*(storage_value) + BZERO.

    If BSCALE is not present it is set to 1. If BZERO is not present it is set
    to 0.
    '''

    nan_array = np.where(np.isnan(out_hdu.data))

    if ('BSCALE' in map_input.meta):
        bscale = map_input.meta['BSCALE']
    else:
        bscale = 1
    if ('BZERO' in map_input.meta):
        bzero = map_input.meta['BZERO']
    else:
        bzero = np.nan

    bitpix = int(map_input.meta['BITPIX'])
    if bitpix <= 0:
        out_hdu.scale(f'float{abs(bitpix)}', bzero=bzero, bscale=bscale)
    else:
        out_hdu.scale(f'int{bitpix}', bzero=bzero, bscale=bscale)

    # Define a blank value and set pixels outside the initial FoV as blanks
    blank = np.int32(-2**(bitpix - 1))  # Not sure for float bitpix
    out_hdu.header['BLANK'] = blank
    out_hdu.data[nan_array] = blank

    return None


def convert_to_heliprojective(map_input, naxis_=4096, output='HP_map.fits'):
    '''
    Read info of Postel map and reprojects it to helioprojective.

    Parameters:
    ----------
    map_input : sunpy.map.Map
        Postel projected map
    '''

    # Get keywords of interest

    t_rec = map_input._date_obs
    if t_rec is None:
        t_rec = map_input.meta['T_REC'][0:19]
    else:
        t_rec = str(t_rec)
    crln_obs = map_input.meta['CRLN_OBS']
    crlt_obs = map_input.meta['CRLT_OBS']
    frame = map_input.coordinate_frame

    # Shift the center to align with the center of the reprojected map
    map_input.meta['CRPIX1'] += 0.5
    map_input.meta['CRPIX2'] += 0.5

    # Get keywords from the Helioprojective map
    keywords = get_hmi_keywords(t_rec)
    naxis1 = naxis_
    naxis2 = naxis_
    crpix1 = float(keywords['CRPIX1'].values[0])
    crpix2 = float(keywords['CRPIX2'].values[0])
    cdelt1 = float(keywords['CDELT1'].values[0])
    cdelt2 = float(keywords['CDELT2'].values[0])
    crota2 = float(keywords['CROTA2'].values[0])

    # Retrieve data of center of map, then convert it to helioprojective
    origin_hgcar = SkyCoord(crln_obs, crlt_obs, unit=u.deg, frame=frame)
    origin = origin_hgcar.helioprojective

    # Create header of a helioprojective projection
    target_header = make_fitswcs_header(
        data=(naxis1, naxis2),  # size of output datacube
        scale=[cdelt1, cdelt2] * u.arcsec/u.pix,
        coordinate=origin,
        rotation_angle=crota2 * u.deg,
        projection_code="TAN",  # TAN: gnomonic projection
        reference_pixel=[crpix1, crpix2] * u.pix
        )

    # Reproject map from Postel into Helioprojective cartesian
    map_out = map_input.reproject_to(target_header, algorithm='exact',
                                     parallel=False)

    # Update header of reprojected map
    update_map_header(map_out, keywords)
    out_hdu = fits.PrimaryHDU(data=map_out.data, header=map_out.fits_header)

    # Change BITPIX of datacube
    check_and_change_bitpix(map_input, out_hdu)

    # Save file
    out_hdu.writeto(output, overwrite=True)


def main(name_input, naxis_=4096, output='HP_map.fits'):
    '''
    Reproject a Postel-projected map into a Helioprojective cartesian

    Parameters:
    ----------
    name_input: str
        Name of the Postel-projected map

    name_output: str
        Name of the reprojected Helioprojective map
    '''

    # Load data
    map_input = Map(name_input, scale_back=True, do_not_scale_image_data=False,
                    uint=False)

    # Convert to helioprojective
    convert_to_heliprojective(map_input, naxis_=naxis_, output=output)

    return None


# -----------------------------------------------------------------------


if __name__ == '__main__':

    # Read input file and assign an output name
    data_input = sys.argv[1]
    data_output = sys.argv[2]

    main(name_input=data_input, output=data_output)
