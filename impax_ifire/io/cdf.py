import os
import sys
from datetime import datetime, timezone

import astropy.time
import numpy as np
from spacepy import pycdf

# CDF global fields:
#
#   - APID
#   - SRC_FILE
#   - INST
#   - MISS
#   - PI
#   - LEVEL
#   - QUALITY
#   - FILE_VERSION
#   - GENERATED_BY
#   - LINK
#
#  zVariables:
#   - EPOCH
#       - TIME
#       - UNITS
#       - DEPEND_1


def cdf_defaults_hk(save_path: str):
    logical_source = "impax_ifire_l0-hk"
    data_version = "0"
    datestr = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    fname = f"{logical_source}_{datestr}_L{data_version}"
    fpath = os.path.join(save_path, fname + ".cdf")

    cdf = pycdf.CDF(fpath, "", readonly=False)

    cdf.attrs["Data_type"] = "L0-raw>level 0 hk"
    cdf.attrs["Data_version"] = data_version
    cdf.attrs["Descriptor"] = (
        "IFIRE>IMPAX FIREBIRD Instrument for Relativistic Electrons"
    )
    cdf.attrs["Instrument_type"] = ["Housekeeping"]
    cdf.attrs["Logical_source"] = logical_source
    cdf.attrs["Logical_file_id"] = fname
    cdf.attrs["Logical_source_description"] = "IMPAX IFIRE level 0 hk"
    cdf.attrs["Mission_group"] = (
        "Imaging Microburst Precipitation with Atmospheric X-ray emissions cubesat (IMPAX)"
    )
    cdf.attrs["PI_affiliation"] = "University of New Hampshire"
    cdf.attrs["PI_name"] = "Harlan Spence"
    cdf.attrs["Source_name"] = (
        "IMPAX>Imaging Microburst Precipitation with Atmospheric X-ray emissions cubesat"
    )
    cdf.attrs["TEXT"] = ["Raw IFIRE HK telemetry fields decoded."]
    cdf.attrs["Date_Start"] = "1970-01-01T00:00:00"
    cdf.attrs["Date_End"] = "2050-01-01T00:00:00"

    cdf.attrs["IMPAX_Quality"] = "Tenuous and preliminary"

    return cdf


def cdf_defaults_sci(save_path: str):
    logical_source = "impax_ifire_l0-sci"
    data_version = "0"
    datestr = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    fname = f"{logical_source}_{datestr}_L{data_version}"
    fpath = os.path.join(save_path, fname + ".cdf")

    cdf = pycdf.CDF(fpath, "", readonly=False)

    cdf.attrs["Data_type"] = "L0-sci>level 0 sci"
    cdf.attrs["Data_version"] = data_version
    cdf.attrs["Descriptor"] = (
        "IFIRE>IMPAX FIREBIRD Instrument for Relativistic Electrons"
    )
    cdf.attrs["Instrument_type"] = ["Particles (space)"]
    cdf.attrs["Logical_source"] = logical_source
    cdf.attrs["Logical_file_id"] = fname
    cdf.attrs["Logical_source_description"] = "IMPAX IFIRE level 0 sci"
    cdf.attrs["Mission_group"] = (
        "Imaging Microburst Precipitation with Atmospheric X-ray emissions cubesat (IMPAX)"
    )
    cdf.attrs["PI_affiliation"] = "University of New Hampshire"
    cdf.attrs["PI_name"] = "Harlan Spence"
    cdf.attrs["Source_name"] = (
        "IMPAX>Imaging Microburst Precipitation with Atmospheric X-ray emissions cubesat"
    )
    cdf.attrs["TEXT"] = ["Raw IFIRE science telemetry fields decoded."]
    cdf.attrs["Date_Start"] = "1970-01-01T00:00:00"
    cdf.attrs["Date_End"] = "2050-01-01T00:00:00"

    cdf.attrs["IMPAX_Quality"] = "Tenuous and preliminary"

    return cdf


def cdf_make_epoch(fields, cdf):
    # todo: check here for rollover of CCSDS_SEQUENCE_COUNT in here, correct for it.
    # in a way that is tolerant packet loss. But still correct. Good luck.
    # determine ordering of all packets, based on the CCSDS sequence counter.

    # order = fields["CCSDS_SEQUENCE_COUNT"].argsort()
    ct = fields["ttag"]  # [order]

    # ttag lsb is 10 ms.
    # t = ct.astype(np.uint64) / 100 * 1e9  # in ns
    t = ct / 100 * 1e9  # in ns

    epoch = cdf.new("Epoch", type=pycdf.const.CDF_TIME_TT2000, recVary=True)
    epoch[...] = t
    epoch.attrs["MONOTON"] = "INCREASE"
    epoch.attrs["VAR_TYPE"] = "support_data"
    epoch.attrs["LABLAXIS"] = "Epoch"
    epoch.attrs["UNITS"] = "ns"
    epoch.attrs["SCALETYP"] = "linear"
    epoch.attrs["TIME_BASE"] = "J2000"
    epoch.attrs["TIME_SCALE"] = "Terrestrial Time"
    epoch.attrs["DICT_KEY"] = "SPASE>Support>SupportQuantity:Temporal"

    return cdf


def cdf_from_500(fields, src_file: str, save_path: str):
    cdf = cdf_defaults_sci(save_path)
    cdf = cdf_make_epoch(fields, cdf)

    # build energy distribution:
    # - see what nbins is
    # - do something with nan (maybe just default?)
    #
    # nbins == 5: check energy edges.
    #

    # I have error handling to make sure all nbin are identical earlier.
    nbin = fields["nbin"][0]

    mux = fields["mux"]
    hk = fields["hk"]
    # print(np.unique(hk[mux[12:18]]))
    # print(f"{nbin=}")

    if nbin == 5:
        # do a whole song and dance to pull the energy edges out of HK.
        eedges_adc = np.unique(hk[mux[12:18]])
        eedges_kev = eedges_adc / 32.7
    else:
        # can still pull energy-space bin values out of HK
        pass

    # add the ADC bin list
    adc1 = cdf.new("Bins_CH1", type=pycdf.const.CDF_UINT1, recVary=True)
    adc1[...] = range(0, nbin + 2, 1)
    adc1.attrs["VAR_TYPE"] = "support_data"
    adc1.attrs["FIELDNAM"] = "Bin Index"
    adc1.attrs["DEPEND_0"] = "Epoch"
    adc1.attrs["LABLAXIS"] = "Bin Index"
    adc1.attrs["UNITS"] = ""

    adc2 = cdf.new("Bins_CH2", type=pycdf.const.CDF_UINT1, recVary=True)
    adc2[...] = range(0, nbin + 2, 1)
    adc2.attrs["VAR_TYPE"] = "support_data"
    adc2.attrs["FIELDNAM"] = "Bin Index"
    adc2.attrs["DEPEND_0"] = "Epoch"
    adc2.attrs["LABLAXIS"] = "Bin Index"
    adc2.attrs["UNITS"] = ""

    # add the image data, which references the pixel and energy variables
    # bins = cdf.new("Bins", type=pycdf.const.CDF_BYTE, recVary=True)
    cdf["Bins1"] = fields["bins"][:, 0 : nbin + 2]
    cdf["Bins1"].attrs["VAR_TYPE"] = "data"
    cdf["Bins1"].attrs["FIELDNAM"] = "Per-bin accumulated ADC value"
    cdf["Bins1"].attrs["CATDESC"] = (
        "Per-bin accumulated ADC value for nbin bins (and an under and overflow bin), each 16 bits deep."
    )
    # note: this is a 1D list of pixels at each timestep, not yet a 2D image.
    cdf["Bins1"].attrs["UNITS"] = ""
    cdf["Bins1"].attrs["DEPEND_0"] = "Epoch"
    cdf["Bins1"].attrs["DEPEND_1"] = "Bins_CH1"
    # bins.attrs["DEPEND_2"] = "ADC"

    cdf["Bins1"].attrs["LABLAXIS"] = "ADC Count"

    cdf["Bins1"].attrs["DICT_KEY"] = "SPASE>Huh"
    # cdf["Bins"].attrs["DISPLAY_TYPE"] = (
    #     "spectrogram>y=energy,z=Bins(*,1),z=Bins(*,7),z=Bins(*,12)"
    # )

    cdf["Bins1"].attrs["AVG_TYPE"] = "standard"
    cdf["Bins1"].attrs["SCALETYP"] = "linear"

    cdf["Bins2"] = fields["bins"][:, nbin + 2 : :]
    cdf["Bins2"].attrs["VAR_TYPE"] = "data"
    cdf["Bins2"].attrs["FIELDNAM"] = "Per-bin accumulated ADC value"
    cdf["Bins2"].attrs["CATDESC"] = (
        "Per-bin accumulated ADC value for nbin bins (and an under and overflow bin), each 16 bits deep."
    )
    # note: this is a 1D list of pixels at each timestep, not yet a 2D image.
    cdf["Bins2"].attrs["UNITS"] = ""
    cdf["Bins2"].attrs["DEPEND_0"] = "Epoch"
    cdf["Bins2"].attrs["DEPEND_1"] = "Bins_CH2"
    # bins.attrs["DEPEND_2"] = "ADC"

    cdf["Bins2"].attrs["LABLAXIS"] = "ADC Count"

    cdf["Bins2"].attrs["DICT_KEY"] = "SPASE>Huh"
    # cdf["Bins"].attrs["DISPLAY_TYPE"] = (
    #     "spectrogram>y=energy,z=Bins(*,1),z=Bins(*,7),z=Bins(*,12)"
    # )

    cdf["Bins2"].attrs["AVG_TYPE"] = "standard"
    cdf["Bins2"].attrs["SCALETYP"] = "linear"

    return cdf


# todo: make this depot path for CDF L0 globally configurable.
def cdf_from_apid(
    apid: int,
    packet: dict,
    src_file: str,
    save_path: str = os.path.join(os.path.dirname(__file__), "..", "data", "log"),
):
    # if apid != packet["CCSDS_APID"]:
    #     raise ValueError

    cdf = cdf_from_500(packet, src_file, save_path)
    cdf.close()
    print(f"wrote {cdf.pathname}")
