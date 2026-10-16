import os
import sys
from datetime import datetime, timezone

import astropy.time
import numpy as np
from spacepy import pycdf

# note: this should point to a template .cdf file, not a plaintext skeleton file.
fire_l0_skeleton_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "impax_ifire_skeleton-L0.cdf"
)


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


def cdf_from_500(
    fields, save_path: str, skeleton_path=fire_l0_skeleton_path, increment=0
):
    logical_source = "impax_ifire_l0"
    data_version = "0"
    datestr = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    fname = f"{logical_source}_{datestr}_L{data_version}_{increment}"
    fpath = os.path.join(save_path, fname + ".cdf")

    cdf = pycdf.CDF(fpath, skeleton_path)

    # Epoch
    cdf["Epoch"][...] = fields["ttag"].astype(np.uint64) * 1e7

    # Bins
    cdf["BINS_CH1_RAW"][...] = fields["bins_ch1"]
    cdf["BINS_CH2_RAW"][...] = fields["bins_ch2"]

    cdf["CCSDS_SEQ"][...] = fields["CCSDS_SEQUENCE_COUNT"]
    cdf["CCSDS_LENGTH"][...] = fields["CCSDS_PACKET_LENGTH"]
    hk_vars = [
        "FAULT",
        "CFLAG",
        "NACC",
        "SEU_RAW",
        "AUTO",
        "V5B_RAW",
        "HVI_RAW",
        "DET_TEMP_RAW",
        "CPU_TEMP_RAW",
        "3V3_RAW",
        "SPARE_23",
        "CAL1_RAW",
        "CAL2_RAW",
        "CMD",
        "TSET_RAW",
        "TOT1_RAW",
        "TOT2_RAW",
        "CLOCK",
        "SEQ",
    ]
    for key in hk_vars:
        if key in fields:
            cdf[key][...] = fields[key]

    return cdf


# todo: make this depot path for CDF L0 globally configurable.
def cdf_from_apid(
    apid: int,
    packet: dict,
    src_file: str,
    save_path: str,
):
    # if apid != packet["CCSDS_APID"]:
    #     raise ValueError

    cdf = cdf_from_500(packet, src_file, save_path)
    cdf.close()
    print(f"wrote {cdf.pathname}")
