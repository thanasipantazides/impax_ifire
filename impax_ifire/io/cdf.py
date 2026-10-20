import os
import sys
from datetime import datetime, timezone

import astropy.time
import numpy as np
from spacepy import pycdf

# note: this should point to a template .cdf file, not a plaintext skeleton file.
fire_l0_skeleton_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "impax_ifire_skeleton-just.cdf"
)


def cdf_from_500(
    fields, save_path: str, skeleton_path=fire_l0_skeleton_path, increment=0
):
    """
    Create a CDF with the packet fields data populated.

    Parameters
    ----------
    fields: dict[str, np.ndarray]
        Keys are names of packet fields, and values are np.ndarrays of those fields spanning time/CCSDS packet index.

    save_path: str
        Path to folder where the CDF folder should be stored.

    skeleton_path: str
        Path to the CDF skeleton file. This should have all the global attribute and variable metadata populated already. Should be a real CDF file skeleton, not a text .skt file.

    increment: int
        Counter to put at the end of the filename
    """

    logical_source = "impax_ifire_l0"
    data_version = "0"
    datestr = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    fname = f"{logical_source}_{datestr}_L{data_version}_{increment}"
    fpath = os.path.join(save_path, fname + ".cdf")

    cdf = pycdf.CDF(fpath, skeleton_path)

    # Epoch
    cdf["Epoch"][...] = fields["TTAG"].astype(np.uint64) * 1e7

    # Bins
    cdf["BINS_CH1_RAW"][...] = fields["BINS_CH1"]
    cdf["BINS_CH2_RAW"][...] = fields["BINS_CH2"]

    cdf["CCSDS_SEQ"][...] = fields["CCSDS_SEQUENCE_COUNT"]
    cdf["CCSDS_LENGTH"][...] = fields["CCSDS_PACKET_LENGTH"]
    hk_vars = [
        "FAULT",
        "CFLAG",
        "NBIN",
        "NBIN_HK",
        "NACC",
        "SHIFT",
        "GAIN1",
        "GAIN2",
        "OFF1",
        "OFF2",
        "CF1",
        "CF2",
        "TH1",
        "TH2",
        "SEU_RAW",
        "AUTO",
        "EDGE1",
        "EDGE2",
        "EDGE3",
        "EDGE4",
        "EDGE5",
        "EDGE6",
        "V5B_RAW",
        "HVI_RAW",
        "DET_TEMP_RAW",
        "CPU_TEMP_RAW",
        "V3V3_RAW",
        "SPARE_23",
        "CAL1_RAW",
        "CAL2_RAW",
        "CMD",
        "TSET",
        "TOT1_RAW",
        "TOT2_RAW",
        "CLOCK",
        "SEQ",
    ]

    #
    for key in hk_vars:
        if key in fields:
            cdf[key][...] = fields[key]

    cdf["BINS_INDEX"][...] = range(0, 130, 1)

    return cdf
