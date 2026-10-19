"""
This module provides a generic file reader.
"""

import csv
import os
import sys
from enum import Enum
from multiprocessing import Value

import numpy as np

from ..io.cdf import cdf_from_500
from ..util.exceptions import IFireNBinChangedError

__all__ = ["read_file"]

demux_file_path = os.path.join(
    os.path.dirname(__file__), "..", "data", "ifire-hk-demux-rules.csv"
)

l0_depot = os.path.join(os.path.dirname(__file__), "..", "data", "log")


def sparky(arr, end="\n"):
    tm = np.iinfo(arr.ravel()[0]).max
    map = {0: ".", 1: "o", tm: "F", tm - 1: "f"}
    if len(arr.shape) == 1:
        for k in arr:
            print(map.get(k, "O"), end=end)
    if len(arr.shape) == 2:
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T
        for r in arr:
            line = "".join([map.get(c, "O") for c in r])
            print(line, end=end)


class FireParams(Enum):
    NBIN_MIN = 4
    NBIN_MAX = 128
    TOTAL_BINS = 130
    # total number of HK MUX elements. Fields that share a uint16 are only counted once.
    TOTAL_MUX = 32
    VALID_APID = 0x500
    # BIN_PAD = 0xFFFF - 1
    FILLVAL = 0xFFFF


def ccsds_version_number(k: int, buff: bytes):
    """Get CCSDS version number of this packet, assuming k is a the beginning of a packet,"""
    return 7 & (buff[k] >> 5)


def ccsds_packet_type(k: int, buff: bytes):
    return 1 & (buff[k] >> 4)


def ccsds_secondary_flag(k: int, buff: bytes):
    return 1 & (buff[k] >> 3)


def apid(k: int, buff: bytes):
    """Get the APID of this CCSDS packet, assuming k is at the beginning of the packet."""
    id = 0x7FF & int.from_bytes(buff[k : k + 2], byteorder="big", signed=False)
    if id != FireParams.VALID_APID.value:
        raise ValueError
    return id


def sequence_flag(k: int, buff: bytes):
    return 3 & (buff[k + 2] >> 6)


def sequence_count(k: int, buff: bytes):
    return 0x3FF & int.from_bytes(buff[k + 2 : k + 3], byteorder="big", signed=False)


def length(k: int, buff: bytes):
    """Get the length field of this CCSDS packet, assuming k is at the beginning of the packet."""
    return 0xFFFF & int.from_bytes(buff[k + 4 : k + 6], byteorder="big", signed=False)


def pflags(k: int, buff: bytes):
    return 0xFF & buff[k + 6]


def accum(k: int, buff: bytes):
    return 0xFF & buff[k + 7]


def nbin(k: int, buff: bytes):
    nbin = 0xFF & buff[k + 8]

    if nbin < 4 or nbin > 128:
        raise ValueError(f"FIRE {nbin=} out of valid range 4-128.")

    return nbin


def mux(k: int, buff: bytes):
    return 0xFF & buff[k + 9]


def hk(k: int, buff: bytes, mux: int):
    # todo: interpret the hk value via the mux value
    return 0xFFFF & int.from_bytes(
        buff[k + 10 : k + 12], byteorder="little", signed=False
    )


def ttag(k: int, buff: bytes):
    return 0xFFFFFFFF & int.from_bytes(
        buff[k + 12 : k + 16], byteorder="little", signed=False
    )


def base1(k: int, buff: bytes):
    return 0xFFFF & int.from_bytes(
        buff[k + 16 : k + 18], byteorder="little", signed=False
    )


def base2(k: int, buff: bytes):
    return 0xFFFF & int.from_bytes(
        buff[k + 18 : k + 20], byteorder="little", signed=False
    )


def rate1(k: int, buff: bytes):
    return 0xFFFF & int.from_bytes(
        buff[k + 20 : k + 22], byteorder="little", signed=False
    )


def rate2(k: int, buff: bytes):
    return 0xFFFF & int.from_bytes(
        buff[k + 22 : k + 24], byteorder="little", signed=False
    )


def bins(k: int, buff: bytes, pflags: int, nbin: int):

    enach1 = 1 & (pflags >> 7)
    enach2 = 1 & (pflags >> 6)

    nbytes = 2 * (nbin + 2)

    # fill up the bins with NaNs for any channel which is not enabled.
    #   todo: consider using a uint16 for this. Which does not support NaN. So a sentinel missing value would be needed.

    bins_ch1 = np.full(
        FireParams.TOTAL_BINS.value, FireParams.FILLVAL.value, dtype=np.uint16
    )
    bins_ch2 = bins_ch1

    bound_ch1 = 24 + nbytes
    bound_ch2 = 24 + 2 * nbytes if enach1 else bound_ch1

    kbin = 24
    kout = 0
    if enach1:
        while kbin < bound_ch1:
            binval = int.from_bytes(
                buff[k + kbin : k + kbin + 2], byteorder="little", signed=False
            )
            bins_ch1[kout] = binval
            kbin += 2
            kout += 1

    kout = 0

    if enach2:
        while kbin < bound_ch2:
            binval = int.from_bytes(
                buff[k + kbin : k + kbin + 2], byteorder="little", signed=False
            )
            bins_ch2[kout] = binval
            kbin += 2
            kout += 1

    return bins_ch1, bins_ch2


def parse_fire(file: str, start_offset: int = 0):
    with open(file, "rb") as f:
        buff = f.read()

    # wrapper type of the whole thing should be dict[np.ndarray]
    #   to build: make an ndarray for each field and append through parsing.
    #   at the end: add keys and make it a dict.

    # numpy structured array (name, type, [shape]). Covers the whole packet EXCEPT bins1 and bins2:
    ptype = np.dtype(
        [
            ("CCSDS_VERSION_NUMBER", np.uint8),
            ("CCSDS_PACKET_TYPE", np.uint8),
            ("CCSDS_SECONDARY_FLAG", np.uint8),
            ("CCSDS_APID", np.uint16),
            ("CCSDS_SEQUENCE_FLAG", np.uint8),
            ("CCSDS_SEQUENCE_COUNT", np.uint16),
            ("CCSDS_PACKET_LENGTH", np.uint16),
            ("PFLAGS", np.uint8),
            ("ACCUM", np.uint8),
            ("NBIN", np.uint8),
            ("MUX", np.uint8),
            ("HK", np.uint16),
            ("TTAG", np.uint32),
            ("BASE1", np.uint16),
            ("BASE2", np.uint16),
            ("RATE1", np.uint16),
            ("RATE2", np.uint16),
            ("BINS_CH1", np.uint16, FireParams.TOTAL_BINS.value),
            ("BINS_CH2", np.uint16, FireParams.TOTAL_BINS.value),
        ]
    )
    res = np.empty(0, dtype=ptype)

    k = start_offset
    last_nbin = 0
    while k < len(buff):
        version = ccsds_version_number(k, buff)
        packet_type = ccsds_packet_type(k, buff)
        secondary_header = ccsds_secondary_flag(k, buff)
        id = apid(k, buff)
        seq_flag = sequence_flag(k, buff)
        seq_count = sequence_count(k, buff)
        L = length(k, buff)

        assert packet_type == 0

        this_pflags = pflags(k, buff)
        this_accum = accum(k, buff)

        # note: if this raises, the caller of parse_fire should handle.
        this_nbin = nbin(k, buff)
        if last_nbin != 0 and this_nbin != last_nbin:
            print("ok, bin size changed. I accept this.")
            # raise IFireNBinChangedError(k, {name: res[name] for name in ptype.names})
        last_nbin = this_nbin

        this_mux = mux(k, buff)
        this_hk = hk(k, buff, this_mux)

        if this_mux == 1 and this_nbin != 0xFF & (this_hk >> 8):
            raise ValueError

        this_ttag = ttag(k, buff)
        this_base1 = base1(k, buff)
        this_base2 = base2(k, buff)
        this_rate1 = rate1(k, buff)
        this_rate2 = rate2(k, buff)
        this_bins_ch1, this_bins_ch2 = bins(k, buff, this_pflags, this_nbin)

        thisp = [
            (
                version,
                packet_type,
                secondary_header,
                id,
                seq_flag,
                seq_count,
                L,
                this_pflags,
                this_accum,
                this_nbin,
                this_mux,
                this_hk,
                this_ttag,
                this_base1,
                this_base2,
                this_rate1,
                this_rate2,
                this_bins_ch1,
                this_bins_ch2,
            )
        ]

        res = np.append(res, np.array(thisp, dtype=ptype))

        k += 6 + L + 1

    # turn this into a dict, CCSDSPy wannabe
    d = {name: res[name] for name in ptype.names}
    return d


def promote(base: str):
    if base == "uint8":
        return "uint16"
    elif base == "uint16":
        return "uint32"
    elif base == "uint32":
        return "uint64"
    elif base == "int8":
        return "int16"
    elif base == "int16":
        return "int32"
    elif base == "int32":
        return "int64"
    else:
        raise ValueError


def first_after(arr, k, ignore=0):
    for i, v in enumerate(arr):
        if i >= k and v != ignore:
            return v
    return None


def first_before(arr, k, ignore=0):
    for i, v in enumerate(arr):
        if i < k and v != ignore:
            return v
    return None


def align_command_hk(p, commandables: list[str], fills: dict):
    # find all indices where any commandable value has changed:
    last_values = {}
    change_indices = []
    # over time/packets:
    for k in range(len(p["CCSDS_VERSION_NUMBER"])):
        # over fields:
        for c in commandables:
            # this field at this time
            v = p[c][k]
            # changed if it's not filler, and if not equal to the last one:
            if v != fills[c]:
                if v != last_values.get(c, None):
                    change_indices.append(k)
                last_values[c] = v

    # filter out the indices that are not leading a change.
    front_indices = [0]
    for v, vnext in zip(change_indices, change_indices[1:]):
        if FireParams.TOTAL_MUX.value < vnext - v:
            front_indices.append(vnext)

    aligns = {k: {} for k in front_indices}
    for k in front_indices:
        for c in commandables:
            nextval = first_after(p[c], k, ignore=fills[c])
            if nextval is not None:
                aligns[k][c] = nextval
            else:  # look backwards too, in case this field wasn't commanded during this change.
                prevval = first_before(p[c], k, ignore=fills[c])
                aligns[k][c] = prevval

    # populate the changes into p
    for k in front_indices:
        for c in commandables:
            p[c][k::].fill(aligns[k][c])

    return p


def demux_hk(p, rules: str = demux_file_path):

    with open(rules, "r") as f:
        reader = csv.DictReader(f)
        mux_map = {}
        # make a dict of the demux data, keyed by HK name:
        for row in reader:
            mux_map[row["HK"]] = {
                "MUX": int(row["MUX"]),
                "HK": row["HK"],
                "Side": row["Side"],
                "Type": row["Type"],
                "Raw": int(row["Raw"]),
                "Promote": int(row["Promote"]),
                "Command": int(row["Command"]),
                "Description": row["Description"],
            }

    fills = {}
    for key, val in mux_map.items():
        # don't handle the static items here.
        # todo: maybe do handle them here actually, raise if they change, etc.
        # if val["Command"]:
        #     # something something len(set(...))
        #     continue

        if val["Promote"]:
            this_type = np.dtype(promote(val["Type"]))
        else:
            this_type = np.dtype(val["Type"])

        # maximum value of the base type of this field, use as bitmask
        mask = np.iinfo(val["Type"]).max
        # maximum value of the promoted type of this field.
        #   ASSUMED to be equal to CDF FILLVAL.
        typemax = np.iinfo(this_type).max
        # an array full of typemaxes.
        res = np.full(len(p["HK"]), typemax, dtype=this_type)

        # little endian: >> for LSB and mask for MSB.
        if val["Side"] == "LSB":
            shift = 8
        else:
            shift = 0

        # for the indices where this HK is selected by the mux,
        #   1. cast to correct type
        #   2. do any shifting and masking needed
        res[p["MUX"] == val["MUX"]] = (
            (p["HK"][p["MUX"] == val["MUX"]]) >> shift
        ) & mask

        name = val["HK"]

        # print(f"{name}: {shift=}, {mask=}")

        if val["Raw"]:
            name += "_RAW"

        p[name] = res
        fills[name] = typemax

    # now, align all command fields to the first command field that changed
    commandables = []
    for k in mux_map:
        if mux_map[k]["Command"]:
            if mux_map[k]["Raw"]:
                commandables.append(k + "_RAW")
            else:
                commandables.append(k)

    p = align_command_hk(p, commandables, fills)

    return p


def read_file(data_filename):
    """
    Read a file.

    Parameters
    ----------
    data_filename: str
        A file to read.

    Returns
    -------
    data: list[str]
        A list of paths to CDF files created from raw telemetry data.

    Raises
    ------
    IFireNBinChangedError

    Examples
    --------
    """

    p = parse_fire(data_filename)
    p = demux_hk(p)
    cdf = cdf_from_500(p, l0_depot)

    return cdf

    # below: tearing off new files whenever nbin changes:
    all_cdfs = []
    inc = 0
    start_offset = 0
    still_workin = True
    while still_workin:
        try:
            p = parse_fire(data_filename, start_offset)
            p = demux_hk(p)
            cdf = cdf_from_500(p, l0_depot, increment=inc)
            all_cdfs.append(cdf.pathname)
            # if we make it down here, file is done cooking.
            still_workin = False
            print("done and normal.")
        except IFireNBinChangedError as e:
            start_offset = e.args[0]
            # print(f"splitting telemetry file at offset: {start_offset:#x}")
            partial = e.args[1]
            p = demux_hk(partial)
            cdf = cdf_from_500(p, l0_depot, increment=inc)
            all_cdfs.append(cdf.pathname)
            inc += 1
            continue

    return all_cdfs


if __name__ == "__main__":
    path_cdf = read_file(sys.argv[1])
