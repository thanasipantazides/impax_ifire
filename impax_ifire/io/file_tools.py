"""
This module provides a generic file reader.
"""

import csv
import os
import sys
from enum import Enum

import numpy as np
from astropy import units as u
from astropy.nddata import NDData
from astropy.time import Time
from astropy.timeseries import TimeSeries
from astropy.wcs import WCS
from ndcube import NDCollection, NDCube
from swxsoc.swxdata import SWXData

from impax_ifire.util.telemetry_config import telemetry_config

__all__ = ["read_file", "l1_depot", "l0_depot"]


# a place to temporarily store L0 CDF files. todo: use tempfile.
l0_depot = os.path.join(os.path.dirname(__file__), "..", "data", "log")


def sparky(arr, end="\n"):
    """
    Print out a character grid showing the content of `arr`.

    The character map is:
        - "." for 0
        - "o" for 1
        - "F" for the max value of the type of `arr` (e.g. 255)
        - "f" for the max value of the type of `arr` minus 1 (e.g. 254)
        - "O" for everything else.

    To visualize patterns in pretty raw data.
    """
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
    # minimum allowed value nbin (actual number of energy bins is two larger)
    NBIN_MIN = 4
    # maximum allowed value nbin (actual number of energy bins is two larger)
    NBIN_MAX = 128
    TOTAL_BINS = 130
    # total number of HK MUX elements. Fields that share a uint16 are only counted once.
    TOTAL_MUX = 32
    # the APID value for all FIRE packets.
    VALID_APID = 0x500
    # BIN_PAD = 0xFFFF - 1
    FILLVAL = 0xFFFF


def ccsds_version_number(k: int, buff: bytes):
    """Get CCSDS version number of this packet, assuming k is a the beginning of a packet."""
    return 7 & (buff[k] >> 5)


def ccsds_packet_type(k: int, buff: bytes):
    """Get CCSDS packet type flag for the packet, assuming k is a the beginning of a packet."""
    return 1 & (buff[k] >> 4)


def ccsds_secondary_flag(k: int, buff: bytes):
    """Get CCSDS flag for secondary header, assuming k is a the beginning of a packet."""
    return 1 & (buff[k] >> 3)


def apid(k: int, buff: bytes):
    """Get the APID of this CCSDS packet, assuming k is at the beginning of the packet."""
    id = 0x7FF & int.from_bytes(buff[k : k + 2], byteorder="big", signed=False)
    if id != FireParams.VALID_APID.value:
        raise ValueError
    return id


def sequence_flag(k: int, buff: bytes):
    """Get CCSDS sequence flag from header, assuming k is a the beginning of a packet."""
    return 3 & (buff[k + 2] >> 6)


def sequence_count(k: int, buff: bytes):
    """Get CCSDS sequence counter from header, assuming k is a the beginning of a packet."""
    return 0x3FF & int.from_bytes(buff[k + 2 : k + 3], byteorder="big", signed=False)


def length(k: int, buff: bytes):
    """Get the length field of this CCSDS packet, assuming k is at the beginning of the packet."""
    return 0xFFFF & int.from_bytes(buff[k + 4 : k + 6], byteorder="big", signed=False)


def pflags(k: int, buff: bytes):
    """Get FIRE packet flags field (as an int), assuming k is a the beginning of a packet."""
    return 0xFF & buff[k + 6]


def accum(k: int, buff: bytes):
    """Get FIRE accumulations counter, assuming k is a the beginning of a packet."""
    return 0xFF & buff[k + 7]


def nbin(k: int, buff: bytes):
    """
    Get FIRE NBIN field from packet, assuming k is a the beginning of a packet.

    Note that this is the NBIN value reported directly in the science packet, not demux'd from the HK data.
    """
    nbin = 0xFF & buff[k + 8]

    if nbin < 4 or nbin > 128:
        raise ValueError(f"FIRE {nbin=} out of valid range 4-128.")

    return nbin


def mux(k: int, buff: bytes):
    """Get FIRE HK mux counter value, assuming k is a the beginning of a packet."""
    return 0xFF & buff[k + 9]


def hk(k: int, buff: bytes, mux: int):
    """
    Get FIRE HK value, assuming k is a the beginning of a packet.

    Note that this does nothing to interpret the HK value given the current MUX. For that, call `demux_hk()`.
    """
    return 0xFFFF & int.from_bytes(
        buff[k + 10 : k + 12], byteorder="little", signed=False
    )


def ttag(k: int, buff: bytes):
    """Get FIRE time tag value, assuming k is a the beginning of a packet."""
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
    """
    Get values of FIRE energy bins (in ADC space) for both detector channels.

    The number of energy bins used is dynamically configurable in the instrument. A placeholder sentinel value (`FireParams.FILLVAL.value`, nominally 255) is used when a bin is not populated by the current instrument configuration.

    A bin may not be populated if the `NBIN` count does not include the bin. The total number of bins used per channel is `NBIN + 2`: `NBIN` evenly spaced energy bins plus an overflow plus an underflow energy bin.
    If `NBIN == 5`, custom energy edges (`EDGE1-EDGE5`) are used for the bin edges.

    All bins for a given channel may be deactivated by bits set in the `pflags` field. If `ENACH1 == 0` in `pflags`, channel 1 is disabled, and its bins will all be filled with `FireParams.FILLVAL.value`.

    Returns
    -------
    bins_ch1: np.array(np.uint8)
        1D array of energy bins, with empty values filled by `FireParams.FILLVAL.value`.

    bins_ch2: np.array(np.uint8)
        1D array of energy bins, with empty values filled by `FireParams.FILLVAL.value`.

    """

    enach1 = 1 & (pflags >> 7)
    enach2 = 1 & (pflags >> 6)

    nbytes = 2 * (nbin + 2)

    # fill up the bins with FILLVALs for any channel which is not enabled.
    #   todo: consider using a uint16 for this. As-is, a totally full (but not rolling over) bin would be indistinguishable from a FILLVAL.

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
    """
    Parse all CCSDS packets in a given `file`.

    Returns
    -------
    results: dict[str, np.ndarray]
        A dictionary mapping string names for fields into ndarrays. The first index in the array spans time i.e. packet counter. Most fields only have on index. The fields `"BINS_CH1"` and `"BINS_CH2"` have a second index which spans energy bin (in ADC space).

    """
    with open(file, "rb") as f:
        buff = f.read()

    # numpy structured array (name, type, [shape]). Covers the whole packet EXCEPT demuxed HK data. Also, bitfields are not broken out.

    # todo: infer types here from csv
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
        longness = length(k, buff)

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
                longness,
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

        k += 6 + longness + 1

    # turn this into a dict, CCSDSPy wannabe
    d = {name: res[name] for name in ptype.names}
    return d


def promote(base: str):
    """
    Hacky way to turn a numpy int type into the next one up.

    E.g. `np.uint8` becomes `np.uint16`. Or same for the signed integers.
    """
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
    """Find first non-`ignore` value in `arr` at or after index `k`."""
    for i, v in enumerate(arr):
        if i >= k and v != ignore:
            return v
    return None


def first_before(arr, k, ignore=0):
    """Find first non-`ignore` value in `arr` before index `k`."""
    for i, v in enumerate(arr):
        if i < k and v != ignore:
            return v
    return None


def align_command_hk(p, commandables: list[str], fills: dict):
    """
    Align multiplexed command HK values to the earliest one.

    Some FIRE HK fields are set by OBC command. But all FIRE HK fields are multiplexed; i.e. each FIRE packet includes one of 32 possible FIRE HK fields. So feedback for OBC commands arrives incrementally and gives the impression that some configuration values are changing slowly, when in fact they have all changed at once.

    When we work with the FIRE data, it is advantageous to have all the OBC-commanded HK fields change together, instead of one-by-one. This function lines up all OBC-commanded FIRE HK fields each time they change.

    Parameters
    ----------
    p: dict[str, np.ndarray]
        The output of `demux_hk`, a dictionary labeling each array, with each array spanning time/packet index.
    commandable: list[str]
        A list of the commandable fields in FIRE.
    fills: dict
        A dictionary, with the same keys as p, and values the CDF FILLVAL values for each key. This is necessary because _some_ types are promoted to wider types in parsing.

    Returns
    -------
    p: dict[str, np.ndarray]
        The same as the input `p` dictionary but with the values of the `commandable` keys changed to be aligned.
    """

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


def demux_hk(p: dict):
    """
    Demultiplex the FIRE housekeeping data.

    FIRE housekeeping data is delivered piecemeal in science packets. Each new packet gives a new HK field. By inspecting the value in the MUX field we can figure out how to interpret the HK field.

    After demultiplexing the HK, this function also calls `align_command_hk` to manually time-align a subset of the HK fields, which are commanded parameters in FIRE.

    Parameters
    ----------
    p: dict[str, np.ndarray]
        The output of `parse_fire`, a dictionary labeling a bunch of numpy arrays, one for each field in the CCSDS packet.

    rules: str
        Path to a CSV file with names and rules for handling FIRE HK data.

    Returns
    -------
    p: dict[str, np.ndarray]
        The same as the input `p` dictionary but with some more keys for all the demultiplexed HK data.
    fills: dict[str, int]
        The same keys as input `p` dictionary but the values are the FILLVAL value for each field. Some fields are promoted to larger types in `demux_hk` to avoid using their max value as a FILLVAL. So this tells you the real FILLVAL.
    """

    tlm_conf = telemetry_config()

    fills = {}
    for key, val in tlm_conf.items():
        if not val.is_hk:
            continue

        # an array full of the max value of the eventual CDF type.
        res = np.full(len(p["HK"]), val.fillval, dtype=val.L0_type)

        # parse the HK data (at the timesteps indicated by MUX) as ints
        res[p["MUX"] == val.hk_mux] = (
            (p["HK"][p["MUX"] == val.hk_mux]) >> val.hk_rightshift
        ) & (2**val.width - 1)

        name = val.icd_name

        p[name] = res
        fills[name] = val.fillval

    # now, align all command fields to the first command field that changed
    # todo: add pflags here?
    commandables = []
    for k in tlm_conf:
        if tlm_conf[k].hk_command:
            commandables.append(k)

    p = align_command_hk(p, commandables, fills)

    return p, fills


def make_timeseries(p, fills):
    tlm_config = telemetry_config()

    # base_epoch = "2000-01-01T00:00:00"

    # todo: correct for rollover.
    #   FIRE ttag counts 10 ms per LSB, so divide by 100.0 to seconds
    t = p["TTAG"].astype(np.float64) / 100.0 * u.s
    t_gps = Time(t, format="gps")
    ddata = {}

    # make an empty bins support variable (record-varying).
    #   These are the values of the energy bin edges.
    #   This is padded out to 130 energy bins, the largest possible value.
    #   Unused bins (based on dynamic NBIN value) get the FILLVAL.
    #   This ebins variable is used as the DEPEND_1 value for ADC-space spectra.
    #   In further-up data levels, ebins should be transformed to the energy-space bin boundaries.
    ebins = np.full(
        (len(p["NBIN"]), FireParams.TOTAL_BINS.value),
        np.iinfo(p["NBIN"][0].dtype).max,
        dtype=p["NBIN"].dtype,
    )
    # At each timestep in the record, update the number of (evenly-spaced) bins
    #   based on the current NBIN value.
    for k in range(len(p["NBIN"])):
        these_bins = np.arange(0, p["NBIN"][k] + 2, 1, dtype=p["NBIN"].dtype)
        ebins[k, 0 : len(these_bins)] = these_bins

    # Format the bins into an NDData object as SWXData desires us to.
    ebins_support = NDData(
        data=ebins,
        meta={"DEPEND_0": "Epoch", "CATDESC": "ADC bin edges", "VAR_TYPE": "data"},
    )
    support = {"ebins": ebins_support}

    # Make valid columns for building a TimeSeries
    #   by applying units and types to each field in p:
    collectable = []
    spectrum_index = 1
    for k in p:
        if len(p[k].shape) == 1:
            # give columns of TimeSeries the desired CDF variable names:
            cname = tlm_config[k].cdf_fieldnam
            ddata[cname] = u.Quantity(p[k], tlm_config[k].L0_unit, dtype=p[k][0].dtype)

        # For the spectra, format these especially using WCS (for SWXData):
        elif len(p[k].shape) == 2:
            espectrum_cs = WCS(naxis=2)
            espectrum_cs.wcs.ctype = ["EBIN", "TIME"]
            espectrum_cs.wcs.cname = ["energy_bin", "time"]
            espectrum_cs.wcs.cunit = ["ct", "s"]
            # why not [0,0]?
            espectrum_cs.wcs.crpix = [1, 1]
            espectrum_cs.wcs.crval = [0, 0]
            espectrum_cs.wcs.cdelt = [1, 1]

            espectrum = (
                "espectrum" + str(spectrum_index),
                NDCube(
                    data=p[k],
                    wcs=espectrum_cs,
                    unit="ct",
                    meta={
                        "CATDESC": "ADC counts per bin",
                        "VAR_TYPE": "data",
                        "DEPEND_0": "Epoch",
                        "DEPEND_1": "ebins",
                    },
                ),
            )

            collectable.append(espectrum)
            spectrum_index += 1

    # Collect the spectra for SWXData:
    ndc = NDCollection(collectable)

    # todo: confirm epoch with Josh
    ts = TimeSeries(time=t_gps, data=ddata)

    return ts, ndc, support


def make_cdf(
    ts: TimeSeries, ndc: NDCollection, support: dict, save_dir: str = l0_depot
):
    """Save the packet data as a CDF file.

    This uses the swxsoc.swxdata.SWXData interface to construct the CDF.

    Parameters
    ----------
    ts: TimeSeries
        An astropy TimeSeries with a column called `time`. The columns of this table become scalar records in the CDF file. Spectra (or other nonscalar records) should be passed in the `ndc` argument. For FIRE data, `ts` should include all packet fields except the bin data and the bin indices.

    ndc: NDCollection
        Contains the spectral data that varies in time. For FIRE, this includes `tuple(name: str, data: NDCube)` for each of the two channels. The NDCube contains WCS information on the spectrum at each timestep, and meta data conforming to the CDF standard.

    support: dict
        Contains support data that may vary in time. For FIRE, one key `ebins` points to an `NDData` record of the bin centers used for the spectral data. These bins are variable in time and depend on a configured value from the flight computer. With the exception of the special `NBIN==5` mode, the bins are evenly spaced. The lowest and highest bins are underflow and overflow respectively.

    Returns
    -------
    cdf: str
        Path to the CDF file created from this data.

    """
    tlm_config = telemetry_config()

    meta = SWXData.global_attribute_template(
        instr_name="ifire", data_level="l0", version="1.0.0"
    )

    data = SWXData(timeseries=ts, meta=meta, support=support, spectra=ndc)

    assert len(set(data.timeseries.keys())) == len(data.timeseries.keys())
    # print(f"timeseries keys: {data.timeseries.keys()}")
    # print(f"support keys: {data.support.keys()}")
    # print(f"spectra: {data.spectra.keys()}")

    for k in data.timeseries.keys():
        tlm_names = list(filter(lambda x: tlm_config[x].cdf_fieldnam == k, tlm_config))
        if not tlm_names:
            continue
        tlm_obj = tlm_config[tlm_names[0]]
        data.timeseries[k].meta["CATDESC"] = tlm_obj.description

    cdf = data.save(save_dir)

    return cdf


def read_file(data_filename):
    """
    Read a raw telemetry file and create a CDF L0 file from it.

    Parameters
    ----------
    data_filename: str
        A file to read.

    Returns
    -------
    data: str
        A path to the CDF file created from the telemetry data.


    Examples
    --------
    .. code-block:: python
        cdf_path = read_file("path/to/raw/ifire_20261019050403.tlm")

    """

    p = parse_fire(data_filename)
    p, fills = demux_hk(p)
    ts, ndc, support = make_timeseries(p, fills)

    cdf = make_cdf(ts, ndc, support)
    return cdf


if __name__ == "__main__":
    path_cdf = read_file(sys.argv[1])
