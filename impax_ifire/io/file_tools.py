"""
This module provides a generic file reader.
"""

import sys

import numpy as np

from ..util.exceptions import IFireNBinChangedException

__all__ = ["read_file"]


FIRE_APID = 0x500


def ccsds_version_number(k: int, buff: bytes):
    """Get CCSDS version number of this packet, assuming k is a the beginning of a packet,"""
    return 7 & (buff[k] >> 5)


def ccsds_packet_type(k: int, buff: bytes):
    return 1 & (buff[k] >> 4)


def ccsds_secondary_flag(k: int, buff: bytes):
    return 1 & (buff[k] >> 3)


def apid(k: int, buff: bytes):
    """Get the APID of this CCSDS packet, assuming k is at the beginning of the packet."""
    return 0x7FF & int.from_bytes(buff[k : k + 2], byteorder="big", signed=False)


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


def nbin(k: int, buff: bytes, last_nbin: int = 0):
    nbin = 0xFF & buff[k + 8]

    if last_nbin != 0 and nbin != last_nbin:
        raise IFireNBinChangedException(k)

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
    bins = np.empty(nbytes)
    bins[:] = np.nan

    kbin = 24
    kout = 0
    if enach1:
        while kbin < 24 + nbytes:
            binval = int.from_bytes(
                buff[k + kbin : k + kbin + 2], byteorder="little", signed=False
            )
            bins[kout] = binval
            kbin += 2
            kout += 1
    kout = len(bins) // 2

    if enach2:
        while kbin < 24 + 2 * nbytes:
            binval = int.from_bytes(
                buff[k + kbin : k + kbin + 2], byteorder="little", signed=False
            )
            bins[kout] = binval
            kbin += 2
            kout += 1

    return bins


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
            ("pflags", np.uint8),
            ("accum", np.uint8),
            ("nbin", np.uint8),
            ("mux", np.uint8),
            ("hk", np.uint16),
            ("ttag", np.uint32),
            ("base1", np.uint16),
            ("base2", np.uint16),
            ("rate1", np.uint16),
            ("rate2", np.uint16),
        ]
    )
    res = np.empty(0, dtype=ptype)

    nbins_known = False

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

        assert id == 0x500
        assert packet_type == 0

        this_pflags = pflags(k, buff)
        this_accum = accum(k, buff)

        # note: if this raises, the caller of parse_fire should handle.
        this_nbin = nbin(k, buff, last_nbin=last_nbin)

        # if all_bins is uninitialized, initialize it with the newfound nbin value.
        if not nbins_known:
            # rewrite the type signature and reinitialize the result array to hold nbins worth of bins:
            ptype = np.dtype(ptype.descr + [("bins", np.uint16, 2 * (this_nbin + 2))])
            if len(res) != 0:
                raise IFireNBinChangedException(k)
            res = np.empty(0, dtype=ptype)
            nbins_known = True

        this_mux = mux(k, buff)
        this_hk = hk(k, buff, this_mux)
        this_ttag = ttag(k, buff)
        this_base1 = base1(k, buff)
        this_base2 = base2(k, buff)
        this_rate1 = rate1(k, buff)
        this_rate2 = rate2(k, buff)
        this_bins = bins(k, buff, this_pflags, this_nbin)

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
                this_bins,
            )
        ]

        res = np.append(res, np.array(thisp, dtype=ptype))

        # update state
        last_nbin = this_nbin
        k += 6 + L + 1

    # turn this into a dict, CCSDSPy wannabe
    d = {k: res[k] for k in ptype.names}
    return d


def read_file(data_filename):
    """
    Read a file.

    Parameters
    ----------
    data_filename: str
        A file to read.

    Returns
    -------
    data: str

    Examples
    --------
    """
    return


if __name__ == "__main__":
    p = parse_fire(sys.argv[1])
