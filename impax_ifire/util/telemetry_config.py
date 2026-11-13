import csv
import os

import numpy as np
from astropy import units as u

# path to the FIRE packet fields config file
config_path = os.path.join(os.path.dirname(__file__), "..", "data", "ifire-fields.csv")


__all__ = ["TelemetryElement", "uptype", "telemetry_config"]


def uptype(T: str):
    """Promote a Numpy type string (e.g. uint8, int32, float64, etc) to the next type up."""
    base = T.rstrip("0123456789")
    size = T[len(base) :]
    size = int(size, base=10)
    Tstr = base + str(2 * size)
    return np.dtype(Tstr)


class TelemetryElement:
    def __init__(
        self,
        icd_name: str = "",
        cdf_fieldnam: str = "",
        offset: int = 0,
        width: int = 8,
        endian: str = "big",
        is_hk: bool = False,
        is_bins: int = 0,
        hk_mux: int = -1,
        hk_rightshift: int = 0,
        promote: bool = False,
        hk_command: bool = False,
        validmin: int | float = 0,
        validmax: int | float = 255,
        L0_type=np.uint8,
        L1_type=np.uint8,
        L0_unit: str | int = 1,
        L1_unit: str | int = 1,
        L0L1_rule=lambda x: x,
        description: str = "",
    ):
        self.icd_name = icd_name
        self.cdf_fieldnam = cdf_fieldnam
        self.offset = offset
        self.width = width
        self.endian = endian
        self.is_hk = is_hk
        self.is_bins = is_bins
        self.promote = promote
        self.hk_mux = hk_mux
        self.hk_rightshift = hk_rightshift
        self.hk_command = hk_command
        self.validmin = validmin
        self.validmax = validmax
        self.L0_type = np.dtype(L0_type)
        self.L1_type = np.dtype(L1_type)
        self.L0_unit = L0_unit
        self.L1_unit = L1_unit
        self.L0L1_rule = L0L1_rule
        self.description = description

        typename = str(self.L0_type)
        if self.promote:
            self.L0_type = uptype(typename)

        # set CDF fillval appropriately based on this type:
        if "float" in typename or "double" in typename:
            self.fillval = -1.0e31
        elif "uint" in typename:
            self.fillval = np.iinfo(self.L0_type).max
        elif "int" in typename:
            self.fillval = np.iinfo(self.L0_type).min
        else:
            raise ValueError

    def parse(self, buff: bytes):
        """Parse the buffer based on this field's offset/width parameters. Start of buff is the start of a CCSDS packet."""
        # if self.is_hk:
        #     raise ValueError

        kstart = self.offset // 8
        kend = kstart + 1 + (self.width - 1) // 8

        if kend >= len(buff):
            raise IndexError

        L0_signedtype = np.issubdtype(self.L0_type, np.signedinteger)

        # always interpret buffer as unsigned, even if resultant should be signed.
        # we still need to shift and mask.
        raw = int.from_bytes(buff[kstart:kend], byteorder=self.endian, signed=False)

        shift = 8 * kend - (self.offset + self.width)
        mask = 2**self.width - 1

        val = mask & (raw >> shift)

        # 2's complement if this is to be a signed integer:
        if L0_signedtype:
            if val & (1 << (self.width - 1)):
                val -= 1 << self.width

        return val

    def parseu(self, buff: bytes):
        val = self.parse(buff)
        valu = u.Quantity(val, self.L0_unit, self.L0_type)
        return valu


def telemetry_config(csv_filename=config_path):
    telemetry_config = {}
    with open(csv_filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["icd_name"] == "BINS1" or row["icd_name"] == "BINS2":
                continue

            tlm = TelemetryElement(
                icd_name=row["icd_name"],
                cdf_fieldnam=row["cdf_name"],
                offset=int(row["offset"], base=10),
                width=int(row["width"], base=10),
                endian=row["endian"],
                is_hk=bool(int(row["is_hk"], base=10)),
                is_bins=int(row["is_bins"], base=10),
                hk_mux=int(row["hk_mux"], base=10),
                hk_rightshift=int(row["hk_rightshift"], base=10),
                promote=bool(int(row["promote"], base=10)),
                hk_command=bool(int(row["hk_command"], base=10)),
                validmin=int(row["validmin"], base=10),
                validmax=int(row["validmax"], base=10),
                L0_type=row["level0_type"],
                L1_type=row["level1_type"],
                L0_unit=row["level0_unit"],
                L1_unit=row["level1_unit"],
                L0L1_rule=row["level0_1_map"],
                description=row["description"],
            )
            telemetry_config[tlm.icd_name] = tlm
    return telemetry_config
