"""Download the public Christmas Island DEM used by the SPSO reference code
(Phung & Ha, Applied Soft Computing 107 (2021) 107376) into this folder and
check its SHA-256.  The file is not redistributed with this package; see the
license of the original repository (https://github.com/duongpm/SPSO)."""
import hashlib
import os
import urllib.request

URL = "https://raw.githubusercontent.com/duongpm/SPSO/main/ChrismasTerrain.tif"
SHA256 = "481F16EFBEA8270A6D609AD5DC147063E8ACBD88482EA0BC36D99B1E99BDA197"

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ChrismasTerrain.tif")
    urllib.request.urlretrieve(URL, out)
    h = hashlib.sha256(open(out, "rb").read()).hexdigest().upper()
    print("SHA-256", h, "OK" if h == SHA256 else "MISMATCH")
