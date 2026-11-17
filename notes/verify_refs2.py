"""Verify metadata of the references added in the second revision (Crossref
public REST API; only DOIs are sent)."""
import json
import sys
import time
import urllib.request

DOIS = {
    "whitley2016": "10.1162/EVCO_a_00184",
    "tinos2015": "10.1145/2725494.2725497",
    "cotta1998": "10.1007/BFb0056873",
    "zhan2025": "10.3390/s25092730",
    "wang2025": "10.3390/drones9090638",
    "rasoulie2026": "10.1038/s41598-026-42372-8",
    "hooshyar2023": "10.3390/drones7120687",
    "fu2024": "10.1007/s10462-024-10716-3",
    "eremeev2014a": "10.2298/YJOR130731040E",
    "eremeev2014b": "10.2298/YJOR131030041E",
    "muehlenbein1996": "10.1007/978-1-4613-1361-8_4",
    "amini1990": "10.1109/34.57681",
    "werling2010": "10.1109/ROBOT.2010.5509799",
    "hansen2001": "10.1162/106365601750190398",
    "auger2005": "10.1109/CEC.2005.1554902",
    "vargha2000": "10.3102/10769986025002101",
    "luus2000": "10.1201/9781420036022",
}


def fetch(doi):
    req = urllib.request.Request("https://api.crossref.org/works/" + doi,
                                 headers={"User-Agent": "reference-check/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))["message"]


def main():
    out = {}
    for key, doi in DOIS.items():
        try:
            m = fetch(doi)
            auth = ", ".join(f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", []))
            year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
            rec = dict(doi=doi, title=(m.get("title") or [""])[0], authors=auth,
                       venue=(m.get("container-title") or [""])[0], volume=m.get("volume", ""),
                       issue=m.get("issue", ""), pages=m.get("page", m.get("article-number", "")),
                       year=year, type=m.get("type", ""))
            print(f"[OK] {key}: {rec['authors']} ({rec['year']}). {rec['title']}. {rec['venue']} "
                  f"{rec['volume']}({rec['issue']}) {rec['pages']} [{rec['type']}]")
        except Exception as e:  # noqa: BLE001
            rec = dict(doi=doi, error=str(e))
            print(f"[FAIL] {key}: {doi}: {e}")
        out[key] = rec
        time.sleep(0.3)
    with open(sys.argv[1] if len(sys.argv) > 1 else "refs_verified2.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
