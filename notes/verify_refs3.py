"""Verify metadata of the references added in the journal version (Crossref
public REST API; only DOIs are sent).  DOIs were chosen from the output of
find_refs_journal.py.

    python notes/verify_refs3.py notes/refs_verified3.json
"""
import json
import sys
import time
import urllib.request

DOIS = {
    "kudela2022": "10.1038/s42256-022-00579-0",
    "aranha2022": "10.1007/s11721-021-00202-9",
    "hansen2021coco": "10.1080/10556788.2020.1808977",
    "chicano2017": "10.1145/3071178.3071285",
    "jones2023": "10.1145/3570723",
    "aitsaadi2022": "10.1007/s11831-022-09742-7",
    "debnath2024": "10.3390/rs16214019",
    "qu2020": "10.1016/j.asoc.2020.106099",
    "yu2020": "10.1016/j.knosys.2020.106209",
    "phung2020": "10.1016/j.asoc.2020.106705",
    "nosek2018": "10.1073/pnas.1708274114",
    "derrac2011": "10.1016/j.swevo.2011.02.002",
    "carrasco2020": "10.1016/j.swevo.2020.100665",
    "campelo2023": "10.1162/artl_a_00402",
    "pivtoraiko2009": "10.1002/rob.20285",
    "deng2023": "10.3390/math11091987",
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
            pub = (m.get("published-print", {}).get("date-parts") or [[None]])[0][0]
            rec = dict(doi=doi, title=(m.get("title") or [""])[0], authors=auth,
                       venue=(m.get("container-title") or [""])[0], volume=m.get("volume", ""),
                       issue=m.get("issue", ""), pages=m.get("page", m.get("article-number", "")),
                       year=year, year_print=pub, type=m.get("type", ""), event=(m.get("event") or {}).get("name", ""))
            print(f"[OK] {key}: {rec['authors']} ({rec['year']}/{pub}). {rec['title']}. {rec['venue']} "
                  f"{rec['volume']}({rec['issue']}) {rec['pages']} [{rec['type']}] {rec['event']}".encode("ascii", "replace").decode())
        except Exception as e:  # noqa: BLE001
            rec = dict(doi=doi, error=str(e))
            print(f"[FAIL] {key}: {doi}: {e}")
        out[key] = rec
        time.sleep(0.3)
    with open(sys.argv[1] if len(sys.argv) > 1 else "refs_verified3.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
