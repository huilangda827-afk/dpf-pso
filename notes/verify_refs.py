"""Verify reference metadata against the public Crossref REST API (no login).
Only DOIs are sent; prints title / authors / venue / year for manual checking."""
import json
import sys
import time
import urllib.request

DOIS = {
    "phung2021": "10.1016/j.asoc.2021.107376",
    "kennedy1995": "10.1109/ICNN.1995.488968",
    "shi1998": "10.1109/ICEC.1998.699146",
    "mirjalili2014": "10.1016/j.advengsoft.2013.12.007",
    "mirjalili2016": "10.1016/j.advengsoft.2016.01.008",
    "heidari2019": "10.1016/j.future.2019.02.028",
    "storn1997": "10.1023/A:1008202821328",
    "tanabe2014": "10.1109/CEC.2014.6900380",
    "fu2012": "10.1109/TSMCA.2011.2159586",
    "aggarwal2020": "10.1016/j.comcom.2019.10.014",
    "whitley2009": "10.1145/1569901.1570026",
    "tinos2020": "10.1162/evco_a_00254",
    "sorensen2015": "10.1111/itor.12001",
    "derrac2011": "10.1016/j.swevo.2011.02.002",
    "kavraki1996": "10.1109/70.508439",
    "karaman2011": "10.1177/0278364911406761",
    "neri2012": "10.1016/j.swevo.2011.11.003",
    "liu2025": "10.1038/s41598-025-85912-4",
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
                       year=year)
            print(f"[OK] {key}: {rec['authors']} ({rec['year']}). {rec['title']}. "
                  f"{rec['venue']} {rec['volume']}({rec['issue']}) {rec['pages']}")
        except Exception as e:  # noqa: BLE001
            rec = dict(doi=doi, error=str(e))
            print(f"[FAIL] {key}: {doi}: {e}")
        out[key] = rec
        time.sleep(0.3)
    with open(sys.argv[1] if len(sys.argv) > 1 else "refs_verified.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
