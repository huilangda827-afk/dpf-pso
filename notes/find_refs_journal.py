"""Look up candidate references for the journal version in the Crossref
public REST API (only paper titles are sent).  Prints the top matches so that
the exact DOI can be chosen by hand; verify_refs3.py then fetches the
metadata of the chosen DOIs.

    python notes/find_refs_journal.py > notes/find_refs_journal_out.txt
"""
import json
import time
import urllib.parse
import urllib.request

QUERIES = [
    ("kudela2022", "A critical problem in benchmarking and analysis of evolutionary computation methods"),
    ("aranha2022", "Metaphor-based metaheuristics, a call for action: the elephant in the room"),
    ("hansen2021coco", "COCO: a platform for comparing continuous optimizers in a black-box setting"),
    ("chicano2017", "Optimizing one million variable NK landscapes by hybridizing deterministic recombination and local search"),
    ("jones2023", "Path-planning for unmanned aerial vehicles with environment complexity considerations: A survey"),
    ("aitsaadi2022", "UAV path planning using optimization approaches: A survey"),
    ("debnath2024", "A review of UAV path-planning algorithms and obstacle avoidance methods for remote sensing applications"),
    ("qu2020", "A novel reinforcement learning based grey wolf optimizer algorithm for unmanned aerial vehicles (UAVs) path planning"),
    ("yu2020", "A constrained differential evolution algorithm to solve UAV path planning in disaster scenarios"),
    ("phung2020", "Motion-encoded particle swarm optimization for moving target search using UAVs"),
    ("nosek2018", "The preregistration revolution"),
    ("derrac2011", "A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms"),
    ("campelo2023", "Lessons from the evolutionary computation Bestiary"),
    ("pivtoraiko2009", "Differentially constrained mobile robot motion planning in state lattices"),
    ("drones_pso_2024", "improved particle swarm optimization three-dimensional UAV path planning"),
    ("drones_de_2025", "differential evolution UAV path planning threat terrain"),
]


def search(q, rows=3):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": q, "rows": rows, "select": "DOI,title,author,container-title,issued,volume,issue,page,type"})
    req = urllib.request.Request(url, headers={"User-Agent": "reference-check/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["items"]


def main():
    for key, q in QUERIES:
        print(f"## {key}: {q}")
        try:
            for it in search(q):
                auth = ", ".join(a.get("family", "") for a in it.get("author", [])[:3])
                year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
                print(f"  - {it.get('DOI')} | {(it.get('title') or [''])[0][:110]} | {auth} | "
                      f"{(it.get('container-title') or [''])[0][:50]} {it.get('volume', '')}({it.get('issue', '')}) "
                      f"{it.get('page', '')} | {year} | {it.get('type')}")
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {e}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
