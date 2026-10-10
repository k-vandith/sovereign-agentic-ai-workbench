import os, sys, tempfile, pathlib
t = tempfile.mkdtemp()
os.environ.update(DATA_DIR=t, UPLOAD_DIR=t+"/u", VECTORSTORE_DIR=t+"/v", CONVERSATION_DIR=t+"/c", LLM_BACKEND="demo", API_TOKEN="x")
sys.path.insert(0, ".")
from src.agent import RAGAgent
docs = {
 "valves.txt": "Valve V-417 maximum working pressure is 38.5 bar. Valve V-418 maximum working pressure is 41.0 bar. Valve V-419 maximum working pressure is 12.0 bar. Valve V-417 is installed on the cooling loop of reactor R-9.",
 "pumps.txt": "Pump P-204 flow rate is 120 cubic metres per hour. Pump P-205 flow rate is 95 cubic metres per hour. Pump P-204 serves the feed line of tank T-3.",
 "safety.txt": "In case of a toxic gas alarm, evacuate to assembly point B and notify the shift supervisor. Eye wash stations must be tested every week. Hot work permits are valid for one shift only.",
 "recipe.txt": "Recipe XR-77 requires heating to 180 degrees Celsius for 45 minutes and then cooling to 40 degrees. Recipe XR-78 requires 150 degrees for 30 minutes.",
}
p = pathlib.Path(t, "docs"); p.mkdir()
for n, s in docs.items(): (p/n).write_text(s)
for i in range(60):  # near-duplicate distractors
    (p/f"log{i}.txt").write_text(f"Maintenance log {i}: inspected valve V-{500+i} pressure test at {20+i%9}.5 bar, pump P-{300+i} flow {80+i} cubic metres per hour, no leaks found.")
a = RAGAgent()
a.ingest_directory(p)
print(f"EMBEDDING BACKEND: {a.store.embedding_status()}")
valid = [  # (question, expected file, expected substring)
 ("What is the maximum working pressure of valve V-417?", "valves.txt", "38.5"),
 ("how much pressure can V-418 take", "valves.txt", "41.0"),
 ("highest allowed pressure for valve 417", "valves.txt", "38.5"),
 ("What is the flow rate of pump P-205?", "pumps.txt", "95"),
 ("which tank does pump P-204 feed", "pumps.txt", "T-3"),
 ("what should I do if the toxic gas alarm goes off", "safety.txt", "assembly point B"),
 ("how often do eyewash stations need testing", "safety.txt", "every week"),
 ("how long is a hot work permit valid", "safety.txt", "one shift"),
 ("what temperature does recipe XR-77 need", "recipe.txt", "180"),
 ("cooking time for XR-78", "recipe.txt", "30 minutes"),
 ("which reactor is valve V-417 connected to", "valves.txt", "R-9"),
 ("Where do we gather during an evacuation?", "safety.txt", "assembly point B"),
]
oos = ["What is the capital of France?", "Maximum working pressure of valve V-999?", "Who won the cricket world cup?",
       "What is the flow rate of pump P-999?", "How do I bake a chocolate cake?", "What is the salary of the plant manager?",
       "What is the maximum working pressure of the Eiffel Tower?", "Tell me about the plant's quarterly revenue"]
ok1 = nf = hit_text = 0
print("VALID QUESTIONS")
for q, f, s in valid:
    r = a.query(q, use_tools=False)
    src = [x["source"] for x in r["sources"]]
    top = src[0] if src else None
    notfound = r["answer"].startswith("Not found")
    txt = any(s.lower() in (x["snippet"]).lower() for x in r["sources"]) or s.lower() in r["answer"].lower()
    ok1 += (top == f); nf += notfound; hit_text += txt
    print(f"  top1={'OK ' if top==f else 'BAD'} notfound={notfound!s:5} fact_in_context={txt!s:5} | {q} -> {top}")
print(f"top1 correct file {ok1}/{len(valid)}, wrongly 'not found' {nf}/{len(valid)}, fact present in returned context {hit_text}/{len(valid)}")
print("OUT-OF-SCOPE QUESTIONS")
bad = 0
for q in oos:
    r = a.query(q, use_tools=False); nfq = r["answer"].startswith("Not found"); bad += (not nfq)
    print(f"  {'OK ' if nfq else 'ANSWERED'} | {q} -> {[x['source'] for x in r['sources']][:2]}")
print(f"out-of-scope answered instead of not-found: {bad}/{len(oos)}")
print("RETRIEVAL QUALITY GATES")
checks = [
    (ok1 >= 10, f"top-1 correct source: {ok1}/12; minimum 10/12"),
    (nf <= 1, f"valid questions wrongly not-found: {nf}/12; maximum 1/12"),
    (bad <= 1, f"out-of-scope questions answered: {bad}/8; maximum 1/8"),
]
for passed, detail in checks:
    print(f"  {'PASS' if passed else 'FAIL'} {detail}")
if not all(passed for passed, _detail in checks):
    raise SystemExit(1)
print("RETRIEVAL QUALITY GATES: PASS")
