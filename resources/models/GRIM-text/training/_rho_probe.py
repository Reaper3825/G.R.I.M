#!/usr/bin/env python3
"""Throwaway probe: extract RHO_BUILDUP tables (PRE/POST backward) and trace them."""
import sys, io

PATH = sys.argv[1] if len(sys.argv) > 1 else r"d:\G.R.I.M\resources\models\GRIM-text\training\logs\training_1781465589204018700.log"

def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

def fnum(s):
    s = s.strip().strip("x").strip("%")
    s = s.replace("+", "")
    try:
        return float(s)
    except ValueError:
        return None

def parse_blocks(path):
    blocks = []
    cur_batch = None
    cur_step = None
    cur_loss = None
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # track context
        if "EXPLICIT_TRAINING_FORWARD_COMPLETE" in line and "batch=" in line:
            try:
                cur_batch = int(line.split("batch=")[1].split()[0])
            except Exception:
                pass
        if "[Step " in line and "loss=" in line:
            try:
                cur_step = int(line.split("[Step ")[1].split("]")[0])
                cur_loss = fnum(line.split("loss=")[1].split()[0])
            except Exception:
                pass
        if "RHO_BUILDUP_EQUATION" in line and "phase=" in line:
            phase = "PRE" if "PRE_BACKWARD" in line else "POST"
            blk = {"phase": phase, "batch": cur_batch, "step": cur_step,
                   "loss": cur_loss, "line": i + 1, "layers": {}}
            j = i + 1
            while j < n:
                lj = lines[j]
                # stop when we leave the indented block (next timestamped line)
                if lj.startswith("[20") or ("RHO_BUILDUP_EQUATION" in lj and "phase=" in lj):
                    break
                s = lj.strip()
                toks = s.split()
                if toks and (toks[0] in ("emb", "lm_in") or (toks[0].startswith("L") and toks[0][1:].isdigit())):
                    # label rho delta(vs) hrms abs signed rawsum centered meanrms muavg muscale interp
                    label = toks[0]
                    try:
                        rho = fnum(toks[1])
                        # delta token may be "—" or "+0.0242(ve)"
                        idx = 2
                        if toks[2] == "—" or toks[2].startswith("\u2014"):
                            idx = 3
                        else:
                            idx = 3
                        hrms = fnum(toks[idx]); abs_dot = fnum(toks[idx+1])
                        signed = fnum(toks[idx+2]); rawsum = fnum(toks[idx+3])
                        centered = fnum(toks[idx+4]); meanrms = fnum(toks[idx+5])
                        muavg = fnum(toks[idx+6]); muscale = fnum(toks[idx+7])
                        blk["layers"][label] = dict(rho=rho, hrms=hrms, abs_dot=abs_dot,
                            signed=signed, rawsum=rawsum, centered=centered,
                            meanrms=meanrms, muavg=muavg, muscale=muscale)
                    except Exception:
                        pass
                if s.startswith("SUMMARY:"):
                    if "growth=" in s:
                        blk["growth"] = fnum(s.split("growth=")[1].split()[0])
                    if "h_rms_growth=" in s:
                        blk["hrms_growth"] = fnum(s.split("h_rms_growth=")[1].split()[0])
                if s.startswith("SPLIT"):
                    if "\u03c1_atom=" in s:
                        blk["rho_atom"] = fnum(s.split("\u03c1_atom=")[1].split()[0])
                    if "\u03c1_nonatom=" in s:
                        blk["rho_nonatom"] = fnum(s.split("\u03c1_nonatom=")[1].split()[0])
                if s.startswith("WORST AMPLIFIER:"):
                    blk["worst"] = s.split("WORST AMPLIFIER:")[1].strip()
                j += 1
            blocks.append(blk)
            i = j
            continue
        i += 1
    return blocks

blocks = parse_blocks(PATH)
print(f"total rho blocks: {len(blocks)}  PRE={sum(1 for b in blocks if b['phase']=='PRE')} POST={sum(1 for b in blocks if b['phase']=='POST')}")

def g(b, lab, key):
    return b["layers"].get(lab, {}).get(key)

# ---- PAIR PRE vs POST by batch ----
bybatch = {}
for b in blocks:
    bybatch.setdefault(b["batch"], {})[b["phase"]] = b
paired = [(bt, d["PRE"], d["POST"]) for bt, d in bybatch.items()
          if bt is not None and "PRE" in d and "POST" in d]
paired.sort(key=lambda x: x[0])
print(f"\n=== PRE vs POST (same batch) : {len(paired)} pairs ===")
print("Does backward mutate the forward-owned hidden tensors? (POST-PRE on identical buffers)")
print("batch  lm_rho_PRE lm_rho_POST  d_lm_rho   L1_PRE  L1_POST  d_L1   emb_PRE emb_POST  d_emb   lm_abs_PRE lm_abs_POST")
psample = paired if len(paired) <= 40 else paired[::max(1,len(paired)//40)]
def d(a,b):
    if a is None or b is None: return None
    return b-a
for bt, pre, post in psample:
    print("{:>5} {:>10} {:>11} {:>9} {:>8} {:>8} {:>7} {:>8} {:>8} {:>7} {:>10} {:>11}".format(
        bt, fmt(g(pre,"lm_in","rho")), fmt(g(post,"lm_in","rho")), fmt(d(g(pre,"lm_in","rho"),g(post,"lm_in","rho"))),
        fmt(g(pre,"L1","rho")), fmt(g(post,"L1","rho")), fmt(d(g(pre,"L1","rho"),g(post,"L1","rho"))),
        fmt(g(pre,"emb","rho")), fmt(g(post,"emb","rho")), fmt(d(g(pre,"emb","rho"),g(post,"emb","rho"))),
        fmt(g(pre,"lm_in","abs_dot")), fmt(g(post,"lm_in","abs_dot"))))

# aggregate magnitude of POST-PRE change
import statistics
for lab in ("emb","L0","L1","L6","lm_in"):
    diffs = [abs(d(g(pre,lab,"rho"),g(post,lab,"rho"))) for _,pre,post in paired
             if g(pre,lab,"rho") is not None and g(post,lab,"rho") is not None]
    if diffs:
        print(f"  |POST-PRE| rho @ {lab:5}: mean={statistics.mean(diffs):.4f} max={max(diffs):.4f}")

# ---- DEPTH PROFILE at selected early batches ----
print("\n=== DEPTH PROFILE (PRE-backward) : rho by layer ===")
order = ["emb","L0","L1","L2","L3","L4","L5","L6","L7","L8","L9","L10","L11","lm_in"]
pre_blocks = [b for b in blocks if b["phase"]=="PRE"]
picks = [pre_blocks[0]] + pre_blocks[1:4] + [pre_blocks[len(pre_blocks)//4], pre_blocks[len(pre_blocks)//2], pre_blocks[-1]]
print("batch " + " ".join(f"{o:>6}" for o in order))
for b in picks:
    print("{:>5} ".format(str(b["batch"])) + " ".join(f"{fmt(g(b,o,'rho')):>6}" for o in order))

# Trajectory of lm_in rho for PRE and POST
print("\n=== TRAJECTORY (sampled) ===")
print("phase step  loss   emb_rho L1_rho lm_rho growth hrms_grow  lm_abs  lm_signed lm_centered lm_meanrms lm_muscale rho_atom rho_nonatom worst")
sample = blocks if len(blocks) <= 60 else blocks[::max(1,len(blocks)//60)]
for b in sample:
    print("{:>4} {:>5} {:>6} {:>7} {:>6} {:>6} {:>6} {:>9} {:>7} {:>9} {:>11} {:>10} {:>10} {:>8} {:>11} {}".format(
        b["phase"], str(b.get("step")), fmt(b.get("loss")),
        fmt(g(b,"emb","rho")), fmt(g(b,"L1","rho")), fmt(g(b,"lm_in","rho")),
        fmt(b.get("growth")), fmt(b.get("hrms_growth")),
        fmt(g(b,"lm_in","abs_dot")), fmt(g(b,"lm_in","signed")),
        fmt(g(b,"lm_in","centered")), fmt(g(b,"lm_in","meanrms")),
        fmt(g(b,"lm_in","muscale")), fmt(b.get("rho_atom")), fmt(b.get("rho_nonatom")),
        b.get("worst","")))
