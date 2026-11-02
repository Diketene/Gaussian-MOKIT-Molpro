#!/usr/bin/env python3
"""Gaussian external driver that talks to Molpro directly.

Differences from gmm.py:

* no PySCF, no MOKIT, no fchk round-trip;
* the geometry from Gaussian's external input file is written straight into a
  Molpro input, Molpro runs its own (RO)HF **in the same basis as the post-HF
  method**, so no orbital alignment is needed at all;
* only two programs are involved: Gaussian (optimizer / freq / IRC) and Molpro
  (energy + analytic gradient).

Requires only Python + numpy (no pandas).

Usage (route section of the Gaussian input):

    %nproc=1
    %chk=TS.chk
    #p opt=(ts,noeigen,nomicro) external="python3 -u /path/to/gmm_molpro.py" nosymm

and a gmm.json in the working directory:

    {
        "num_proc": 8,
        "mem": "16GB",
        "basis": "avtz-f12",
        "scf": "{df-hf;}",
        "post_hf": ["{ccsd(t)-f12,ansatz=3C(FIX)}"]
    }

`scf` is the SCF command itself (use `{hf;}` for conventional SCF, `{df-hf;}`
for density fitted SCF, which is much cheaper); do not put `{df-hf;}` in
`pre_hf` and also keep `{hf;}` in `scf`, that would run the SCF twice.
`pre_hf` is only for extra directives that must come before the SCF.

Notes
-----
* `derivs == 2` (analytic Hessian) is not implemented; use `freq=num` so that
  Gaussian builds the Hessian from external gradients.
* If Molpro cannot produce an analytic gradient for the requested method, the
  `grad.csv` file will be missing and the run aborts with a hint.
"""

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from lib import GauDriver

_INP_NAME = "mol_ext.com"
_OUT_NAME = "mol_ext.out"


@dataclass
class Config:
    num_proc: int = 4
    mem: str = "8GB"          # total memory, e.g. "16GB" / "8000MB"
    basis: str = "avtz-f12"   # Molpro basis name
    scf: str = "{hf;}"        # RHF for closed shell, ROHF for open shell
    pre_hf: List[str] = field(default_factory=list)   # directives before SCF
    post_hf: List[str] = field(default_factory=list)  # method(s) to run
    workdir: str = "tmp"
    keep_workdir: bool = False

    @classmethod
    def from_json(cls, filename: str = "gmm.json") -> "Config":
        with open(filename) as f:
            data = json.load(f)

        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown key(s) in {filename}: {sorted(unknown)}; "
                f"allowed: {sorted(known)}"
            )
        return cls(**data)


def _read_numeric_csv(filename: str) -> np.ndarray:
    """Read a file saved by Molpro's {table;save,...} command.

    Molpro writes Fortran style exponents (1.0D+00), a header line with the
    variable names, and comma or whitespace separated columns.
    """
    rows = []
    with open(filename, "r", errors="replace") as f:
        for line in f:
            line = line.strip().replace("D", "E").replace("d", "e")
            if not line:
                continue
            fields = line.replace(",", " ").split()
            try:
                rows.append([float(x) for x in fields])
            except ValueError:
                continue  # header / comment line

    if not rows:
        raise RuntimeError(f"no numeric data found in {filename}")
    return np.array(rows)


def _mem_to_mw(mem: str, num_proc: int) -> int:
    """Convert total memory to Molpro's per-process megawords."""
    if "GB" in mem:
        mem_mb = int(float(mem.replace("GB", "")) * 1024)
    elif "MB" in mem:
        mem_mb = int(float(mem.replace("MB", "")))
    else:
        raise ValueError(f"unsupported memory setting: {mem}")

    # 1 word = 8 bytes, Molpro's memory is per process
    return max(1, mem_mb // (8 * num_proc))


def build_input(config: Config, gau: GauDriver) -> str:
    lines = [
        "***, Gaussian-Molpro external driver",
        f"memory,{_mem_to_mw(config.mem, config.num_proc)},m",
        # keep the frame Gaussian uses; no symmetry, no reorientation
        "symmetry,nosym,noorient",
        "geometry={",
    ]

    # Gaussian gives coordinates in bohr, which is Molpro's default unit
    for i, (sym, xyz) in enumerate(zip(gau.symbols, gau.coords), start=1):
        lines.append(
            f"{sym.strip()}{i:<4d}"
            f"{xyz[0]:>22.14f}{xyz[1]:>22.14f}{xyz[2]:>22.14f}"
        )
    lines.append("}")

    # Molpro's `spin` is 2S, Gaussian's multiplicity is 2S+1
    lines.append(f"wf,charge={gau.charge},spin={gau.multiplicity - 1}")
    lines.append(f"basis={config.basis}")

    lines.extend(config.pre_hf)
    lines.append(config.scf)
    lines.extend(config.post_hf)

    if gau.derivs >= 1:
        # varsav keeps the wavefunction, needed by the following {table}
        lines.append("{force;varsav;}")

    lines.append("{table,energy;save,energy.csv,new;}")
    if gau.derivs >= 1:
        lines.append("{table,gradx,grady,gradz;save,grad.csv,new;}")

    return "\n".join(lines) + "\n"


def main() -> None:
    config = Config.from_json()
    gau = GauDriver.from_stdio()

    if gau.derivs == 2:
        raise NotImplementedError(
            "analytic Hessian (derivs=2) is not implemented; "
            "use freq=num in the Gaussian route section"
        )

    workdir = os.path.abspath(config.workdir)
    os.makedirs(workdir, exist_ok=True)

    # stale results must never be picked up as fresh ones
    for name in ("energy.csv", "grad.csv", _OUT_NAME, "molpro.stdout"):
        path = os.path.join(workdir, name)
        if os.path.exists(path):
            os.remove(path)

    with open(os.path.join(workdir, _INP_NAME), "w") as f:
        f.write(build_input(config, gau))

    cmd = (
        f"molpro -s -n {config.num_proc} -t 1 "
        f"-m {_mem_to_mw(config.mem, config.num_proc)}m {_INP_NAME}"
    )
    with open(os.path.join(workdir, "molpro.stdout"), "w") as f_out:
        proc = subprocess.run(
            args=shlex.split(cmd),
            cwd=workdir,
            stdout=f_out,
            stderr=subprocess.STDOUT,
        )

    energy_csv = os.path.join(workdir, "energy.csv")
    grad_csv = os.path.join(workdir, "grad.csv")

    if not os.path.exists(energy_csv):
        raise RuntimeError(
            f"Molpro failed (exit code {proc.returncode}); "
            f"input: {os.path.join(workdir, _INP_NAME)}\n"
            + _tail(os.path.join(workdir, _OUT_NAME))
        )

    energy = float(_read_numeric_csv(energy_csv)[0, 0])

    gradients: Optional[np.ndarray] = None
    if gau.derivs >= 1:
        if not os.path.exists(grad_csv):
            raise RuntimeError(
                "Molpro produced no gradient; the requested method probably has "
                "no analytic gradient (for F12 try ansatz=3C(FIX)).\n"
                + _tail(os.path.join(workdir, _OUT_NAME))
            )
        gradients = _read_numeric_csv(grad_csv).reshape(-1, 3)
        if gradients.shape[0] != gau.natom:
            raise RuntimeError(
                f"gradient has {gradients.shape[0]} rows but the molecule has "
                f"{gau.natom} atoms"
            )

    gau.write(
        energy=energy,
        gradients=gradients,
        force_constants=None,
    )

    if not config.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)


def _tail(filename: str, n: int = 60) -> str:
    if not os.path.exists(filename):
        return f"(no Molpro output file {filename})"
    with open(filename, "r", errors="replace") as f:
        return "".join(f.readlines()[-n:])


if __name__ == "__main__":
    main()
