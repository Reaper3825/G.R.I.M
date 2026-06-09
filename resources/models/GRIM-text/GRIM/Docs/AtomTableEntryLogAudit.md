# AtomTable Entry Log Audit

## Current Run

* **Log file:** `resources/models/GRIM-text/training/logs/tokenizer.log`
* **Session ID:** `17806075746220244`
* **Started:** `2026-06-04 17:12:54`
* **Scope:** AtomTable registration, numeric atom spans, and `arg_number` population.

## Main Observations

* The log reports:

  * `864776` atoms detected across `60000` texts.
  * `21` unparseable spans were treated as text.
  * `atom_reasoning=on`.
* Byte fallback still contains punctuation that may belong to numeric structures:

  * `.` appears in byte fallback.
  * `-` appears in byte fallback.
* The tokenizer spec says detector-emitted numeric spans should throw if AtomTable registration fails.
* The current log shows some invalid numeric spans are being kept as text instead.

## Issues Found

### Implementation Update — 2026-06-09

Issues **1**, **2**, and **3** are implemented in the tokenizer code path:

* `createAtomTableFromRawTextDetections()` now pre-validates detector-emitted `ATOM_INT` / `ATOM_FLOAT` spans through `AtomTable::parseAtom()` and throws with detector name, raw text, byte offsets, atom type, and parse reason before any text fallback can occur.
* `arg_number` population now handles signs, decimals, leading-dot floats, and scientific notation by recording mantissa digit bindings plus sign, decimal, and exponent metadata.
* `malformed_numbers` is no longer used as a quiet continuation path for detector-emitted numeric atoms. If a numeric atom cannot populate the required side channel, the AtomTable creation boundary throws.

### 1. Numeric spans are being silently downgraded to text

* The log says `21` unparseable spans were treated as text.
* The tokenizer contract says numeric detector spans should fail loudly instead.
* This means bad detector output can slip through instead of exposing the real bug.

**Required behavior:**

* If a detector marks something as `ATOM_INT` or `ATOM_FLOAT`, and AtomTable cannot register it, the pipeline should throw.
* The error should include:

  * detector name
  * raw text
  * byte offsets
* Do not send failed numeric spans back into the plain text path.

Bad examples that should fail:

* `arg_number`
* standalone `-`
* standalone `+`
* alphabetic text mislabeled as numeric

### 2. `arg_number` only accepts digits

* `appendArgNumberFromSpan()` rejects anything that is not `0` through `9`.
* But AtomTable passes the full detected numeric span.
* Numeric detectors allow signs and floats.

This means these can register as atoms but fail `arg_number`:

* `-4`
* `+4`
* `.75`
* `75.0`
* `1.48`
* `1e6`
* `-1.5e-4`

**Required behavior:**

* Either make `arg_number` support signs, decimals, and exponents.
* Or keep `arg_number` digit-only and pass only the digit content span.
* Do not quietly increment `malformed_numbers` and continue when `arg_number` is required.

### 3. Leading-dot floats lose their `arg_number`

Examples from the corpus:

* `.75`
* `.75 kilograms`

Likely current behavior:

1. Float detector sees `.75`.
2. AtomTable registers it as a float.
3. `arg_number` sees `.`.
4. `arg_number` rejects it.
5. The float atom exists, but the numeric decomposition is missing.

**Required behavior:**

* `.75` should either:

  * get a real float-aware numeric side channel, or
  * fail loudly as unsupported.
* It should not register successfully while silently losing `arg_number`.

### 4. The `-` sign is inconsistent

* Detectors allow signed numbers.
* AtomTable passes the sign as part of the numeric span.
* `arg_number` rejects the sign.
* Some unparseable spans are currently kept as text.

This creates inconsistent handling for signed numbers.

**Required behavior:**

* For detector-emitted numeric atoms, leading `-` or `+` must be handled directly.
* Either:

  * store the sign separately from the digit-only `arg_number`, or
  * throw if signs are unsupported.
* Do not let signed numeric spans partially succeed.

### 5. `arg_number` text should never reach the numeric path

* If the literal text `arg_number` reaches AtomTable as a numeric atom, that is an upstream labeling bug.
* It should not be treated as malformed user data.
* It should fail immediately.

**Required behavior:**

* Any alphabetic text marked as `ATOM_INT` or `ATOM_FLOAT` should throw.
* The error should include enough context to find the bad source span.

## Contract Drift

The current system is inconsistent:

* **Spec:** Numeric detector spans must throw on registration failure.
* **Log:** `21` unparseable spans were treated as text.
* **Implementation:** `arg_number` can fail by incrementing `malformed_numbers` and continuing.

This is silent downgrade behavior and should be removed.

## Recommended Policy

1. Do not silently fallback to text for detector-emitted numeric spans.
2. Do not silently continue when required `arg_number` data is missing.
3. Standalone `-` or `+` should fail if routed as numeric.
4. Alphabetic text on a numeric path should fail immediately.
5. Split numeric representation into:

   * raw span
   * digit-only content span
   * sign metadata
   * decimal metadata
   * exponent metadata

## Next Checks

* Log all `21` unparseable spans with raw text and offsets.
* Decide what `arg_number` is supposed to support:

  * unsigned integers only, or
  * signed / float / exponent numbers.
* If `arg_number` stays digit-only, create digit-only content spans before calling it.
* Throw on unsupported numeric forms instead of continuing silently.
