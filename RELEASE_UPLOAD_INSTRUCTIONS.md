# Release package upload instructions

1. Download `VQLS_AC_Power_Flow_GitHub_Ready.zip` from the associated ChatGPT conversation and extract its contents.
2. In a terminal, run:

```bash
git clone https://github.com/divinelybless/VQLS-Corrections-for-AC-Power-Flow.git
cd VQLS-Corrections-for-AC-Power-Flow
# Copy the *contents* of the extracted release folder here, not the enclosing directory.
git add .
git commit -m "Deposit fixed-seed AC power-flow reproducibility archive"
git push origin main
```

3. Check that `archived_study/results.json`, `independent_verification/independent_decimal_krawczyk.py`, `supplement/trial_level_geometry.csv`, and `metadata/claim_manifest.json` are visible in the repository. Compare SHA256 checksums with `metadata/SHA256SUMS.txt`.
4. Replace any draft Data Availability wording with the exact published release URL or commit SHA. For a DOI, archive a tagged GitHub release through Zenodo.

Do not upload passwords, tokens, author private review correspondence, or copyrighted material lacking redistribution permission. A separate external audit remains outstanding.
