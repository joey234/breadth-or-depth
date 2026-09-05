Saved run outputs used by the paper tables.

  gepa_suite/<tier>_<task>_acceptall/study_results.json
      Table tab:gepa. Accept-all / select-all paired S2 vs S3, n=10.

  gepa_suite/<tier>_<task>_strict/study_results.json
      Table tab:gepa-replication-ci. Official GEPA acceptance, n=10.

  paired/<tier>_<task>_acceptall/results.json
      Tables tab:real-boundary and tab:boundary-details, n=10.

  paired/<tier>_{massive,structured}_strict/results.json
      Appendix strict-improvement supplements (MASSIVE n=3; structured n=3).

  chain/{mid,frontier}_results.json
      Tables tab:info-matched-chain and tab:full-spec-chain, n=5 per cell.

  probe/{mid,frontier}_results.json
      K=8 matched-probe draws. Figure 1 / tab:probe-full use the locked
      values in scatter_data.json, not a re-derivation from these files.

  scatter_data.json
      Locked Figure 1 points, matching tab:probe-full.

Reprint the tables:

  python experiment/summarize_results.py
