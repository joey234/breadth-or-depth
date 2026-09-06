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
      Tables tab:info-matched-chain and tab:full-spec-chain, n=5 per cell
      except explicit-overall D=4 (n=10).

  probe/{mid,frontier}_results.json
      K=8 matched-probe draws, including MASSIVE and word D=4. Figure 1
      uses the locked values in scatter_data.json, not a re-derivation
      from these files. Old dchain* cells are archived only.

  scatter_data.json
      Locked Figure 1 points: tab:probe-full plus MASSIVE (n=10 paired)
      and word D=4 (n=10 information-matched total effect).

Reprint the tables:

  python experiment/summarize_results.py
