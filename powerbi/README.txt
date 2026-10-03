POWER BI IMPORT GUIDE (real Home Credit data)
Get Data > Text/CSV, one file per table. Relationships (many-to-one, single direction):
  fact_loans[score_band]            -> dim_score_band[score_band]
  fact_el_by_band[band]             -> dim_score_band[score_band]
Notes
  - fact_loans.csv is NOT in the repository (loan-level data from Kaggle). Run dashboard/build_powerbi.py to create it. It has all 307,511 applicants (dataset = train / test). pd_12m, lgd, ead, expected_loss, expected_loss_severe are filled only for the
    92,254 hold-out loans (dataset = test), viewed at origination.
  - fact_simulated_* tables are SIMULATED months (S01-S12). They are not real history; label any visual that uses them.
  - Sort score_band by band_order.
Starter measures (DAX)
  Loans          = COUNTROWS(fact_loans)
  Expected Loss  = SUM(fact_loans[expected_loss])
  Severe EL      = SUM(fact_loans[expected_loss_severe])
  EAD            = SUM(fact_loans[ead])
  EL Rate        = DIVIDE([Expected Loss], [EAD])
  Avg PD 12m     = AVERAGE(fact_loans[pd_12m])
  Avg PD life    = AVERAGE(fact_loans[pd_scorecard])
  Actual DR      = AVERAGE(fact_loans[defaulted])
Charts: EL by band (bar), stress scenarios (bar), LGD x CCF sensitivity (matrix), simulated predicted vs actual by month (line), simulated PSI by month (line + 0.10 line).
PSI status: <0.10 Stable, 0.10-0.25 Watch, >0.25 Major shift.
