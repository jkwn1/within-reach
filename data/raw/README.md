# Raw data (not committed)

Put these files in this folder with exactly these names, then run `python run_all.py` from the repo root.

| File name here | What it is | Where to get it |
| --- | --- | --- |
| `indeed_job_listings.csv` | Indeed job postings, California, posted 2025-2026 (~2.1 GB) | Datathon shared drive, Indeed Jobs folder |
| `zillow_part1.csv`, `zillow_part2.csv`, `zillow_part3.csv` | Zillow Bay Area listings in 3 parts | Datathon shared drive, Zillow Housing folder (only ingested; not used in the final analysis) |
| `google_maps.csv` | Google Maps places (essential services) | Datathon shared drive, Google Maps Places folder (only ingested) |
| `google_maps_reviews.csv` | Google Maps reviews | Datathon shared drive, Google Maps Reviews folder (only ingested) |
| `college_results_2021.xlsx` | "College Results View 2021 Data Dump for Export.xlsx" | Datathon shared drive, College Affordability folder |
| `affordability_gap_2022_23.xlsx` | "Affordability Gap Data AY2022-23 2.17.25.xlsx" | Datathon shared drive, College Affordability folder |
| `tract_outcomes_simple.csv` | Opportunity Atlas: Household Income and Incarceration for Children from Low-Income Households by Census Tract, Race, and Gender | https://opportunityinsights.org/data/ |
| `tract_covariates.csv` | Opportunity Atlas: Neighborhood Characteristics by Census Tract | https://opportunityinsights.org/data/ |
| `census_tracts_list_06.txt` | 2010 Census Gazetteer, California census tracts | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/census_tracts_list_06.txt |
| `2010_place_list_06.txt` | 2010 Census Gazetteer, California places | https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2010_place_list_06.txt |

Notes
- Step 0 only needs the Zillow and Google Maps files because it ingests every provided dataset we
  screened. To skip them, run `python src/00_ingest.py gazetteer atlas college indeed`, then
  `python run_all.py 1 2 3 4 5 5b 6 7 8 9`.
- The Gazetteer parser also accepts text extracted from a printed (PDF) copy of those two pages, which
  is how our copy was saved; the original tab-delimited `.txt` files work the same way.
