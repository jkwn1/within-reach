# Hand checks (do these yourselves, by hand)

The Advanced Track requires checking a sample of our data decisions by hand. Each sheet is a random
sample (fixed seed, so it is the same every run). Fill the empty columns with `y` or `n` and a note
when something is wrong. Report the counts on the data-decisions slide (e.g. "20/20 rows match").

Time needed: about 50 minutes for both of you. Split the sheets; each of you also re-checks 5 rows
of the other's sheet, so you can say how often you agreed.

## 1. `college_rows_20.csv` (about 15 min)
Open `College Results View 2021 Data Dump for Export.xlsx` and `Affordability Gap Data AY2022-23 2.17.25.xlsx`.
For each row, search the first file for the `unit_id` (column `UNIQUE_IDENTIFICATION_NUMBER_OF_THE_INSTITUTION`):
- `earnings_matches_excel`: does "Median Earnings of Dependent Students Working and Not Enrolled 10 Years After Entry" equal `earnings_dep_10yr`?
- `pell_matches_excel`: does "Percent of First-Time, Full-Time Undergraduates Awarded Pell Grants" equal `pell_share`?
- `msi_flags_match`: does the HBCU column match `HBCU`? In the second file (search `Unit ID`), does the HSI column match `HSI`?
- `net_price_matches_excel`: in the second file, on the row where "Student Family Earnings Ceiling" is 30000, does "Net Price" equal `net_price_low_income`? (Blank in our sheet = college not in the second file.)

## 2. `postings_30.csv` (about 20 min)
For each Indeed posting:
- `pay_correct`: is `pay` the midpoint of `salary_formatted`, converted to a year (hourly x 2,080, monthly x 12, weekly x 52, daily x 260)? Blank pay is correct when there is no salary or it is per mile/session.
- `city_correct`: is `city` the city in `location`?
- `entry_correct`: is it entry-level (no senior/lead/manager title, and at most 1 year of experience asked)? Open the posting text in the raw CSV by `jobid` if needed.
- `degree_correct`: does the description ask for a bachelor's degree? Write down if it is only "preferred" (we count that as required: a known limitation).

## 3. `tracts_10.csv` (about 5 min)
Open each `google_maps_link`. `inside_named_county`: is the point inside the county with that FIPS
code? (1 Alameda, 13 Contra Costa, 41 Marin, 55 Napa, 75 San Francisco, 81 San Mateo, 85 Santa Clara, 95 Solano, 97 Sonoma)

## 4. `also_consider_pairs_15.csv` (about 10 min)
These are "also consider" suggestions the tool makes (a label we created). For each row, look up the
suggested college (`option_name`, `option_city`) on its own website or Wikipedia:
- `same_degree_level`: do both colleges mainly award the same kind of degree (bachelor's vs associate's/certificate)? (`level` is our label for the starting college.)
- `still_open`: is the suggested college still open and admitting students (not closed or merged)?
- `coed_and_in_person`: does it admit all genders and teach mainly in person?
- `similar_programs`: does it offer broadly similar programs (not a single-field school, e.g. only art or only nursing)?
Write `y`/`n` and a short note when `n`. Report it like "13/15 pairs passed all four".
