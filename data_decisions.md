# Data decisions

Every cleaning rule, join and label we made, why, and how we checked it. Numbers come from
`results/college_data_quality.json`, `results/act1_summary.json` and `results/key_numbers.md`.

## College data (the main analysis)

**1. The all-students earnings column is unusable, so the target is dependent-student earnings.**
"Median Earnings of Students Working and Not Enrolled 10 Years After Entry" has only **3 distinct
values** ($27,937, $36,041 and $47,921.50) across 5,831 colleges in this export. A column with three
values cannot tell colleges apart, and a model trained on it looks accurate for the wrong reason (an
early model trained on it scored suspiciously well, which is how we noticed). We use "Median
Earnings of Dependent Students Working and Not Enrolled 10 Years After Entry" instead: 4,194 colleges,
3,350 distinct values. Dependent students are mostly traditional-age students supported by their
families, the group whose path out of a low-income childhood this project is about.

**2. Campuses that share one federal reporting unit are kept together.** Federal earnings data are
reported per reporting unit, not per campus, so the campuses of a chain or system carry copies of the
same figures. We define a reporting unit as colleges with identical dependent-student earnings, AND
identical independent-student earnings, AND identical median dependent debt. (Matching on earnings alone
grouped 1,221 colleges, but an independent review estimated that hundreds of those matches were
coincidences, e.g. SF State and Norwich University both at $59,200; requiring all three figures to match
separates them.) Three
consequences: every split (train/test, every cross-validation fold) keeps a unit on one side, so the
model cannot "predict" a campus by memorizing its sibling; when fitting, each unit counts once (a college
in a unit of k gets weight 1/k); and colleges in multi-college units are left out of every ranking, the
bottom-20% comparison and the opportunity list, because their figure describes the whole chain or system,
not the campus. Counts are in `results/key_numbers.md`.

**3. The affordability table has up to 5 rows per college; we keep the lowest-income one.** Rows differ by
family-income ceiling ($30k, $48k, $75k, $110k, $150k). We keep the $30,000 row, one per college
(checked: unique). 3,742 of 4,194 colleges (89.2%) match on Unit ID. For matched colleges the
sector code agrees between the two files 98.9% of the time, and the HBCU flag agrees 100%.

**4. Minority-serving designations.** HBCU, Tribal (TCU), Predominantly Black (PBI) and AANAPISI flags
come from the results file (97% filled). The HSI flag is empty there, so it comes from the
affordability file, read from any of a college's rows (an earlier version read only the $30k row and
missed 21 HSIs). Colleges absent from that file count as not HSI (a small undercount).

**5. Model inputs describe who enrolls and where, never what the college produces.** Inputs: share of
Pell (low-income) students, race/ethnicity shares, share of women, of students 25+, part-time,
distance-only and full-time first-time students, enrollment, admission rate, SAT/ACT 25th
percentiles, open admissions, sector, Carnegie type, urbanization, size and state. Excluded on
purpose: graduation and retention rates, debt, spending, net price and program mix. Value-added
should credit a college for those, not explain them away. The Carnegie codes in this file are the 2000
classification, whose codes 51-59 name the field of a specialized school (medicine, engineering, art,
law...). That is program mix, so those nine codes are merged into one "specialized school" code. Code 60
(tribal colleges) is kept, like the race/ethnicity inputs (decision 6).

**6. Race/ethnicity inputs: included, with a sensitivity check.** With them, a college is compared with
colleges serving similar students. Without them, colleges serving Black, Latino or Native students are
blamed for earnings gaps created in the labor market. We report value-added both ways
(`results/race_sensitivity.csv`).

**7. Rankings use colleges with at least 500 students and their own earnings figure.** Earnings medians
for very small colleges are noisy (one of the largest test errors is a college with 8 students).

## Indeed and Opportunity Atlas (Act 1)

**8. Salary text to annual pay.** Postings are dated 2025-01-01 to 2026-09-21. We take the midpoint of the first two dollar amounts and annualize
by the stated unit: hour x 2,080, day x 260, week x 52, month x 12, year x 1, including Spanish units
(por hora, por mes, por año, por semana, por día). Pay per mile, session, class or visit is dropped.
Results outside $15,000-$600,000 are dropped. Hourly pay is annualized at full-time hours, so this is
full-time-equivalent pay. 84.9% of postings end up with pay.

**9. Duplicates.** Postings with the same title, company and location are dropped (140,816 -> 136,136).

**10. Location to place.** The city is the text right before ", CA" (works for "City, CA",
"City, CA 95051" and "2788 San Tomas Expressway, Santa Clara, CA 95051"). County-level and
"San Francisco Bay Area" locations are dropped. Names are matched to the 2010 Census Gazetteer of
California places; a name used by several places (e.g. Burbank city in LA County and Burbank CDP in
Santa Clara County) goes to the most populous. 93.8% of postings match. Some postings give a street
address, but every posting is placed at its place's Gazetteer point (decision 14). A posting is "Bay Area" if
its place's nearest tract centroid is in one of the 9 counties and within 10 miles. Known miss: "El
Sobrante" (13 Bay Area postings) goes to the slightly larger El Sobrante in Riverside County.

**11. Entry-level and degree flags.** Entry-level = the title has no senior/lead/manager-type word
("staff" only in "staff engineer"-type titles, so "kitchen staff" stays in), is not a licensed profession
(physician, dentist, nurse practitioner, physician assistant, pharmacist, attorney...), and the
description asks for at most 1 year of experience or names none. The experience pattern skips
"21 years of age", and values over 15 years are treated as company boilerplate ("over 50 years of
experience serving clients"). Degree = the description mentions a bachelor's degree (this also catches
"preferred", so it is an upper bound on "required").

**12. Spatial statistics.** Neighbouring tracts share almost the same job pool, so they are not
independent observations. Every p-value in Act 1 uses standard errors clustered on ~10-mile grid cells
(0.15 degrees), and correlations are reported without p-values. All three radii (5, 10, 20 miles) are
reported, not only the one that tells the cleanest story. The Atlas outcomes describe kids born
1978-83; the postings are from 2025-26. Act 1 compares today's job map with an older generation's
outcomes: a description, not a causal test.

**13. Tracts.** Opportunity Atlas tracts (2010 boundaries) in the 9 counties, joined to the 2010
Census Gazetteer for each tract's internal point. 1,578 tracts; 1,568 with an outcome.
The Gazetteer text was extracted from a printed copy of the Census page; every record parsed
(8,057 tracts, which is California's 2010 tract count, and all 1,523 places in the file).

**14. San Francisco's location (found by an independent review).** The Gazetteer's internal point for
San Francisco lies offshore near the Farallon Islands, because the city's boundary includes 185 square
miles of water and only 47 of land. An earlier version used that point, so every San Francisco posting,
the region's largest job centre, fell outside every tract's job pool. We now place San Francisco at City
Hall (37.7793, -122.4193). This changed Act 1: entry-level pay within 10 miles is now higher near
high-mobility neighborhoods ($57,200 in the lowest fifth vs $67,500 in the highest), where the earlier
version showed it as about flat. Beyond poverty and college share, it still adds little (R² +0.006 at 10
miles, p = 0.31; +0.016 at 20 miles, p = 0.01). The review found no other high-volume place placed this way.

## Choosing a college (step 5b)

**15. "Similar students."** Among ranked colleges with a Pell share at or above the median (43%), we
compare colleges whose expected earnings are within 10% of the median for that group ($37,032), and
report results by value-added quarter. Prices are reported as they are: close overall, but at colleges
with 2,000+ students the best quarter costs more ($2,638 at the median), so we never call them "the same
price". The dollar gap is a spread between colleges, not a gain any student is promised.

**16. "Also consider" options (a label we created).** An option must be within 25 miles (straight line;
half of community-college students enroll within 10 miles of home and 79% within 25, Hillman/TICAS 2023),
the same level, the same kind of institution, have expected earnings within 10%, a Pell share at most 10
points lower, a net price for families earning up to $30,000 no higher, admission no harder (open
admissions, or an admission rate at most 10 points lower), and value-added at least 10 points higher both
with and without race inputs. The nearest qualifying option is shown first, so the gap is not inflated by
picking the most extreme college. Level: "bachelor's" means at least 50 bachelor's degrees a year (a review
found a 4-year-labelled community college awarding almost no bachelor's degrees, and the file's Carnegie codes
are from 2000, too old to tell); otherwise 2-year, or under-2-year by sector.

**17. Safeguards from reviewing the suggested pairs by name.** Tribal colleges (the model is biased for
them), specialized colleges (Carnegie specialized code, plus three by name), online-by-design colleges (a
hand list: a distance-education share can't be used because the intake data cover a pandemic year) and
colleges known to have closed or merged since the data were collected (a hand list, matched on exact names)
are never starting points or options. Single-gender colleges are never options. Below bachelor's level,
colleges without a Carnegie classification (mostly technical centres and career schools) are only compared
with each other. The lists are in
`src/05b_choice.py` and may be incomplete; the tool tells users to check that a college is still open.

**18. Testing the suggestions on other students.** Independent-student earnings (mostly older students)
are not a model input or part of the rules. A review showed that comparing all suggested pairs with all
starting colleges' nearest comparable college mixes different starting colleges, so the main test uses the
same starting colleges: for the 55 that have both a stronger option and another comparable college,
independent students earn more at the nearest stronger option 80% of the time vs 58% at the nearest other
comparable college (difference +9 to +36 points, bootstrap over starting colleges). Dependent and
independent earnings move together (71% benchmark), so this shows the gaps aren't just noise, not cause.

**19. Reach is counted in students, not dollars.** We count first-time, full-time Pell recipients (an
exact count in College Results), and never multiply an earnings gap by students and years.

**20. Pell group in the bottom-20% comparison.** Pell terciles are cut on all 4,194 colleges, so among
the 2,496 colleges in that comparison the "top third" is really a Pell share above 61%; the group is
labelled that way. Parity for this group, and largely for HBCUs and PBIs, is partly built in, because
Pell share and race shares are model inputs; the version without race inputs is the more informative one.

## Hand checks

Random samples (fixed seed) are in `handcheck/` with instructions in `handcheck/README.md`:
20 college rows against the raw Excel files, 30 Bay Area postings (pay, city, flags), and 10 tract
locations on a map, plus 15 "also consider" pairs. The team hand-checked them against the raw files; a second, independent re-check of every row (recorded in the sheets) found: 20/20 college rows, 27/30 postings (3 descriptions ask for a "BS" our degree flag misses: an undercount of degree mentions), 10/10 tracts, 12/15 pairs (3 suggested colleges have a narrower program mix).
