# Within Reach

**Helping low-income students find nearby, affordable colleges where students like them earn more.**

Jian Kweon, Freya Sutedja · 8th Annual Datathon for Social Good · Advanced Track

- **Tool:** `docs/index.html`, the Within Reach College Finder (one self-contained page; open it in any browser, or publish it with GitHub Pages from the `docs/` folder).
- **Numbers:** every result below comes from `results/key_numbers.md`, which the pipeline writes. Outside facts (Census, TICAS, NASFAA) are cited where they appear.

## The problem

Low-income students mostly choose among colleges close to home that they can afford: half of community-college students enroll within 10 miles of home and 79% within 25 ([Hillman, TICAS 2023](https://ticas.org/wp-content/uploads/2023/11/HIllman-Geography-of-Opportunity-Brief-2_2023.pdf)). What they usually see, price and earnings rankings, doesn't tell them which of those nearby colleges pays off for students like them. Example: East Los Angeles College and Glendale Community College are 10 miles apart and enroll similar students (expected earnings $37,456 and $39,315). For a family earning up to $30,000, Glendale costs $6,488 a year against $9,023, and its former students earn $44,076 ten years after entry against $37,578.

**Who it's for:** counselors at high-poverty high schools, college-access advisers, and students choosing where to enroll or transfer. **What they do with it:** look up each college on a student's list, add any suggested option to the application and FAFSA list, and compare aid offers.

## What we found

1. **Colleges with similar low-income students get very different results.** Among 555 colleges where at least 43% of students get Pell grants and the model expects similar earnings (within 10% of $37,032), median earnings run from $33,211 in the lowest value-added quarter to $40,751 in the highest, a $7,540 spread. At colleges with 5,000+ students, where earnings figures are least noisy, the spread is $5,561. This is a spread between colleges, not a gain promised to any student; prices are close overall but not at larger colleges (decision 15).
2. **Families can't see it.** Among high-Pell colleges, the low-income net price tracks raw earnings (rank correlation 0.38) but not value-added (0.09). In 521 local, affordable choice sets, the college with the highest raw earnings is also the highest value-added one only 48% of the time. The top 20% of colleges by raw earnings are 13% high-Pell colleges, against 50% of all ranked colleges.
3. **About 17,600 first-time Pell students a year start at a college with a stronger option nearby.** Of 1,146 high-Pell colleges compared, 45 (17,567 first-time, full-time Pell students a year) have a college within 25 miles with similar students, the same or lower price, no harder admission and value-added at least 10 points higher with and without race inputs. The nearest such option is a median 11.1 miles away, $4,475 cheaper, and its former students earn $6,239 more a year. 177 colleges (85,633 students) have nearby colleges with similar students but none stronger by this rule; 426 (110,041 students) have no other rated college of the same level within 25 miles. For most low-income students, then, the honest answer is that nothing nearby beats their college, which is a case for investing where they are.
4. **A different group of students points the same way.** Independent students (mostly older) are not a model input or part of the rules. For the same 55 starting colleges, they earn more at the nearest stronger option 80% of the time, against 58% at the nearest other comparable college (difference +9 to +36 points, 95% interval). Across all 151 suggested pairs it is 78%, and 71-83% across 15/25/50-mile radii and 5/10/15-point gaps. Dependent and independent earnings move together (when dependent earnings are higher, independent earnings are too 71% of the time), so this shows the gaps aren't just noise, not that the colleges cause them.
5. **Raw-earnings yardsticks push colleges that serve low-income students to the bottom.** Of the 82 HBCUs with 500+ students and their own earnings figure, 50% are in the bottom fifth by raw earnings; by value-added 23%, or 32% without race inputs. For predominantly Black institutions: 69%, 16%, 21%.
6. **Student parents can't close the gap with work.** Even at the 75 "opportunity colleges" (high value-added, many Pell students, low price), a student parent paying for center-based child care needs about 50 hours of minimum-wage work a week in total to cover costs, against about 13 for a student without children.
7. **Where we started, jobs near home (Act 1).** Joining 45,312 Bay Area Indeed postings to the Opportunity Atlas: entry-level pay posted within 10 miles is higher near high-mobility neighborhoods ($67,500 median in the highest-mobility fifth vs $57,200 in the lowest), but that mostly tracks neighborhood poverty and education. Beyond those, nearby pay adds little to predicting how kids fared (R² +0.006 at 10 miles, clustered p = 0.31; +0.016 at 20 miles, p = 0.01). Postings that mention a bachelor's degree pay 19% more in the same job family and city (95% CI 15-23%). So we focused on which college.

## What to do with it

- **Students and counselors:** look up every college on the list in the finder; consider any "stronger option nearby"; check programs, admission, transfer rules and aid offers before deciding.
- **Colleges:** learn from nearby peers that do better with similar students.
- **Policymakers:** publish intake-adjusted results beside raw earnings, including beside the federal earnings test (program-level, graduates only; first results July 2027, [NASFAA](https://www.nasfaa.org/uploads/documents/OB3_Accountability_Changes_Brief_for_Campus_Leadership.pdf)); invest where students have no alternative nearby; fund child care for student parents.

## The tool: Within Reach College Finder

`docs/index.html`, built by `src/09_finder.py` from the pipeline's results (2,408 colleges). Look up a college to see its price for families earning up to $30,000, former students' earnings against what the model expects for similar students, Pell students' 6-year completion, minority-serving status, and one answer: **stronger option nearby** ("also consider"), **no stronger option nearby**, **no comparable college within 25 miles**, or **not rated** (tribal, specialized, online-by-design or closed colleges). Lists are sorted by name. Suggestions are labeled "also consider". A browse view filters by state, level, price and city, and lists the 75 opportunity colleges nationwide.

## How it works

| Step | Script | What it does |
| --- | --- | --- |
| 0 | `src/00_ingest.py` | Reads every raw file once into small Parquet files |
| 1 | `src/01_job_access.py` | Act 1: salary text to annual pay, entry-level and degree flags, postings within 5/10/20 miles of each Bay Area tract, tests against the Atlas with spatially clustered errors |
| 2 | `src/02_college_prepare.py` | One row per college: target, intake inputs, reporting units, affordability join, minority-serving flags |
| 3 | `src/03_college_model.py` | Baselines, tuned gradient-boosted trees, held-out test set, bootstrap intervals, cross-fitted value-added |
| 4 | `src/04_college_fairness.py` | Errors by group, who lands in the bottom 20%, race-input sensitivity, worst predictions |
| 5 | `src/05_opportunity.py` | Ranked colleges, opportunity colleges, student-parent gap |
| 5b | `src/05b_choice.py` | Similar-students spread, "also consider" options, independent-student test, reach |
| 6 | `src/06_figures.py` | Every chart (`figures/`, slide versions in `figures/slides/`) |
| 7 | `src/07_handcheck_sheets.py` | Random samples for the manual checks in `handcheck/` |
| 8 | `src/08_key_numbers.py` | `results/key_numbers.md` |
| 9 | `src/09_finder.py` | `docs/index.html`, the finder |

### The model

- **Target:** median earnings of dependent students on federal aid, working and not enrolled, 10 years after entry (completers or not), from College Results View 2021. The all-students column has only 3 distinct values across 5,831 colleges, so it can't be used (`data_decisions.md` #1).
- **Inputs:** who enrolls and where: Pell share, race/ethnicity shares, share of women, of students 25+, part-time, distance-only and first-time full-time students, enrollment, admission rate, SAT/ACT, open admissions, sector, Carnegie type (specialized fields merged), urbanization, size, state. No outcomes (graduation, debt), spending, price or program mix.
- **Split:** trained on 3,363 colleges and tested on 831 held-out colleges (698 reporting units). Campuses that share one federal earnings figure always stay on the same side, and each unit counts once when fitting.
- **Tuning:** 108 settings by 5-fold grouped cross-validation on training data only. The test set was never used for tuning or model choice. (An earlier development run, before the reporting-unit definition was fixed, scored a different split.)
- **Value-added:** actual earnings vs the model's expectation, from cross-fitted predictions for all 4,194 colleges (each college predicted by models that never saw it or its sibling campuses), averaged over 5 fold assignments.

| Model (held-out test, 831 colleges) | Average error | 95% interval | R² (log earnings) |
| --- | --- | --- | --- |
| National median | $10,782 | $9,931-$11,623 | -0.03 |
| Pell share + sector, linear | $6,908 | $6,347-$7,506 | 0.54 |
| All inputs, linear (ridge) | $4,981 | $4,566-$5,413 | 0.72 |
| **Our model, gradient-boosted trees** | **$4,467** | **$4,115-$4,852** | **0.78** |

Intervals come from a bootstrap that resamples whole reporting units. Our model's error is $2,441 below the Pell + sector baseline (95% interval $2,000-$2,886) and $514 below ridge ($232-$817).

### Fairness and error analysis

| Group (colleges with 500+ students and their own figure, n = 2,496) | Bottom 20% by raw earnings | by value-added | by value-added, no race inputs |
| --- | --- | --- | --- |
| HBCUs (n=82) | 50% | 23% | 32% |
| Predominantly Black institutions (n=67) | 69% | 16% | 21% |
| Pell share above 61% (n=400) | 57% | 24% | 24% |
| Tribal colleges (n=11) | 91% | 36% | 55% |
| Hispanic-serving institutions (n=462) | 16% | 18% | 16% |
| AANAPISIs (n=163) | 8% | 23% | 15% |

Parity near 20% is partly built in for groups defined by model inputs (Pell share, race shares), so the column without race inputs is the more informative one (`data_decisions.md` #20).

- **Bias by group (cross-fitted, all 4,194 colleges):** the mean percentage error is within ±3% for every minority-serving designation, sector and Pell third, except tribal colleges (+9.3%, n=23: the model expects more than their former students earn, so their value-added may be understated) and colleges in U.S. territories (+4.0%, n=118). The tool never rates or compares tribal colleges.
- **Accuracy by group:** the typical miss (median absolute error) is 4.9% at public 2-year and 4-year colleges, 7.3% at private nonprofit 4-year, and 11.5-13.9% at private 2-year, under-2-year and for-profit 4-year colleges.
- **Held-out test set (smaller groups, noisier):** HBCUs +6.2% (n=21), PBIs +5.0% (n=14), private 2-year +10.5% (n=97), for-profit 4-year +5.6% (n=53), top Pell third +5.7% (n=264).
- **Where the model fails:** specialized health and law schools (Albany College of Pharmacy: $118,349 actual vs $67,930 predicted; Samuel Merritt; SUNY Downstate), because the model never sees field of study; fully online universities for mostly adult students (Capella, American Public University System); campuses sharing one figure (3 of the 15 largest test errors are UConn branch campuses sharing $72,439); and tiny colleges (ASPIRA City College, 8 students). It also over-predicts some, e.g. National University of Health Sciences. See `results/worst_test_predictions.csv`.

### "Also consider" rules (step 5b)

Within 25 miles; same level (bachelor's, 2-year, under 2-year) and kind of institution; expected earnings within 10%; Pell share at most 10 points lower; net price for families earning up to $30,000 no higher; admission no harder; value-added at least 10 points higher with and without race inputs. Tribal, specialized, online-by-design and closed or merged colleges are never compared, and single-gender colleges are never suggested. Details and reasons: `data_decisions.md` #15-19. Of the 45 nearest options for high-Pell colleges, 35 are minority-serving (36 distinct colleges, 26 of them minority-serving).

## Advanced Track checklist

| Requirement | Where |
| --- | --- |
| Train a predictive model | `src/03_college_model.py` (gradient-boosted trees, tuned) |
| Compare with a simple baseline | National median, Pell share + sector, and ridge on all inputs (`results/college_metrics.json`) |
| Metrics on a held-out test set | MAE, median % error, R², bootstrap intervals; test set never used for tuning |
| Error and fairness analysis, 3+ subgroups, failure examples | `results/errors_by_group.csv`, `bottom20_by_measure.csv`, `race_sensitivity.csv`, `worst_test_predictions.csv` |
| Data decisions explained and hand-checked | `data_decisions.md`; the team hand-checked the samples in `handcheck/` against the raw files. A second, independent re-check of every row is recorded in the sheets (`checked_by` column): 20/20 college rows, 27/30 postings (3 degree flags missed "BS"), 10/10 tracts, 12/15 suggested pairs (3 have a narrower program mix) |
| Code from raw data to results | `python run_all.py` |

## Reproduce it

1. Python 3.13: `pip install -r requirements.txt`
2. Put the raw files in `data/raw/` with the names listed in `data/raw/README.md` (datathon shared drive, Opportunity Insights and the Census Gazetteer).
3. `python run_all.py` (about 30 minutes on 2 cores; step 3's grid search takes most of it). Single steps: `python run_all.py 4 5 5b 6 8 9`.

## Ethics: who could be hurt

- **Colleges.** Being compared could cost a college students and funding, and hurt the students who stay. There is no ranked "worst" list: the tool sorts lists by name and shows a college's numbers when someone looks it up, and the policy ask is investment where students have no alternative.
- **Minority-serving colleges.** Raw-earnings yardsticks push HBCUs, PBIs and high-Pell colleges toward the bottom; value-added mostly doesn't (not for every group: AANAPISIs fare worse by value-added). Most suggested options are themselves minority-serving, and the tool shows minority-serving status and says what earnings miss (belonging, support). 4 of the 8 HBCU/PBI starting colleges with an option are pointed to a college that is neither.
- **Tribal colleges.** The model is biased for them, so they are never rated or compared.
- **Students who can't move or get in.** Options are within 25 miles, same level, and no harder to get into; online-only, single-gender and closed colleges are excluded.
- **Value judgments.** Including race as an input is a choice; suggestions must hold with and without race inputs, and both versions are reported.
- **Coverage.** Colleges that share an earnings figure or have under 500 students aren't rated; ranked colleges enroll 73% of first-time Pell students.

## Limitations

- **Value-added is not proof of cause.** It credits a college for anything its inputs miss: unmeasured student traits, program mix (e.g. a business-heavy college), local labor markets, noise. The independent-student test shows the gaps aren't just noise; it doesn't show the college caused them.
- **Old outcomes, newer prices.** Earnings describe students who entered college about 15 years ago (measured 10 years after entry); prices are from AY2022-23. Colleges change, close and merge.
- **Former students, not graduates.** Earnings cover federally aided dependent students, including those who didn't finish.
- **Act 1** compares today's postings with outcomes of kids born 1978-83, treats postings as openings, not hires, and places each posting at its city's point. It is a description, not a causal test.
- **Privacy.** Results are at college and census-tract level. The hand-check sheet lists 30 public job postings (employer, job ID, location) so they can be checked; no person is identified.

## Prior work

Value-added college rankings (Rothwell & Kulkarni, "Beyond College Rankings", Brookings, 2015), Mobility Report Cards (Chetty, Friedman, Saez, Turner & Yagan, 2017) and the federal College Scorecard. New here: nearby, same-price comparisons a student can act on, a test on students the model never saw, safeguards for minority-serving and tribal colleges, and a free tool.

## Sources

- Datathon datasets: College Affordability (College Results View 2021; Affordability Gap AY2022-23) and Indeed Jobs (Bright Data, The Bright Initiative). Zillow and Google Maps files were ingested during screening but are not used in any result.
- [Opportunity Insights: The Opportunity Atlas data](https://opportunityinsights.org/paper/the-opportunity-atlas/) (tract outcomes and neighborhood characteristics)
- [Census Bureau: 2010 Gazetteer files](https://www.census.gov/geographies/reference-files/2010/geo/gazetter-file.html)
- [Hillman, Geography of Opportunity Brief #2 (TICAS, October 2023)](https://ticas.org/wp-content/uploads/2023/11/HIllman-Geography-of-Opportunity-Brief-2_2023.pdf)
- [NASFAA brief on the One Big Beautiful Bill Act's accountability changes (updated July 2026)](https://www.nasfaa.org/uploads/documents/OB3_Accountability_Changes_Brief_for_Campus_Leadership.pdf)
- [Chetty & Hendren, The Impacts of Neighborhoods on Intergenerational Mobility I (QJE 2018)](https://economics.mit.edu/sites/default/files/2023-07/pub%20vers.pdf)
