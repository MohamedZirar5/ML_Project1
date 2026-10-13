**How we would like to juge those features ?**
- Include or remove in case of side info not that important from prediction: some admin info...
- Infor about cardiac rehab - its kind of giving info that there was already a cardiac probem -- data leakage?
- Infor about astma, prostate cancer etc ?
- HIV shots HPVADSHT is it imprtant?

## How we organize the drop list

We don't want to drop a feature just because it has a lot of missing values. Some missing values can also give information. First we look at what the column actually means, and then we put it in one of these groups:

- administrative information: IDs, interview dates, phone screening, survey weights, and other information about how the interview was done;
- leakage risk: questions that can be asked after a heart problem, such as cardiac rehabilitation, aspirin, or advice from a doctor;
- optional modules: questions asked only in some states or only to a small subgroup of people;
- follow-up questions: questions asked only after a previous answer, for example diabetes, smoking, arthritis, or cancer-screening follow-ups;
- data-quality flags: columns saying that another answer is missing or implausible;
- redundant features: several columns measuring the same thing when the BRFSS already provides a cleaner summary variable.

This way the decision is easier to understand and to change. The list is still written explicitly because we want to be able to check every group. The missing-value ratio is used as a diagnostic, not as the only reason for dropping a feature. All decisions are made from the training data and the test data is only transformed with the selected columns.

Drop all of these too. They're almost entirely optional state modules, follow-ups that only certain people were asked, or data about someone other than the respondent. Grouped so you can see why:
Aspirin module (CVDASPRN, RDUCHART, RDUCSTRK, RLIVPAIN, ASPUNSAF): these are about taking aspirin to prevent heart attack or stroke. Like the rehab questions before, they're a leakage risk, because people with heart disease are often told to take aspirin. Drop them even though they sound relevant.
Sodium module (WTCHSALT, DRADVISE, LONGWTCH): "Did a doctor advise you to reduce salt?" is also leaky, since that advice often follows a heart or blood pressure diagnosis.
Diabetes follow-ups (DOCTDIAB, CHKHEMO3, FEETCHK, FEETCHK2, BLDSUGAR, EYEEXAM, DIABEYE, DIABEDU, INSULIN, DIABAGE2, PDIABTST): only asked of diabetics (or only of non-diabetics, for the prediabetes test). Diabetes is an important risk factor, but use the core diabetes question (DIABETE3 or whatever version your year has) instead of these.
Smoking follow-ups (LASTSMK2, STOPSMK2): skip-pattern questions for smokers and former smokers. Use _SMOKER3 instead, which already summarizes smoking status for everyone.
Asthma follow-ups (ASTHNOW, CASTHDX2, CASTHNO2): use the core ASTHMA3 if you want asthma at all. The CASTH ones are about a child in the household.
Random child selection (RCSGENDR, RCSRLTN2, _CPRACE, _CRACE1, _CHISPNC, _CLLCPWT): these describe a child in the household, not the respondent, so they have nothing to do with your outcome.
Cancer screening (mammogram, Pap, HPV test, hysterectomy, breast exam, colonoscopy/sigmoidoscopy, blood stool test, PSA): HADMAM, HOWLONG, HADPAP2, LASTPAP2, HPVTEST, HPLSTTST, HADHYST2, PROFEXAM, LENGEXAM, BLDSTOOL, LSTBLDS3, HADSIGM3, HADSGCO1, LASTSIG3, PSATEST1, PCPSA*. These are sex- or age-restricted and off-topic.
Optional modules on depression screening (AD*), mental health and life satisfaction (MISTMNT, EMTSUPRT, LSATISFY), arthritis management (ARTH*, ARTTODAY), cognitive decline (CD*), caregiving (CRGV*), social context (SCNT*), and vaccines (HPVADVC2, SHINGLE2, TETANUS): too sparse to be useful.
Other (NUMPHON2, PREGNANT): phone line count is administrative, and pregnancy is only asked of women of childbearing age.




Features to drop:
COLGHOUS, PCDMDECN, ASERVIST, ASDRVIST, ASNOSLEP, (everything AS... drop), STREHAB1, VINOCRE2, PCPSADE1, HAREHAB1, ASTHMAGE, CCLGHOUS,HPVADSHT, VIINSUR2, VIEYEXM2, VIMACDG2,VIPRFVS2, VIGLUMA2, (all VI.. drop), PSATIME, PCPSARS1, CDHELP, COLGHOUS, LADULT, PCDMDECN, ASERVIST, ASDRVIST, ASNOSLEP, ASACTLIM, ASYMPTOM, ASTHMED3, ASINHALR, ASATTACK, ASRCHKUP, STREHAB1, VINOCRE2, PCPSADE1, HAREHAB1, ASTHMAGE, CCLGHOUS, HPVADSHT, VIEYEXM2, CDHELP, LENGEXAM, RDUCSTRK, RDUCHART, RLIVPAIN, PROFEXAM, PCPSADI1, PSATEST1, PCPSAAD2, PCPSARE1, CASTHNO2, HPLSTTST, HPVADVC2, CDHOUSE, CDSOCIAL, CDDISCUS, CDASSIST, NUMPHON2, ASPUNSAF, HPVTEST, HOWLONG, CVDASPRN, ADPLEASR, ADENERGY, LSTBLDS3, ARTHEXER, ADSLEEP, ARTHWGT, ADEAT1, ARTTODAY, ARTHEDU, ADMOVE, ADDOWN, ADTHINK, ADFAIL, EMTSUPRT, LSATISFY, ADANXEV, MISTMNT, LASTPAP2, CRGVREL1, CRGVPRB1, HADHYST2, CRGVHRS1, HADPAP2, HADMAM, CRGVMST2, CRGVLNG1, CRGVHOUS, CRGVPERS, LONGWTCH, SCNTLPAD, CHKHEMO3, SHINGLE2, FEETCHK, DOCTDIAB, FEETCHK2, BLDSUGAR, EYEEXAM, DIABEYE, DIABEDU, INSULIN, SCNTPAID, SCNTWRK1, TETANUS, SCNTLWK1, HADSGCO1, LASTSIG3, LASTSMK2, DRADVISE, WTCHSALT, CASTHDX2, BLDSTOOL, HADSIGM3, DIABAGE2, RCSGENDR, ASTHNOW, RCSRLTN2, _CPRACE, _CRACE1, _CLLCPWT, STOPSMK2, _CHISPNC, PREGNANT, SCNTMNY1, SCNTMEL1, CRGVEXPT, PDIABTST,PREDIAB1, WHRTST10, CAREGIV1, HIVTSTD3, JOINPAIN, CIMEMLOS, ARTHDIS2, _PNEUMO2, LMTJOIN3, ARTHSOCL, _FLSHOT6, SXORIENT, TRNSGNDR, PAFREQ2_, BPMEDS, HHADULT, SMOKDAY2, IMFVPLAC, LANDLINE, CADULT, CTELNUM1, CELLFON2, PVTRESD2, CSTATE, FLSHTMY2, PADUR2_, EXERHMM2, EXEROFT2, MAXDRNKS, AVEDRNK2, DRNK3GE5, POORHLTH, PAFREQ1_, MSCODE, CPDEMO1, NUMHHOL2, NUMMEN, NUMWOMEN, NUMADULT, STATERES, CTELENUM, CELLFON3, PVTRESD1, _DUALUSE, _DUALCOR, PAMIN21_, EXRACT21, PAMIN11_, _MINAC21, METVL21_, _HCVU651, PAVIG21_, _MINAC11, ACTIN21_, EXRACT11, PAVIG11_, PADUR1_, EXERHMM1, PA1MIN_, METVL11_, PA1VIGM_, EXEROFT1, ACTIN11_, INCOME2, _AGEG5YR, PNEUVAC3, CHOLCHK, TOLDHI2, STRFREQ_, HIVTST6, _AIDTST3, _PAREC1, _PASTAE1, _PA30021, _PA300R2, _PACAT1, _PA150R2, _VEGLT1, FLUSHOT6, STRENGTH, _PASTRNG, _FRTLT1, _RFSEAT3, _RFSEAT2, SEATBELT, VEGEDA1_, VEGETAB1, BEANDAY_, FVBEANS, ORNGDAY_, FVORANG, FTJUDA1_, FRUITJU1, PAMISS1_, EXERANY2, GRENDAY_, FVGREEN, FRUTDA1_, _BMI5CAT, _RFBMI5, FRUIT1, WEIGHT2, WTKG3, DROCDY3_, DRNKANY5, ALCDAY5, HTIN4, _LMTWRK1, _RFSMOK3, _MRACE1, SMOKE100, _LMTACT1, _LMTSCL1, HTM4, HEIGHT3, _PRACE1, BLOODCHO, _RACEG21, _RACE_G1, _RACEGR3, INTERNET, MAXVO2_, FC60_, _HISPANC, CHILDREN, _CASTHM1, _ASTHMS1, _DRDXAR1, EDUCA, _LTASTH1, _RFHYPE5, _RFHLTH, _DRNKWEK, _AGE_G, _AGE65YR, QSTLANG, Id, SEQNO, IDAY, IMONTH, IDATE, FMONTH, IYEAR, DISPCODE, QSTVER, _VEGETEX, _FRUITEX, _VEG23, _FRT16, _VEGRESP, _FRTRESP, _MISVEGN, _MISFRTN, _STRWT, _PSU, _STSTR, _LLCPWT, _WT2RAKE, _RAWRAKE 




Here's why each group in that list is dropped. There are really only four reasons: the column is about survey logistics, it's a narrow optional module or follow-up, it's a duplicate of a cleaner version you're keeping, or it's off-topic for heart disease.
1. Survey administration and IDs
Id, SEQNO, IDAY, IMONTH, IDATE, FMONTH, IYEAR, DISPCODE, QSTVER, QSTLANG
These are record numbers, interview dates, the questionnaire version, the interview language, and the interview completion code. They describe how the interview was done, not the person's health. Keeping an ID in a model is also a known trap: the model can memorize it without learning anything real.
2. Phone screening and household roster
LANDLINE, CADULT, CTELNUM1, CELLFON2, PVTRESD2, CSTATE, STATERES, CTELENUM, CELLFON3, PVTRESD1, HHADULT, NUMHHOL2, NUMMEN, NUMWOMEN, NUMADULT, CPDEMO1, MSCODE, _DUALUSE, _DUALCOR
These are the same kind of questions as COLGHOUS from the start: "Is this a private residence?", "Is this a cell phone?", "How many adults live here?" They were used to check eligibility and pick a respondent. Most are only asked in one survey mode (landline or cell), which is why they're about 42% or 57% missing. MSCODE (metropolitan status) is also landline-only.
3. Survey weights and design variables
_STRWT, _PSU, _STSTR, _LLCPWT, _WT2RAKE, _RAWRAKE
These are numbers the CDC uses so that results represent the US population. They aren't characteristics of the person, so they don't belong in the features. As mentioned, save _LLCPWT, _STSTR, and _PSU separately if you ever want population-level estimates.
4. Data-quality flags for fruit and vegetable intake
_VEGETEX, _FRUITEX, _VEG23, _FRT16, _VEGRESP, _FRTRESP, _MISVEGN, _MISFRTN
These only mark whether the fruit and vegetable answers were missing or implausible. The actual intake is already in _FRUTSUM and _VEGESUM, which you keep.
5. Optional modules asked only in some states
WHRTST10, HIVTSTD3, CAREGIV1, CIMEMLOS, SXORIENT, TRNSGNDR, IMFVPLAC, FLSHTMY2
These cover HIV testing location and date, caregiving, memory loss, sexual orientation, gender identity, and flu shot place and date. They're about 55–75% missing because most states didn't ask them, and they're not related to heart disease anyway.
6. Follow-up questions (skip patterns)
JOINPAIN, ARTHDIS2, LMTJOIN3, ARTHSOCL, _LMTWRK1, _LMTACT1, _LMTSCL1: only asked of people with arthritis. You keep HAVARTH3 instead.
SMOKDAY2: only asked of people who have smoked. Covered by _SMOKER3.
AVEDRNK2, DRNK3GE5, MAXDRNKS: only asked of people who drink. Covered by _RFDRHV5 and _RFBING5.
POORHLTH: only asked if someone reported bad physical or mental health days. Covered by PHYSHLTH and MENTHLTH.
BPMEDS: only asked of people with high blood pressure, and blood pressure medication is closely tied to heart care, so it's also a mild leakage risk. You keep BPHIGH4.
TOLDHI2: only asked of people whose cholesterol was checked. Covered by _RFCHOL.
_HCVU651: health coverage for ages 18–64 only. Covered by HLTHPLN1.
7. Duplicates: physical activity
PAFREQ1_, PAFREQ2_, PADUR1_, PADUR2_, EXERHMM1, EXERHMM2, EXEROFT1, EXEROFT2, EXRACT11, EXRACT21, PAMIN11_, PAMIN21_, _MINAC11, _MINAC21, METVL11_, METVL21_, PAVIG11_, PAVIG21_, ACTIN11_, ACTIN21_, PA1MIN_, PA1VIGM_, PAMISS1_, STRFREQ_, STRENGTH, _PASTRNG, _PAREC1, _PASTAE1, _PA30021, _PA300R2, _PACAT1, _PA150R2, EXERANY2
These are the raw exercise answers (which activity, how often, how long) and the many summaries the CDC calculates from them. They all measure the same thing. You keep _TOTINDA (any exercise at all) and _PAINDX1 (meets the activity guideline), which summarize it cleanly.
8. Duplicates: diet
FRUIT1, FRUITJU1, FVGREEN, FVORANG, FVBEANS, VEGETAB1, FRUTDA1_, FTJUDA1_, GRENDAY_, ORNGDAY_, BEANDAY_, VEGEDA1_, _FRTLT1, _VEGLT1
These are the individual food questions and their per-day conversions. They're already combined into _FRUTSUM and _VEGESUM, which you keep.
9. Duplicates: body size
WEIGHT2, WTKG3, HEIGHT3, HTIN4, HTM4, _BMI5CAT, _RFBMI5
BMI is calculated from height and weight, so _BMI5 already contains this information. _BMI5CAT and _RFBMI5 are just grouped versions of it. Raw WEIGHT2 and HEIGHT3 also use awkward mixed-unit codes that are easy to get wrong.
10. Duplicates: smoking and alcohol
SMOKE100, _RFSMOK3: covered by _SMOKER3, which combines them into one variable (current daily, current some days, former, never).
ALCDAY5, DRNKANY5, DROCDY3_, _DRNKWEK: covered by _RFDRHV5 (heavy drinking) and _RFBING5 (binge drinking). _DRNKWEK also uses 99900 as a code for "unknown," which would badly distort a model if missed.
11. Duplicates: demographics
_AGEG5YR, _AGE_G, _AGE65YR: covered by _AGE80 (actual age, topped at 80), which is also the most complete.
_MRACE1, _PRACE1, _RACEG21, _RACE_G1, _RACEGR3, _HISPANC: covered by _RACE, which already includes Hispanic origin.
EDUCA: covered by _EDUCAG.
INCOME2: covered by _INCOMG, which has fewer missing values.
CHILDREN: covered by _CHLDCNT.
12. Duplicates: health conditions
_RFHYPE5 duplicates BPHIGH4 (high blood pressure).
_RFHLTH is a yes/no version of GENHLTH (general health), which has more detail.
_CASTHM1, _ASTHMS1, _LTASTH1 duplicate ASTHMA3.
_DRDXAR1 duplicates HAVARTH3.
CHOLCHK and BLOODCHO duplicate _CHOLCHK (whether and when cholesterol was checked).
13. Calculated from other columns
MAXVO2_, FC60_: these estimated fitness scores are computed by the CDC from age and sex only, so they add nothing beyond _AGE80 and SEX.
14. Off-topic health behaviors
SEATBELT, _RFSEAT2, _RFSEAT3, HIVTST6, _AIDTST3, FLUSHOT6, _FLSHOT6, PNEUVAC3, _PNEUMO2, INTERNET
Seatbelt use, HIV testing, vaccinations, and internet use have no direct link to heart disease. Vaccinations mostly track age and doctor visits, which you already capture with _AGE80 and CHECKUP1.
