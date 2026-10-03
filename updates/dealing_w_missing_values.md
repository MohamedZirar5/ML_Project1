**CATHEGORICAL DATA:**
When to use which:
- Mode when there are few missing values (say under 5–10%). Guessing the most common answer barely changes anything.
- Constant when many values are missing, or when the missingness itself might mean something. Filling 30% of a column with the mode would create a big artificial spike at one value.




**Numerical data:**
1. Clean special codes. Convert BRFSS codes into real values first: "don't know" and "refused" (77, 99, 777, 999, etc.) become np.nan, and in PHYSHLTH and MENTHLTH, 88 ("none") becomes 0. Check the codebook for each numeric variable, since the codes differ.
2. Fill missing values. Use the median for skewed columns (most health data, like BMI or fruit intake), and the mean only for roughly symmetric ones. The median is the safer default. Optionally, add a 0/1 "was missing" column for features with a lot of missingness, so the model knows which values were filled in.
3. Handle outliers. Extreme values (like a BMI of 90 or 20 fruit servings a day) can distort the model. A common fix is clipping: capping values at, say, the 1st and 99th percentiles of the training data.
4. Standardize. Transform each column to mean 0 and standard deviation 1. This matters a lot for models trained with gradient descent, like logistic regression, because otherwise large-scale features (BMI around 2,800 in BRFSS units) dominate small ones (a yes/no of 0 or 1).
All the statistics (medians, percentiles, means, standard deviations) must come from the training set only and then be applied unchanged to the test set.
