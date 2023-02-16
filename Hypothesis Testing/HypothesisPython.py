# 1 Sample Sign Test
# 1 Sample Z-Test
# Mann-Whitney test
# Paired T-Test
# Moods-Median Test
# 2 sample T-Test
# One - Way Anova
# 2-Proportion
# Chi-Square Test
# Tukey's Test


import pandas as pd
import numpy as np
import scipy
from scipy import stats
import statsmodels.stats.descriptivestats as sd
from statsmodels.stats import weightstats as stests


############ 1 Sample Sign Test ################
# Student Scores Data
marks = pd.read_csv("C:/Users/vaibh/Desktop/360 Digitmg/Hypothesis Testing/Signtest.csv")
marks

# Normal Q-Q plot
import pylab

# Checking Whether data is normally distributed
stats.probplot(marks.Scores, dist="norm", plot=pylab)

# Normality Test
stats.shapiro(marks.Scores) # Shapiro Test

# Hypothsis
# H0 = Data are Normal
# Ha = Data are not Normal
# p-value = 0.0243 < 0.05 so p low null go => Data are not Normal

# Distribution of the Data
marks.Scores.describe()

import matplotlib.pyplot as plt
plt.boxplot(marks.Scores)

# Non-Parameteric Test --> 1 Sample Sign Test
sd.sign_test(marks.Scores, mu0=80)
sd.sign_test(marks.Scores, mu0=marks.Scores.median())

# p-value = 0.647 > 0.05 so p high null fly
# Ho: Scores are either equal to or less than 80%


############# 1-Sample Z-Test #############
#  importing the data
fabric = pd.read_csv("C:/Users/vaibh/Desktop/360 Digitmg/Hypothesis Testing\Fabric_data.csv")

# calculating the normality test
print(stats.shapiro(fabric))

#calculating the mean
np.mean(fabric)

# ztest
# parameters in ztest, value is mean of data
ztest, pval = stests.ztest(fabric, x2 = None, value = 150)

print(float(pval))

# p-value = 7.156e-06 < 0.05 so p low null go


######### Mann-Whitney Test ############
# Vehicles with and without addictive

fuel = pd.read_csv("C:/Users/vaibh/Desktop/360 Digitmg/Hypothesis Testing\mann_whitney_additive.csv")
fuel

fuel.columns = "Without_additive", "With_additive"

# Normality test 
print(stats.shapiro(fuel.Without_additive))  # p high null fly
print(stats.shapiro(fuel.With_additive))     # p low null go

# Non-Parameteric Test case
# Mann-Whitney test
scipy.stats.mannwhitneyu(fuel.Without_additive, fuel.With_additive)

# p-value = 0.222 > 0.05 so p high null fly
# Ho: fuel additive does not impact the performance


 
############### Paired T-Test ##############
# A univariate test that tests for a significant difference between 2 related variables.

sup = pd.read_csv("C:/Users/vaibh/Desktop/360 Digitmg/Hypothesis Testing/paired2.csv")
sup.describe()

# Normality Test - # Shapiro Test
stats.shapiro(sup.SupplierA) 
stats.shapiro(sup.SupplierB)
# Data are Normal

import seaborn as sns
sns.boxplot(data=sup)

# Assuming the external Conditions are same for both the samples
# Paired T-Test
ttest, pval = stats.ttest_rel(sup['SupplierA'], sup['SupplierB'] )
print(pval)
# Ho: There is no significant difference between means of suppliers of A and B
# Ha: There is significant difference between means of suppliers of A and Bx
# p-value = 0 < 0.05 so p low null go



###### Moods-Median Test ######

# Import libraries
import pandas as pd

#Import dataset
animals = pd.read_csv("C:/Data/hypothesis/animals.csv")

# moods median test
from scipy.stats import median_test
stat, p, med, tbl = median_test(animals.Pooh, animals.Piglet, animals.Tigger)

# The median
med

# p-value is too large to conclude that the medians are not the same:
p # 0.18637397603941

# ties: str, optional
# Determines how values equal to the grand median are classified in the contingency table. 
# The string must be one of:
# "below":Values equal to the grand median are counted as "below".
# "above":Values equal to the grand median are counted as "above".
# "ignore":Values equal to the grand median are not counted.

stat, p, med, tbl = median_test(animals.Pooh, animals.Piglet, animals.Tigger, ties="above")
p #0.00036461568873027327



############ 2 sample T-Test #############

# Load the data
prom = pd.read_excel("C:/Data/hypothesis/Promotion.xlsx")
prom

prom.columns = "InterestRateWaiver", "StandardPromotion"

# Normality Test
stats.shapiro(prom.InterestRateWaiver) # Shapiro Test

print(stats.shapiro(prom.StandardPromotion))

# Data are Normal

# Variance test
help(scipy.stats.levene)

scipy.stats.levene(prom.InterestRateWaiver, prom.StandardPromotion)
# p-value = 0.287 > 0.05 so p high null fly => Equal variances

# 2 Sample T test
scipy.stats.ttest_ind(prom.InterestRateWaiver, prom.StandardPromotion)
help(scipy.stats.ttest_ind)
# Ho: equal means
# Ha: unequal means
# p-value = 0.024 < 0.05 so p low null go

scipy.stats.ttest_ind(prom.InterestRateWaiver, prom.StandardPromotion, alternative='greater')
# Ho: InterestRateWaiver < StandardPromotion
# Ha: InterestRateWaiver > StandardPromotion
# p-value = 0.012 < 0.05 so p low null go



############# One-Way ANOVA #############

con_renewal = pd.read_excel("C:/Data/ContractRenewal_Data(unstacked).xlsx")
con_renewal
con_renewal.columns = "SupplierA", "SupplierB", "SupplierC"

# Normality Test
stats.shapiro(con_renewal.SupplierA) # Shapiro Test
stats.shapiro(con_renewal.SupplierB) # Shapiro Test
stats.shapiro(con_renewal.SupplierC) # Shapiro Test

# Variance test
help(scipy.stats.levene)
# All 3 suppliers are being checked for variances
scipy.stats.levene(con_renewal.SupplierA, con_renewal.SupplierB, con_renewal.SupplierC)

# One - Way Anova
F, p = stats.f_oneway(con_renewal.SupplierA, con_renewal.SupplierB, con_renewal.SupplierC)

# p value
p  # P High Null Fly
# All the 3 suppliers have equal mean transaction time



######### 2-Proportion Test #########
import numpy as np

two_prop_test = pd.read_excel("C:/Data/JohnyTalkers.xlsx")

from statsmodels.stats.proportion import proportions_ztest

tab1 = two_prop_test.Person.value_counts()
tab1
tab2 = two_prop_test.Drinks.value_counts()
tab2

# crosstable table
pd.crosstab(two_prop_test.Person, two_prop_test.Drinks)

count = np.array([58, 152])
nobs = np.array([480, 740])

stats, pval = proportions_ztest(count, nobs, alternative = 'two-sided') 
print(pval) # Pvalue 0.000

stats, pval = proportions_ztest(count, nobs, alternative = 'larger')
print(pval)  # Pvalue 0.999  



############### Chi-Square Test ################

Bahaman = pd.read_excel("C:/Users/vaibh/Desktop/360 Digitmg/Hypothesis Testing/Bahaman.xlsx")
Bahaman

count = pd.crosstab(Bahaman["Defective"], Bahaman["Country"])
count
Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square



##### Tukey's Test #####
import numpy as np
import scipy.stats as stats

# Create four random groups of data with a mean difference of 1

mu1, sigma1 = 10, 3 # mean and standard deviation
group1 = np.random.normal(mu1, sigma1, 50)

mu2, sigma2 = 11, 3 # mean and standard deviation
group2 = np.random.normal(mu2, sigma2, 50)

mu3, sigma3 = 12, 3 # mean and standard deviation
group3 = np.random.normal(mu3, sigma3, 50)

mu4, sigma4 = 13, 3 # mean and standard deviation
group4 = np.random.normal(mu4, sigma4, 50)

# Show the results for Anova

F_statistic, pVal = stats.f_oneway(group1, group2, group3, group4)

print (pVal)

# Put into dataframe
import pandas as pd
df = pd.DataFrame()
df['treatment1'] = group1
df['treatment2'] = group2
df['treatment3'] = group3
df['treatment4'] = group4

# Stack the data (and rename columns):

stacked_data = df.stack().reset_index()
stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                            'level_1': 'treatment',
                                            0:'result'})
# Show the first 8 rows:

print (stacked_data.head())

# Tukey’s multi-comparison method
# Tukey's Honest Significant Difference
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

# Set up the data for comparison (creates a specialised object)
MultiComp = MultiComparison(stacked_data['result'],
                            stacked_data['treatment'])

# Show all pair-wise comparisons:
# Print the comparisons
print(MultiComp.tukeyhsd().summary())


### End ###