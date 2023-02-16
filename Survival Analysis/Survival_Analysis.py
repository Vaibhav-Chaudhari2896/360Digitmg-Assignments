# pip install lifelines
# import lifelines

import pandas as pd
# Loading the the survival un-employment data
survival_unemp = pd.read_csv("C:\\Users\\vaibh\\Desktop\\360 Digitmg\\Survival Analysis\\survival_unemployment.csv")
survival_unemp.head()
survival_unemp.describe()

survival_unemp["spell"].describe()

# Spell is referring to time 
T = survival_unemp.spell
# pip install lifelines
# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed=survival_unemp.event)

# Time-line estimations plot 
kmf.plot()

# Over Multiple groups 
# For each group, here group is ui
survival_unemp.ui.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[survival_unemp.ui==1], survival_unemp.event[survival_unemp.ui==1], label='1')
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[survival_unemp.ui==0], survival_unemp.event[survival_unemp.ui==0], label='0')
kmf.plot(ax=ax)
