import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, kstest, anderson

# Charger le dataset
data = pd.read_csv('path_to_your_file/Heart_Disease_Prediction.csv')

# Visualisation
plt.hist(data['age'], bins=30, edgecolor='black')
plt.title('Histogramme de l\'âge')
plt.xlabel('Âge')
plt.ylabel('Fréquence')
plt.show()

stats.probplot(data['age'], dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()

# Tests statistiques
stat, p = shapiro(data['age'])
print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Les données suivent une distribution normale (ne rejette pas H0)')
else:
    print('Les données ne suivent pas une distribution normale (rejette H0)')

stat, p = kstest(data['age'], 'norm')
print('Kolmogorov-Smirnov Test: Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Les données suivent une distribution normale (ne rejette pas H0)')
else:
    print('Les données ne suivent pas une distribution normale (rejette H0)')

result = anderson(data['age'], dist='norm')
print('Anderson-Darling Test: Statistic: %.3f' % result.statistic)
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
        print('A la significativité %.3f, on ne rejette pas H0 (Les données suivent une distribution normale)' % sl)
    else:
        print('A la significativité %.3f, on rejette H0 (Les données ne suivent pas une distribution normale)' % sl)
