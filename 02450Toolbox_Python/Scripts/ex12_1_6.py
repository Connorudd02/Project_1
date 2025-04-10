# ex12_1_6
# Load resources from previous exercise
from apyori import apriori
from ex12_1_4 import mat2transactions, print_apriori_rules
from ex12_1_5 import Xbin, attributeNamesBin

# Given the processed data in the previous exercise this becomes easy:
T = mat2transactions(Xbin, labels=attributeNamesBin)
rules = apriori(T, min_support=0.3, min_confidence=0.6)
print_apriori_rules(rules)
