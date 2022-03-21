



import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, chi2
from itertools import combinations

from sklearn import tree


def discretize(data, target, modality):
    """ Descretize a numeric feature values into categorical feature of *n* modality
        using Decsision tree
    parameters:
    ------------
            data: [pandas.Series, shape(n_individuals)] Input data vector
            target: [pandas.Series of shape (n_samples)] Target data vector
            modality: [int]  Number of modality of the resulted data vector
    return:
    -------
            data_disc: [pandas.Series, shape(n_individuals] discrete categorical data where the value is serie's name + node_id
    """

    name = data.name

    data = data.to_numpy().reshape(-1,1)
    target = target.to_numpy().reshape(-1,1)

    clf = tree.DecisionTreeClassifier(max_leaf_nodes=modality)
    clf.fit(data, target)
    leaves_id = clf.apply(data)

    data_disc = [name+"_"+str(leaf_id) for leaf_id in leaves_id]
    data_disc = pd.Series(data_disc, name=name)
    # data_disc = data_disc.astype(CategoricalDtype(categories=list(data_disc.unique())))
    return data_disc

def statitical_tests(data, alpha=0.05):
    """ compute chi2 test, o values and Cramer's V between each features of dataframe two by two
    parameters:
    -----------
        data: [pandas.DataFrame, shape(n_individuals)] a dataframe that consists of only categorical features
        alpha: [float] alpha of phi2 test
    return:
    -------
        df_test [pandas.DataFrame  shape(comb_features, ["variable 1","variable 2", "critical values", "chi2",
        "p values", "Cramer's V "]
    """
    list_cat = list(data.columns)
    combs_cat = list(combinations(list_cat, 2))

    p_values_list = []
    chi2_list = []
    critical_list = []
    var_1 = []
    var_2 = []
    cramer_v_list = []

    for comb in combs_cat:

        contingency_tab = pd.crosstab(data[comb[0]], data[comb[1]])  # contingency matrix

        chi2_coeff, critical, p = chi2_test(contingency_tab, alpha=alpha)
        cramer_coeff = cramers_V(contingency_tab)

        p_values_list.append(p)
        chi2_list.append(chi2_coeff)
        critical_list.append(critical)
        cramer_v_list.append(cramer_coeff)
        var_1.append(comb[0])
        var_2.append(comb[1])


    # create a dataframe
    df = pd.DataFrame({'variable 1': var_1,
                            'variable 2': var_2,
                            'critical values': critical_list,
                            'chi2 ': chi2_list,
                            'p values': p_values_list,
                            "Cramer's V": cramer_v_list})
    return df

def statitical_tests_target(data, target, alpha=0.05):
    """ compute chi2 test, o values and Cramer's V between features and target
    parameters:
    -----------
        data: [pandas.DataFrame, shape(n_individuals, n_feautures)] a dataframe that consists of only categorical features
        alpha: [float] alpha of phi2 test
        target: [pd.Series, shape(n_individuals)] target values
    return:
    -------
        df_test [pandas.DataFrame  shape(comb_features, ["variable 1", "critical values", "chi2",
        "p values", "Cramer's V "]
    """
    list_cat = list(data.columns)

    p_values_list = []
    chi2_list = []
    critical_list = []
    var_list = []
    cramer_v_list = []

    for col in list_cat:

        contingency_tab = pd.crosstab(data[col], target)  # contingency matrix

        chi2_coeff, critical, p = chi2_test(contingency_tab, alpha)
        cramer_coeff = cramers_V(contingency_tab)

        p_values_list.append(p)
        chi2_list.append(chi2_coeff)
        critical_list.append(critical)
        cramer_v_list.append(cramer_coeff)
        var_list.append(col)


    # create a dataframe
    df = pd.DataFrame({'variables': var_list,
                            'critical values': critical_list,
                            'chi2 ': chi2_list,
                            'p values': p_values_list,
                            "Cramer's V": cramer_v_list})
    return df

def cramers_V(contingency_tab):
    chi2 = chi2_contingency(contingency_tab)[0]  # chi2 coefficient
    N    = contingency_tab.sum().sum()   # number of samples
    return np.sqrt(chi2 / (N*(min(contingency_tab.shape)-1)))

def chi2_test(contingency_tab, alpha=0.05):
    '''
    ## Chi 2 test
    # if p <= aplpha --> reject the H0 "indipendece hypothesis"
    #    p > alpha  --> fail to reject H0
    # if abs(stat) >= critical --> eject the H0
    #    abs(stat) < critical -->  fail to reject H0
    '''
    chi2_coeff, p, dof, expected = chi2_contingency(contingency_tab)

    prob = 1 - alpha

    critical = chi2.ppf(prob, dof)

    return chi2_coeff, critical, p

def chi2_data(data, alpha=0.05):

    list_cat = list(data.columns)
    combs_cat = list(combinations(list_cat, 2))

    p_values_list = []
    chi2_list = []
    critical_list = []
    var_1 = []
    var_2 = []

    for comb in combs_cat:
        print(comb)
        table = pd.crosstab(data[comb[0]], data[comb[1]])  # contingency matrix
        chi2_coeff, p, dof, expected = chi2_contingency(table)

        prob = 1 - alpha

        critical = chi2.ppf(prob, dof)

        p_values_list.append(p)
        chi2_list.append(chi2_coeff)
        critical_list.append(critical)
        var_1.append(comb[0])
        var_2.append(comb[1])

    # create a dataframe
    df_chi2 = pd.DataFrame({'variable 1': var_1,
                            'variable 2': var_2,
                            'critical values': critical_list,
                            'chi2 ': chi2_list,
                            'p values': p_values_list})
    return df_chi2

