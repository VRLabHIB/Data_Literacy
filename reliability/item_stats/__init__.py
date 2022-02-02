import pandas as pd
import pingouin as pg
from scipy.stats import pearsonr


def item_stats(df=None,level=None,drop=None):
    """
    Calculate and display different statistical values to state item and scale reliability for questionnaire items.

    Parameters
    ------------------
    df::py:class:'pandas.DataFrame'
        Wide format i.e. df.loc[case, variable]
    level: {"nominal", "ordinal", "scale", "scales"}
        Specifies the scale level of the item values.

        nominal:
            States frequency of item values.
        scale:
            Questions with only one item. 
        scales: 
            More than one item. Computes item-total correlation and cronbachs alpha
        drop: array (1 dim)
            Name variables that need to be removed from the dataframe before the analysis 

    Returns
    ------------------
    If level=='scale':
        df_alpha::py:class:'pandas.DataFrame'
            Contains Number of cases, total mean, total standard deviation, alpha and 95% CFI
            total mean and std is the mean of the sum-scores of all items
        stats::py:class:'pandas.DataFrame'
            Contains mean, standard deviation and item-total correlation for each item
    """
    if drop!=None:
        df = df.drop(drop, axis=1)

    if level == 'scale':
        #scale with one item
        n = len(df)
        mean = df.sum(axis=1).mean()
        std = df.sum(axis=1).std()
        deskriptives = pd.DataFrame({'N':n,'Mean':round(mean,3),'Std':round(std,3)})
        frequencies = pd.DataFrame(df.value_counts())
        return[deskriptives,frequencies]

    if level == "scales":
        #more than 1 item is necessary
        n = len(df)
        mean = df.sum(axis=1).mean()
        std = df.sum(axis=1).std()

        gca = pg.cronbach_alpha(df)
        df_alpha = pd.DataFrame({'N':n,'Mean':round(mean,3),'Std':round(std,3), 'Alpha':[round(gca[0],3)],'CFI_low':[round(gca[1][0],3)],'CFI_high':[round(gca[1][1],3)]})
        df_alpha.index = ['Total Scores']
        item_lst = list()
        mean_lst = list()
        std_lst = list()
        itc_lst = list()

        for column in df:
            mean = df[column].mean()
            std = df[column].std()
            sub_df = df.drop([column], axis=1)
            pr = pearsonr(sub_df.mean(axis=1), df[column])

            item_lst.append(column)
            mean_lst.append(mean)
            std_lst.append(std)
            itc_lst.append(pr[0])

        stats = pd.DataFrame({"Mean":mean_lst, "Std":std_lst,"Item-Total Correlation":itc_lst})
        stats.index = item_lst
        stats = stats.round(3)

    return [df_alpha, stats]
