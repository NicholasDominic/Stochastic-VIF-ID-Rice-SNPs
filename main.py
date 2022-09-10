from snp_info import load_dataset
from pandas import DataFrame
from numpy import arange, square, ndarray
from statsmodels.api import OLS
from collections import Counter as C
from time import time as t
from warnings import filterwarnings as fw; fw("ignore")
from random import sample
from numpy import array
from numpy.random import shuffle
from statsmodels.api import add_constant as const
from sys import argv

def randomize(df : DataFrame, with_replacement : bool = False, subset_feature : bool = False, *args, **kwargs) -> DataFrame:
    """
    Args:
        df : DataFrame
            original dataframe (each column represents a SNP)
        with_replacement : bool
            perform random resampling (row) with or without replacement
        subset_feature : bool
            perform feature selection on whole or subset features;
            if True, then total features per bootstrapsed dataset will be log(root(n))
    Return:
        df_[ftr] : DataFrame
            dataframe with selected features
    """
    
    df_ = df.copy()
    
    # 1. bootstraping: randomize row
    df_ = df_.sample(frac=1, replace=with_replacement)
    
    # 2. random feature selection: randomize column
    # r_seed(0); np_seed(0)
    ftr = array(df_.columns)
    shuffle(ftr)
    ftr = ftr.tolist()
    
    if subset_feature:
        num_of_subset = int(sqrt(len(df_.columns)))
        ftr = sample(ftr, num_of_subset)

    return df_[ftr]

def vif_(exog : ndarray, exog_idx : int, regularization_term : float = 0, *args, **kwargs) -> tuple:
    """
    Args:
        exog : ndarray, (nobs, k_vars)
            design matrix with all explanatory variables, as for example used in regression
        exog_idx : int
            index of the exogenous variable in the columns of exog
        regularization_term : float
            to be further used for calculating xgb_regressor_similarity
    Return:
        (vif, _) : tuple
            variance inflation factor score
    """
    
    k_vars = exog.shape[1]
    x_i = exog[:, exog_idx] # get the dependent variable from (exog_idx)-th independent variables
    mask = arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    
    ols = OLS(x_i, const(x_noti)).fit()
    r_squared_i = ols.rsquared
    vif = 1. / (1. - r_squared_i)
    
    # xgb_regressor_similarity is not implemented / deployed yet in this version
    xgb_regressor_similarity = round(sum(square(ols.resid)) / (len(ols.resid) + regularization_term), 3)
    
    return vif, xgb_regressor_similarity

def detect(data, temp : dict, thresh : float, *args, **kwargs) -> list:
    new_dict = {k:v for k,v in temp.items() if v > thresh}
    detected_var = list(new_dict.keys())
    data.drop(detected_var, axis=1, inplace=True)
    return detected_var

def calc_vif(data : DataFrame, vif_threshold : float = 2.5, *args, **kwargs) -> list:
    cols, dump = data.columns[:2], []
    idx = 1 # start column index
    last_col = data.columns[len(data.columns)-1]

    temp = {}
    while last_col not in temp:
        temp = {}
        for i, j in enumerate(data[cols]):
            # why do we need to iterate over i (i.e., index column of independent variables)?
            # e.g., you have four independent variables (exog): A, B, C, D
            # with regression terms, VIF calcuation repeats the process as:
            #    A <- B, C, D
            #    B <- A, C, D
            #    C <- A, B, D
            #    D <- A, B, C
            # learn more on https://statisticsbyjim.com/regression/variance-inflation-factors/
            
            vif_score, similarity = vif_(exog=data[cols].values, exog_idx=i, regularization_term=0)
            temp[j] = round(vif_score, 3)
        
        validate = any(v > vif_threshold for v in temp.values())
        if validate: # there is VIF value above the threshold, detect and save the correlated SNPs
            snp_corr = detect(data, temp, vif_threshold)
            dump.extend(snp_corr)
        else: # add new column
            idx += 1
            
        # e.g., in the first iteration, number of columns used: 2
        # in the second iteration, number of columns used: 3
        # in the third iteration, number of columns used: 4, etc.
        cols = data.columns[:idx+1]
    return dump


def execute(data : DataFrame, n_simulations : int = 100, vif_thresh : float = 2.5, print_per_iter : int = 1, *args, **kwargs) -> dict:
    print("Starting ...")
    list_snps = []
    simulation_start_time = t()
    
    # NOTE: randomize() second param = with_replacement=True/False
    # randomize() third param = subset_feature=True/False
    for i, dataset in enumerate([randomize(data, False, False) for n in range(n_simulations)]):
        start_time = t()
        detected_snps = calc_vif(dataset, vif_thresh)
        
        if print_per_iter == 1:
            print("Highly correlated SNPs: {} (total: {})".format(detected_snps, len(detected_snps)))
        
        print("Simulation {} completed (in {:.3f} seconds).".format(i+1, t()-start_time), end="\n\n")
        list_snps.extend(detected_snps)

    print("[SUCCESS] All simulations were completed (in {:.3f} minutes).".format((t()-simulation_start_time)/60))
    return dict(C(list_snps).most_common()) # aggreagte decision

def greetings(*args, **kwargs) -> None:
    print("")
    print("-" * 60)
    print("WELCOME! -by Nicholas Dominic, S.Kom., M.T.I.")
    print("BINUS University Bioinf. & Data Sci. Research Center, Indonesia", end="\n\n")
    print("Notes for args:")
    print("Args 1: input chromosome number, between 1 to 12")
    print("Args 2: input number of simulations, recommended between 5 to 100")
    print("Args 3: input VIF threshold, recommended between 2.5 to 10.0")
    print("Args 4: if you wish to print the highly correlated SNPs per iteration, choose 1; otherwise 0")
    print("Args 5: if you wish to save the result in DataFrame (.csv), choose 1; otherwise 0")
    print("Args 6: if you wish to also save the gene co-expression result, choose 1; otherwise 0", end="\n\n")
    print("Example command to run the script:")
    print("  >> python main.py 5 5 2.5 0 1 1", end="\n\n")
    print("-" * 60, end="\n\n")

if __name__ == "__main__":
    # default params
    SAVE_PATH = "./results"
    inp_chr_num = 1
    inp_n_simulation = 5
    inp_vif_thresh = 2.5
    inp_print_per_iter = 1 # True
    inp_save_to_df = 1 # True
    inp_generate_coexp = 1 # True
    
    greetings()
    inp_chr_num = int(argv[1]) # mandatory: between 1 to 12
    inp_n_simulation = int(argv[2]) # recommended: between 5 to 100
    inp_vif_thresh = float(argv[3]) # recommended: between 2.5 to 10.0
    inp_print_per_iter = int(argv[4]) # mandatory: either 1 (True) or 0 (False)
    inp_save_to_df = int(argv[5]) # mandatory: either 1 (True) or 0 (False)
    inp_generate_coexp = int(argv[6]) # mandatory: either 1 (True) or 0 (False)
    
    curr_chr = load_dataset(chr_num=inp_chr_num)
    result = execute(
        data=curr_chr,
        n_simulations=inp_n_simulation,
        vif_thresh=inp_vif_thresh,
        print_per_iter=inp_print_per_iter
    )
    
    result_to_df = DataFrame({
        "SNP" : list(result.keys()),
        "agg_count" : list(result.values())
    })
    
    if inp_save_to_df == 1:
        mypath = SAVE_PATH+"/svif/chr-{}_(thresh-{}).csv".format(inp_chr_num, inp_vif_thresh)
        result_to_df.to_csv(path_or_buf=mypath, index=False)
        print("[SAVED] Completed: {}".format(mypath))
    elif inp_save_to_df == 0:
        print("[WARNING] Consider saving your result.")
        print("SVIF result:\n", result_to_df, end="\n\n")
        
    if inp_generate_coexp == 1:
        mypath = SAVE_PATH+"/svif-gene-coexp/chr-{}.csv".format(inp_chr_num)
        curr_chr.corr().to_csv(path_or_buf=mypath, index=False)
        print("[SAVED] Completed: {}".format(mypath))
    elif inp_generate_coexp == 0:
        print("[WARNING] Consider saving your result.")
        print("Gene coexpression result:\n", curr_chr.corr())
    
    print("[SUCCESS] All process were completed.")