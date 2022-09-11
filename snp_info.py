from pandas import read_csv, DataFrame, Series
from numpy import log2, nansum
from sklearn.metrics import mutual_info_score as mis
from collections import Counter

def load_dataset(chr_num=None, *args, **kwargs) -> DataFrame:
    df = read_csv("./data/gp_table.csv") # gp = genotype-phenotype
    df = df[df.columns[:len(df.columns)-3]] # -3 to exclude loc, var, and yield variables
    
    if chr_num == None:
        return df
    else:
        if chr_num < 1 or chr_num > 12:
            raise ValueError("[ERROR] Please choose chromosome number between 1 to 12 only.")
        else:
            ref_chr = read_csv("./data/ind-rg.csv", index_col=0)
            select_chr = ref_chr[ref_chr.chr==chr_num]
            df_chr = df[df.columns.intersection(list(map(lambda x, y: x + "_" + y, select_chr.id, select_chr.ref)))]
            
            # Invoke error and warning
            if len(df_chr.columns) > 1000:
                raise ValueError("[ERROR] In this version, the program cannot consume more than 1,000 SNPs.")
            elif len(df_chr.columns) > 500:
                print("[WARNING] Computational cost (CPU and RAM usages) will be high, since you have more than 500 SNPs.")

            return df_chr

class SNPinfo():
    def __init__(self, snps : DataFrame, *args, **kwargs) -> None:
        super(SNPinfo, self).__init__()
        self.snps = snps
        
    def check_for_unassigned_class_val(self, data : Series, class_ : list = [0, 1, 2], *args, **kwargs) -> Series:
        # if the SNP only have 0 value, then we should append {1:0, 2:0}
        # if the SNP only have 0 and 1 values, then we should append {2:0}
        for i in set(data.keys()) ^ set(class_):
            data[i] = 0
        return data
    
    def get_prob(self, data : Series, N : int, *args, **kwargs) -> list:
        data = Counter(data)
        data = self.check_for_unassigned_class_val(data)
        return list(map(lambda x: float(x/N), data.values())) # assume that each SNP has the same prob (hence, uniform dist.)
    
    def entropy(self, data : Series, joint : bool = False, *args, **kwargs) -> float:
        N = len(self.snps)
        
        if joint:
            N *= len(self.snps.columns)
            p = [self.get_prob(p, N) for p in data]
        else:
            p = self.get_prob(data, N)
        return -nansum(p * log2(p))
    
    def snp_combined_info(self, end_idx : int, *args, **kwargs) -> tuple:
        data = self.snps.columns[:end_idx]
        entropy_ = [self.entropy(self.snps[i]) for i in data]
        joint_entropy_ = round(self.entropy(self.snps[self.snps.columns[:end_idx]].T.values, joint=True), 5)

        mutual_info_ = [round(mis(self.snps[i], self.snps[data[len(data)-1]]), 5) for i in data]
        joint_entropy_mi = self.entropy(self.snps.T.values[:-1], joint=True)
        joint_entropy_lower_bound = joint_entropy_mi + self.entropy(self.snps.T.values[-1]) - max(mutual_info_)
        joint_entropy_upper_bound = joint_entropy_mi + self.entropy(self.snps.T.values[-1]) - round(sum(mutual_info_), 5)

        # total correlation, or multi-information
        total_corr_ = round(sum(entropy_) - joint_entropy_, 5)
        est_total_corr_lower = round(sum(entropy_) - joint_entropy_lower_bound, 5)
        est_total_corr_upper = round(sum(entropy_) - joint_entropy_upper_bound, 5)
        return joint_entropy_, total_corr_, est_total_corr_lower, est_total_corr_upper