import numpy as np
import pandas as pd
import pingouin
from decimal import Decimal


def combine_summaries(outfile_name='total_summary.csv'):
    domains = ['Lunar Lander', 'Slime Volleyball']
    summary_file_names = ['lunar_lander_summary.csv', 'slime_volleyball_summary.csv']
    for d, s in zip(domains, summary_file_names):
        df = pd.read_csv(s, dtype = {'Algorithm': str})
        df['Domain'] = d
        if 'result_df' in vars():
            result_df = pd.concat([result_df, df])
        else:
            result_df = df
    result_df.to_csv(outfile_name, index=False)


def qd_coverage_table(summary_file_name='total_summary.csv', outfile_name='qd_coverage.txt'):
    summary_df = pd.read_csv(summary_file_name, dtype = {'Algorithm': str})
    algorithms = summary_df['Algorithm'].unique()
    domains = ['Lunar Lander', 'Slime Volleyball']
    metrics = ['QD-Score', 'Coverage']

    max_itr = summary_df.Iteration.max()
    mean_df = summary_df[summary_df.Iteration == max_itr].groupby(['Domain', 'Algorithm'])[metrics].mean()

    table_df = pd.DataFrame(
        index=algorithms,
        columns=pd.MultiIndex.from_product([domains, metrics]),
        dtype=str,
    )
    table_df.rename_axis('Algorithm')

    for d in domains:
        for a in algorithms:
            for m in metrics:
                mean = mean_df.loc[d, a][m]
                metric_str = f"{mean:,.2f}" if m == "QD-Score" else f"{mean:,.2f}\%"
                if mean == mean_df.loc[d].max()[m]:
                    metric_str = f"\\textbf{{{metric_str}}}"
                
                table_df[d, m][a] = metric_str

    with open(outfile_name, "w") as file:
        file.write("\\begin{table*}[t]\n")
        file.write("\\caption{Mean QD-score and coverage values after 10,000 iterations for each QD algorithm per domain.}\n")
        file.write("\\label{tab:results_new_domains}")
        file.write("\\centering")
        file.write("\\resizebox{1.0\linewidth}{!}{")
        file.write(
            table_df.to_latex(
                column_format="l" + "|rr" * len(domains),
                # multicolumn_format=
                escape=False,
            ))
        file.write("}\n\\end{table*}\n")


def one_way_anova_table(summary_file_name='total_summary.csv', outfile_name='one_way_anova.txt'):
    summary_df = pd.read_csv(summary_file_name, dtype = {'Algorithm': str})
    domains = ['Lunar Lander', 'Slime Volleyball']
    metrics = ['QD-Score', 'Coverage']
    max_itr = summary_df.Iteration.max()
    final_itr_df = summary_df[summary_df.Iteration == max_itr]

    anova_df = pd.DataFrame(
        index=domains,
        columns=pd.MultiIndex.from_product([metrics, ['F-value', 'p-value']]),
        dtype=str,
    )

    for d in domains:
        domain_df = final_itr_df[final_itr_df.Domain == d]
        for m in metrics:
            metric_anova_res = pingouin.anova(
                dv=m,
                between='Algorithm',
                data=domain_df,
                detailed=True
            )
            anova_df[m, 'F-value'][d] = f"F({metric_anova_res['DF'][0]}, {metric_anova_res['DF'][1]}) = {metric_anova_res['F'][0]:.2f}"
            anova_df[m, 'p-value'][d] = f"{Decimal(metric_anova_res['p-unc'][0]):.2E}"

    with open(outfile_name, "w") as file:
        file.write("\\begin{table*}[t]\n")
        file.write("\\caption{One-way ANOVA results in each domain.}\n")
        file.write("\\label{tab:one_way_anova_results}")
        file.write("\\centering")
        file.write("\\resizebox{1.0\linewidth}{!}{")
        file.write(
            anova_df.to_latex(
                column_format="l" + "|rr" * len(domains),
                # multicolumn_format=
                escape=False,
            ))
        file.write("}\n\\end{table*}\n")
        

def bonferroni_ttest(main_algo='CMA-MAE', summary_file_name='total_summary.csv'):
    summary_df = pd.read_csv(summary_file_name, dtype = {'Algorithm': str})
    algorithms = summary_df['Algorithm'].unique()
    other_algos = list(algorithms)
    other_algos.remove(main_algo)
    domains = ['Lunar Lander', 'Slime Volleyball']
    metrics = ['QD-Score', 'Coverage']
    max_itr = summary_df.Iteration.max()
    final_itr_df = summary_df[summary_df.Iteration == max_itr]

    ttest_res = {}
    for d in domains:
        domain_df = final_itr_df[final_itr_df.Domain == d]
        ttest_metric_res = {}
        for m in metrics:
            df = pd.concat(
                [
                    pingouin.ttest(
                        domain_df[domain_df.Algorithm == main_algo][m],
                        domain_df[domain_df.Algorithm == algo][m],
                        paired=False,
                        alternative="two-sided",
                    )[["T", "dof", "alternative", "p-val"]]
                    for algo in other_algos
                ],
                ignore_index=True,
            )

            # Some hypotheses require overriding bonf_n.
            bonf_n = len(df["p-val"])
            # Adapted from pingouin multicomp implementation:
            # https://github.com/raphaelvallat/pingouin/blob/c66b6853cfcbe1d6d9702c87c09050594b4cacb4/pingouin/multicomp.py#L122
            df["p-val"] = np.clip(df["p-val"] * bonf_n, None, 1)
            df["significant"] = np.less(
                df["p-val"],
                0.05,  # alpha
            )
            df = pd.concat(
                [
                    pd.DataFrame({
                        "Algorithm 1": [main_algo] * len(other_algos),
                        "Algorithm 2": other_algos,
                    }),
                    df,
                ],
                axis=1,
            )
            ttest_metric_res[m] = df
        
        ttest_res[d] = ttest_metric_res
    
    return ttest_res


def tukey_table(alpha=0.05, summary_file_name='total_summary.csv', outfile_name='tukey.txt'):
    summary_df = pd.read_csv(summary_file_name, dtype = {'Algorithm': str})
    algorithms = summary_df['Algorithm'].unique()
    domains = ['Lunar Lander', 'Slime Volleyball']
    max_itr = summary_df.Iteration.max()
    final_itr_df = summary_df[summary_df.Iteration == max_itr]

    raw_pairwise = {}
    for env in domains:
        domain_df = final_itr_df[final_itr_df.Domain == env]
        raw_pairwise[env] = pingouin.pairwise_tukey(domain_df, dv="QD-Score", between="Algorithm")

    # Processed to look like:
    #
    #               sep-CMA-MAE      CMA-MAE    ...
    # sep-CMA-MAE       N/A             >
    #   CMA-MAE          <
    processed_pairwise = pd.DataFrame(
        index=algorithms,
        columns=pd.MultiIndex.from_product([domains, algorithms]),
        dtype=object,
    )

    for env in raw_pairwise:
        # Algorithms can't compare with themselves.
        for algo in algorithms:
            processed_pairwise[env, algo][algo] = "N/A"

        # Use symbols to mark other comparisons.
        # pylint: disable = unused-variable
        for index, row in raw_pairwise[env].iterrows():
            p = row.loc["p-tukey"]
            a, b = row.loc["A"], row.loc["B"]
            if p > alpha:
                # No significant difference.
                processed_pairwise[env, b][a] = "~"
                processed_pairwise[env, a][b] = "~"
            else:
                # Significant result.
                if row.loc["T"] < 0:
                    processed_pairwise[env, b][a] = "<"
                    processed_pairwise[env, a][b] = ">"
                else:
                    processed_pairwise[env, b][a] = ">"
                    processed_pairwise[env, a][b] = "<"

    with open(outfile_name, "w") as file:
        file.write("\\begin{table*}[t]\n")
        file.write("\\caption{Pairwise comparisons for QD-score in each domain.}\n")
        file.write("\\label{tab:tukey_results}")
        file.write("\\centering")
        file.write("\\resizebox{1.0\linewidth}{!}{")
        latex_str = processed_pairwise.to_latex(
            column_format="l" + ("|"+"r"*len(algorithms))*len(domains),
            escape=False,  # Escape all the special < characters.
            multirow=True,
        )
        latex_str = (latex_str.replace("<", "$<$").replace(">", "$>$").replace(
            "~", "$-$"))
        file.write(latex_str)
        file.write("}\n\\end{table*}\n")


combine_summaries()
qd_coverage_table()
one_way_anova_table()
tukey_table()
print(bonferroni_ttest())