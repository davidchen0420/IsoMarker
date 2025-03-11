import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc

import matplotlib.pyplot as plt
import seaborn as sns

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from jenkspy import JenksNaturalBreaks

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import gmean
from scipy.stats import zscore
from scipy.stats import rankdata

from upsetplot import from_contents
from upsetplot import UpSet

from adjustText import adjust_text
from itertools import chain


"""
Visualization Functions
"""
def lineage_gex(adata, gene, lineage, log=True):

    plot_df = pd.DataFrame(adata.to_df()[gene])

    plot_df['Lineage'] = adata.obs[lineage]

    order = plot_df.groupby('Lineage')[gene].mean().sort_values(ascending=False).index

    fig, ax = plt.subplots(1,1,figsize=(10,5))
    sns.boxplot(data = plot_df,
                  x = 'Lineage',
                  y = gene,
                   palette='Blues_r',
                  ax=ax,
               order=order,
               fliersize=5,
               linewidth=1,
               flierprops={"marker": "."})

    ax.set_ylabel('Normalized Gene\nExpression', size=20)
    ax.set_xlabel('Lineage', size=20)
    ax.set_title(gene, size=20)

    ax.set_xticklabels([x.replace("_", "-") for x in order], size=15, ha='right')

    ax.tick_params('x', rotation=45)
    ax.tick_params('both', length=5, width=1, labelsize=15)

    [x.set_linewidth(1) for x in ax.spines.values()]
    ax.spines[['right', 'top']].set_visible(False)
    
    if log is True:
        ax.set_yscale('log')

    plt.show()
    
    
def pca_plot_lineage(adata, lineage, batch=None):
    
    if batch != None:
        sc.pp.combat(adata, key=batch, inplace=True)

    sc.tl.pca(adata, svd_solver='arpack')

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    sc.pl.pca(adata, color=lineage, add_outline=False,
                   legend_fontsize=12, legend_fontoutline=2,frameon=False,
                    cmap='jet', ax=ax)

def umap_plot_lineage(adata, lineage, batch=None):
    
    if batch != None:
        sc.pp.combat(adata, key=batch, inplace=True)

    sc.tl.pca(adata, svd_solver='arpack')
    
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    sc.tl.umap(adata)

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    sc.pl.umap(adata, color=lineage, add_outline=False,
                   legend_fontsize=12, legend_fontoutline=2,frameon=False,
                    cmap='jet', ax=ax)

   
def pca_plot_marker(adata, marker, batch=None):
    
    if batch != None:
        sc.pp.combat(adata, key=batch, inplace=True)
        
    sc.tl.pca(adata, svd_solver='arpack')

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    sc.pl.pca(adata, color=marker, add_outline=False,
                   legend_fontsize=12, legend_fontoutline=2,frameon=False,
                    cmap='Greens', ax=ax)

def umap_plot_marker(adata, marker, batch=None):
    
    if batch != None:
        sc.pp.combat(adata, key=batch, inplace=True)

    sc.tl.pca(adata, svd_solver='arpack')
    
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    sc.tl.umap(adata)

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    sc.pl.umap(adata, color=marker, add_outline=False,
                   legend_fontsize=12, legend_fontoutline=2,frameon=False,
                    cmap='Greens', ax=ax)
    
def shap_summary_plot(adata, shap_values, lineage, max_display=20, cmap='RdBu_r'):
    shap.summary_plot(np.array(shap_values[lineage]), adata.to_df(), max_display=20, cmap=cmap, 
                  color_bar_label='Gene Expression')
    
def shap_dependency_plot(adata, shap_values, lineage, marker, reference, max_display=20, cmap='RdBu_r'):
    
    shap_figure = shap.dependence_plot(marker, np.array(shap_values[lineage]), adata.to_df(), interaction_index=reference, cmap=cmap, show=False)
    
    # Get the current figure and axes objects. from @GarrettCGraham code
    fig, ax = plt.gcf(), plt.gca()

    # Modifying main plot parameters
    ax.tick_params(labelsize=15)
    ax.set_ylabel(marker+" SHAP", fontsize=20)
    ax.set_xlabel(marker+" Gene Expression", fontsize=20)

    # Get colorbar
    cb_ax = fig.axes[1] 

    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel(reference + " Gene Expression", fontsize=20)

    plt.show()
    
def upset_plot(lineage, lfc_data, auc_data, top_marker_df, auc_threshold=0.6, z_threshold = 2, p_threshold = 0.05):

    plot_lfc = lfc_data[lineage]
    plot_lfc.dropna(inplace=True)
    
    plot_auc = auc_data[auc_data['group'] == lineage]

    volcano_df = pd.DataFrame()
    volcano_df["gene"] = plot_lfc.index
    volcano_df["lfc"] = list(plot_lfc['log2FoldChange'])
    volcano_df["z"] = zscore(volcano_df["lfc"]) 
    volcano_df['p-value'] = list(plot_lfc['padj'].replace(0,1e-100))

    enriched = volcano_df[(volcano_df['z'] > z_threshold) & (volcano_df['p-value'] < p_threshold)].copy()
    depleted = volcano_df[(volcano_df['z'] < -z_threshold) & (volcano_df['p-value'] < p_threshold)].copy()

    predictive_gene_list = list(top_marker_df[lineage].dropna().index)
    enriched_gene_list = enriched['gene'].tolist()
    depleted_gene_list = depleted['gene'].tolist()
    high_auc_gene_list = plot_auc[(plot_auc['auc'] > auc_threshold)]['names'].tolist()
    low_auc_gene_list = plot_auc[(plot_auc['auc'] < (1-auc_threshold))]['names'].tolist()

    hit_dict = {"Enriched Gene List": enriched_gene_list,
               "High AUC Gene List": high_auc_gene_list,
                "Predictive Gene List": predictive_gene_list,
               "Depleted Gene List": depleted_gene_list,
               "Low AUC Gene List": low_auc_gene_list}

    enriched_upset_df = from_contents(hit_dict)
    upset = UpSet(enriched_upset_df, subset_size='count', show_counts=True)

    upset.style_subsets(present=["Predictive Gene List", 
                                 "Enriched Gene List",
                                "High AUC Gene List"], 
                        absent=["Depleted Gene List",
                                "Low AUC Gene List"],
                        facecolor="royalblue", 
                        label="Positive Marker")

    upset.style_subsets(present=["Predictive Gene List", 
                                 "Depleted Gene List",
                                "Low AUC Gene List"], 
                        absent=["Enriched Gene List",
                                "High AUC Gene List"],
                        facecolor="firebrick", 
                        label="Negative Marker")

    upset.plot()

    output_df = enriched_upset_df.reset_index()
 
    plt.show()

    return output_df

def corr_plot(df, gene_list, corr='pearson'):

    def pearson_function(df):
        dfcols = pd.DataFrame(columns=df.columns)
        pvalues = dfcols.transpose().join(dfcols, how='outer')
        rvalues = dfcols.transpose().join(dfcols, how='outer')
        for r in df.columns:
            for c in df.columns:
                tmp = df[df[r].notnull() & df[c].notnull()]
                pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
                rvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[0], 4)
        return rvalues, pvalues
    
    def spearman_function(df):
        dfcols = pd.DataFrame(columns=df.columns)
        pvalues = dfcols.transpose().join(dfcols, how='outer')
        rvalues = dfcols.transpose().join(dfcols, how='outer')
        for r in df.columns:
            for c in df.columns:
                tmp = df[df[r].notnull() & df[c].notnull()]
                pvalues[r][c] = round(spearmanr(tmp[r], tmp[c])[1], 4)
                rvalues[r][c] = round(spearmanr(tmp[r], tmp[c])[0], 4)
        return rvalues, pvalues

    input_df = df.loc[:, list(set(df.columns) & set(gene_list))].copy()
    
    plot_df = input_df
    if corr == 'pearson':
        r, p = pearson_function(input_df)
    elif corr == 'spearman':
        r, p = spearman_function(input_df)
    r = r.astype(float).round(3)
    
    for row in p.index:
        for column in p.columns:
            if p.loc[row, column] < 0.001:
                p.loc[row, column] = "\n***"
            elif p.loc[row, column] < 0.01:
                p.loc[row, column] = "\n**"
            elif p.loc[row, column] < 0.05:
                p.loc[row, column] = "\n*"
            else:
                p.loc[row, column] = ""
    
    annot_df = r.astype(str) + "" + p.astype(str)
    
    fig , ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(plot_df.astype(float).corr(),
               ax=ax,
               cmap="RdBu", 
                annot=False, #annot_df.values
               vmin=-1,
               vmax=1,
               linewidths=1,
            linecolor='white',
               square=True,
               annot_kws={"fontsize":10},
               cbar_kws={"shrink": 0.8},
               fmt='')
    
    ax.tick_params(axis='x', which='major', labelsize=18, width=2, length=5, rotation=45)
    ax.tick_params(axis='y', which='major', labelsize=18, width=2, length=5, rotation=0)
    ax.set_xticklabels(input_df.columns, ha="right")
    
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=18)
    cax.set_ylabel("Correlation Coefficient",size=18, labelpad=10)
    
    plt.show()
    
def volcano_plot(lineage, lfc_data, auc_data, shap_data, x_axis='Log2 Fold Change', y_axis='P-Value', size='AUC', score='GMean(LFC_AUC_-log10(Padj))', auc_threshold=0.6, z_threshold = 2, p_threshold = 0.05):

    lineage_output_data = output_data(lineage=lineage,
                lfc_data = lfc_data,
    auc_data = auc_data,
    shap_data = shap_data)
    
    volcano_df = pd.DataFrame()
    volcano_df["gene"] = list(lineage_output_data.index)
    volcano_df["lfc"] = list(lineage_output_data[x_axis])
    volcano_df["z"] = list(zscore(volcano_df["lfc"]))
    volcano_df['p-value'] = list(lineage_output_data[y_axis])
    volcano_df['size'] = list(np.abs(lineage_output_data[size] - 0.5) + 0.5)
    volcano_df['score'] = list(lineage_output_data[score])
    
    volcano_df['p-value'] = volcano_df['p-value'].replace(0,1e-100)
    
    enriched = volcano_df[(volcano_df['z'] > z_threshold) & (volcano_df['p-value'] < p_threshold)].copy()
    depleted = volcano_df[(volcano_df['z'] < -z_threshold) & (volcano_df['p-value'] < p_threshold)].copy()
    
    enriched_gene_list = enriched['gene'].tolist()
    depleted_gene_list = depleted['gene'].tolist()

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    
    ax.scatter(x=volcano_df['z'], y=-np.log10(volcano_df['p-value']),
               s=2,
               color="darkgrey")
    
    
    ax.scatter(x=enriched['z'], y=-np.log10(enriched['p-value']),
               s=(enriched['size'] * 5) ** 4,
               color="seagreen",
               lw=1,
              edgecolor="black")
    
    
    ax.scatter(x=depleted['z'], y=-np.log10(depleted['p-value']),
               s=(depleted['size'] * 5) ** 4,
               color="indianred",
               lw=1,
              edgecolor="black")
    
    [x.set_linewidth(2) for x in ax.spines.values()]
    ax.tick_params(axis='both', which='major', labelsize=25, width=2, length=5)
    ax.set_ylabel("$-log_{10}$(p-value)", size=30, labelpad=20)
    ax.set_xlabel("Z-score", size=30, labelpad=20)
    
    ax.set_ylim(0,)
    
    annot_df = volcano_df[volcano_df['gene'].isin(enriched_gene_list + depleted_gene_list)]
    texts = [ax.text(annot_df.loc[x, 'z'], 
                     -np.log10(annot_df.loc[x, 'p-value']), 
                    annot_df.loc[x,'gene'], size=15) for x in annot_df.index]
    adjust_text(texts,
                arrowprops=dict(arrowstyle="-", color='black', lw=1))
    
    from matplotlib.lines import Line2D
    
    legend_elements = [Line2D([0], [0], marker='o', color='seagreen', label='DEG Enriched',
                              markerfacecolor='seagreen', markersize=25, linestyle='', markeredgecolor='black'),
                      Line2D([0], [0], marker='o', color='indianred', label='DEG Depleted',
                              markerfacecolor='indianred', markersize=25, linestyle='', markeredgecolor='black')
                      ]
    leg_type = plt.legend(handles=legend_elements, frameon=False, fontsize=25, bbox_to_anchor=(1,1),
                         title = '')
    leg_type.get_title().set_fontsize('25')
    fig.add_artist(leg_type)
    
    plt.show()

def shap_dot_plot(df, adata, marker):

    if type(marker) == dict:
        marker_list = list(chain.from_iterable(marker.values()))
    else:
        marker_list = marker
    
    marker_df = df[marker_list].copy()
    adata_top_marker = ad.AnnData(marker_df)
    adata_top_marker.obs = adata.obs
    adata_top_marker.var_names_make_unique()
    
    sc.set_figure_params(scanpy=True, fontsize=15)
    mp = sc.pl.dotplot(adata_top_marker, marker, groupby='Lineage', figsize =(len(marker_list),len(set(adata_top_marker.obs['Lineage']))/3), cmap='RdBu', return_fig=True,
                       standard_scale='var',var_group_rotation=45)
    mp.show()
    
def lfc_heatmap_plot(adata, lineage_label, df, marker):

    lineage_set = list(set(adata.obs[lineage_label]))

    lineage_data = []
    for x in lineage_set:
        lineage_data.append(df[x]['log2FoldChange'])

    plot_df = pd.concat(lineage_data, axis=1)
    plot_df.columns = lineage_set

    plot_df = plot_df.loc[list(set(plot_df.index) & set(marker)),:].copy()
    plot_df.fillna(0, inplace=True)

    g = sns.clustermap(plot_df, row_cluster=False, col_cluster=True, figsize=(plot_df.shape[1],plot_df.shape[0]*1.5),
                     cmap='RdBu_r', xticklabels=True, yticklabels=True, center=0.5,
                  linewidths=0.5, vmin=0, vmax=1,standard_scale=1)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 16, rotation=45, ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 16, rotation=0)
    cax = g.figure.axes[-1]
    cax.tick_params(labelsize=16)
    
    plt.show()
    
def auc_heatmap_plot(adata, lineage_label, df, marker):

    df.sort_values(['group', 'names'], inplace=True)
    lineage_set = list(set(adata.obs[lineage_label]))
    plot_df = pd.DataFrame()
    for x in lineage_set:
        plot_df[x] = list(df[df['group'] == x]['auc'])
    plot_df.index = list(df[df['group'] == x]['names'])
    
    plot_df = plot_df.loc[list(set(plot_df.index) & set(marker)),:].copy()
    plot_df.fillna(0, inplace=True)

    
    g = sns.clustermap(plot_df, row_cluster=False, col_cluster=True, figsize=(plot_df.shape[1],plot_df.shape[0]*1.5),
                     cmap='RdBu_r', xticklabels=True, yticklabels=True, center=0.5,
                  linewidths=0.5, vmin=0, vmax=1,standard_scale=1)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 16, rotation=45, ha='right')
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 16, rotation=0)
    cax = g.figure.axes[-1]
    cax.tick_params(labelsize=16)
    
    plt.show()
    


"""
Analysis Functions
"""




def pydeseq2_1_vs_rest(adata, cluster_label, reference, design_factors, log_add1=1):
    
    pydeseq_adata = adata.copy()
    pydeseq_adata.obs[cluster_label] = ['Other' if x != reference else reference for x in pydeseq_adata.obs[cluster_label]]
    pydeseq_adata.obs[cluster_label] = pydeseq_adata.obs[cluster_label].astype(str)

    try:
        pydeseq_adata.X = adata.X.todense() + log_add1
        pydeseq_adata.X = pydeseq_adata.X.astype(int)
    except:
        pass

    dds = DeseqDataSet(
        counts=pydeseq_adata.to_df().astype(int),
        metadata=pydeseq_adata.obs,
        ref_level=[cluster_label, 'Other'],
        design_factors=design_factors,
        refit_cooks=True,
    )

    dds.deseq2()

    stat_res = DeseqStats(dds,
                    contrast=[cluster_label, 'Other', reference.replace('_', '-')])

    stat_res.summary()

    stat_res.lfc_shrink(coeff=cluster_label + "_" + reference.replace('_', '-') + "_vs_Other")

    
    return stat_res.results_df

def pydeseq2_1_vs_rest_wrapper(adata, cluster_label, design_factors):
    
    lineage_set = list(set(adata.obs[cluster_label]))
    
    deseq_dict = {}
    
    for reference in lineage_set:

        deseq_dict[reference] = pydeseq2_1_vs_rest(adata=adata, cluster_label=cluster_label, reference=reference, design_factors=design_factors)

    return deseq_dict
        
def get_expr(adata, layer=None):
    """Get expression matrix from adata object."""
    if layer is not None:
        x = adata.layers[layer]
    else:
        x = adata.X
        
    if hasattr(x, "todense"):  # Convert sparse matrix to dense if necessary
        expr = np.asarray(x.todense())
    else:
        expr = np.asarray(x)

    return expr

def expr_auroc_over_groups(expr, uni_groups, groups):
    """Calculate AUROC for each feature over groups using proper rank assignment."""
    n_features = expr.shape[1]
    auroc = {group: np.zeros(n_features) for group in uni_groups}

    for group in uni_groups:
        binary_labels = np.array(groups) == group
        for i in range(n_features):
            pos = expr[binary_labels, i]
            neg = expr[~binary_labels, i]

            # Combine positive and negative, and calculate ranks
            combined = np.concatenate([pos, neg])
            ranks = rankdata(combined)  # Proper rank assignment with tie handling

            # Extract ranks for the positive group
            rank_pos = ranks[:len(pos)]

            # Calculate AUROC
            auroc[group][i] = (rank_pos.sum() - (len(pos) * (len(pos) + 1) / 2)) / (len(pos) * len(neg))
    return auroc
    
def wilcoxauc(adata, cluster_label, layer=None):
    """
    Wilcoxon AUC calculation using basic NumPy.

    Parameters:
        adata: AnnData object
        cluster_label: Name of the group column in adata.obs
        layer: Layer of data to use, or None for .X

    Returns:
        DataFrame with Wilcoxon AUC results for each group and feature.
    """

    expr = get_expr(adata, layer=layer)
    groups = adata.obs[cluster_label].tolist()
    uni_groups = adata.obs[cluster_label].unique()
    auroc = expr_auroc_over_groups(expr, uni_groups, groups)

    # Perform Wilcoxon ranking using Scanpy
    if layer is not None:
        features = adata.var.index
        sc.tl.rank_genes_groups(adata, cluster_label, layer=layer, use_raw=False,
                                method='wilcoxon', key_added="wilcoxon")
    else:
        features = adata.var.index
        sc.tl.rank_genes_groups(adata, cluster_label,
                                method='wilcoxon', key_added="wilcoxon")

    # Create a DataFrame for AUROC values
    auroc_df = pd.DataFrame(auroc).T
    auroc_df.columns = features
    auroc_df.index = uni_groups
    auroc_df=auroc_df.T

    # Combine Scanpy statistics with AUROC values
    res = pd.DataFrame()
    for group in uni_groups:
        cstats = sc.get.rank_genes_groups_df(adata, group=group, key='wilcoxon')
        cauc = pd.DataFrame(auroc_df[group]).reset_index().rename(columns={'index': 'names', group: 'auc'})
        cres = pd.merge(cstats, cauc, on='names')
        cres['group'] = group
        res = pd.concat([res, cres])

    res = res.reset_index(drop=True)
    return res

def shap_analysis(adata, cluster_label, layer = None, model=RandomForestClassifier(random_state=0)):

    if layer is not None:
        X = pd.DataFrame(data = get_expr(adata, layer='vst'),
               index = adata.obs.index,
               columns = adata.var.index)
    else:
        X = adata.to_df()

    le = LabelEncoder()
    y = le.fit_transform(adata.obs[cluster_label]) 


    mapping = dict(zip(le.classes_, range(len(le.classes_))))

    model = model.fit(X, y)
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    shap_dict = {}
    for x in mapping.keys():
        output_df = pd.DataFrame(index = adata.obs.index,
                 columns = adata.var.index,
                 data = shap_values[:,:,mapping[x]])

        shap_dict[x] = output_df

    return shap_dict

def jenks_shap(adata, lineage, shap_values):

    top_marker_df = pd.DataFrame(index=adata.var.index)
    shap_df = pd.DataFrame(index=adata.var.index)
    
    lineage_set = list(set(adata.obs[lineage]))
    
    for lineage in lineage_set:
        x = np.abs(shap_values[lineage]).mean(0)
    
        jnb = JenksNaturalBreaks(2) 
        jnb.fit(x) 
        jnb_break = jnb.breaks_[1]
    
        shap_df[lineage] = x
        
        top_marker = x[x>jnb_break].index
        top_marker_df[lineage] = np.nan
        for x in top_marker:
            top_marker_df.loc[x, lineage] = 'Marker'
    
    return shap_df, top_marker_df
        
def output_data(lineage, lfc_data, auc_data, shap_data):

    plot_lfc = lfc_data[lineage]
    plot_auc = auc_data[auc_data['group'] == lineage]
    plot_auc = plot_auc[plot_auc['names'].isin(plot_lfc.index)]
    shap_data = shap_data.loc[list(set(shap_data.index) & set(plot_lfc.index)), :]
    
    plot_lfc.sort_index(inplace=True)
    plot_auc.sort_values('names', inplace=True)
    shap_data.sort_index(inplace=True)
    
    output_df = pd.DataFrame(index = plot_lfc.index)
    
    output_df['Mean Expression'] = plot_lfc['baseMean']
    output_df['Log2 Fold Change'] = plot_lfc['log2FoldChange']
    output_df['Log2 Fold Change Standard Error'] = plot_lfc['lfcSE']
    output_df['P-Value'] = plot_lfc['pvalue']
    output_df['Adjusted P-Value'] = plot_lfc['padj']
    output_df['Wilcoxon Score'] = list(plot_auc['scores'])
    output_df['AUC'] = list(plot_auc['auc'])
    output_df['SHAP'] = list(shap_data[lineage])

    gmean_list = []
    for x in output_df.index:
        lfc_val = np.abs(output_df.loc[x, ['Log2 Fold Change']].values[0])
        p_val = np.abs(-np.log10(output_df.loc[x, ['P-Value']].values[0]))
        auc_val = np.abs(output_df.loc[x, ['AUC']].values[0])
    
        if output_df.loc[x, ['Log2 Fold Change']].values[0] > 0:
            gmean_list.append(gmean([lfc_val, p_val, auc_val]))
        else:
            gmean_list.append(-gmean([lfc_val, p_val, auc_val]))
    output_df['GMean(LFC_AUC_-log10(Padj))'] = gmean_list

    output_df.sort_values('GMean(LFC_AUC_-log10(Padj))', ascending=False, inplace=True)
            
    return output_df

def prep_anndata(raw_data_file_path, metadata_file_path, norm_data_file_path=None):

    raw_data = pd.read_csv(raw_data_file_path, index_col=0)
    metadata = pd.read_csv(metadata_file_path, index_col=0)

    adata = ad.AnnData(X = raw_data, obs = metadata)

    if norm_data_file_path is not None:
        norm_data = pd.read_csv(norm_data_file_path, index_col=0)
        adata.layers['vst'] = norm_data
    else:
        pass

    return adata

def consensus_isomarker(raw_data_file_path, norm_data_file_path, metadata_file_path, cluster_label, design_factors, shap_model=RandomForestClassifier(random_state=0), norm_layer=None):

    output_dict = {}
    
    adata = prep_anndata(raw_data_file_path, metadata_file_path, norm_data_file_path)
    
    lfc_data = pydeseq2_1_vs_rest_wrapper(adata=adata, cluster_label=cluster_label, design_factors=design_factors)
    
    auc_data = wilcoxauc(adata = adata, cluster_label = cluster_label, layer=norm_layer)
    
    shap_data = shap_analysis(adata=adata, cluster_label = cluster_label, layer=norm_layer, model=shap_model)
    shap_summary, top_shap_marker_df = jenks_shap(adata=adata, shap_values=shap_data, lineage=cluster_label)
    
    for cluster in list(set(adata.obs[cluster_label])):
    
        cluster_output_data = output_data(lineage=cluster,
                    lfc_data = lfc_data,
        auc_data = auc_data,
        shap_data = shap_summary)
        
        output_dict[cluster] = cluster_output_data
    
    output_dict['Top SHAP Marker'] = top_shap_marker_df
    output_dict['SHAP Raw Output'] = shap_summary

    return output_dict