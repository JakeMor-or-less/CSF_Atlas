import os
import senepy as sp
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random 

random.seed(42)

# Set working directory
os.chdir('/fs/scratch/PAS2598/Morales/CSF_workspace/csf')

# load senepy hubs
hubs = sp.load_hubs(species = 'Human') 
hubs.metadata

# load blood hub
blood_hub = hubs.metadata[(hubs.metadata.tissue == 'blood')]
blood_hub

# merge blood hub
hubs.merge_hubs(blood_hub, new_name = 'blood')
gene_list = hubs.get_genes(('blood')[:10])
gene_list

# export blood hub gene list
df = pd.DataFrame(gene_list, columns=['Gene'])
df.to_csv('blood_merged_gene_list.csv', index=False)

# Load atlas data
adata = sc.read_h5ad('adata_raw_age_filt_500.h5ad')
# not with >=500 counts
#adata = sc.read_h5ad('adata_with_raw_age_filtered.h5ad')

# Check what you loaded
print(adata)
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")

# View the metadata (equivalent to @meta.data in R)
print(adata.obs.head())


# In[19]:


adata.obs


# In[20]:


# Subset samples with "control" in the disease_group column
cd8 = adata[(adata.obs['cell_type'] == 'CD8 T cells')].copy()

old = cd8[(cd8.obs['organ'] == 'CSF') & 
                    (cd8.obs['disease_group'] == 'control') &
                    (cd8.obs['age_comparison'] == '≥60 years')].copy()

young = cd8[(cd8.obs['organ'] == 'CSF') & 
                    (cd8.obs['disease_group'] == 'control') &
                    (cd8.obs['age_comparison'] == '≤25 years')].copy()

adata_cd8_coy = old.concatenate(young)


# In[21]:


# Check cell overlap
print(f"Processed cells: {len(cd8)}")
print(f"Raw count cells: {len(cd8.layers['raw_counts'])}")


# In[22]:


# Count cells per sample
print("Cells per sample:")
print(adata_cd8_coy.obs['sample'].value_counts().sort_index())

# Or get it as a sorted table
sample_counts = adata_cd8_coy.obs['sample'].value_counts().sort_index()
print(f"\nTotal samples: {len(sample_counts)}")
print(f"Cells per sample range: {sample_counts.min()} - {sample_counts.max()}")
print(f"Mean cells per sample: {sample_counts.mean():.1f}")


# In[23]:


translator = sp.translator(hub = hubs.hubs[('blood')], data = adata_cd8_coy)


# In[24]:


#score_hub returns a list and we can save this directly to the adata.obs dataframe

# CD4_subset = adata[adata.obs['cell_type'] == 'CD4 T cells'].copy()

adata_cd8_coy.obs['sen_score'] = sp.score_all_cells(adata_cd8_coy, hubs.hubs[('blood')], 
                                     identifiers = ['sex', 'age_comparison'])


# In[25]:


adata_cd8_coy.obs.head()


# In[26]:


young = adata_cd8_coy[adata_cd8_coy.obs['age_comparison'] == '≤25 years'].copy()


# In[27]:


e = young.obs.sen_score.mean() #distribution mean


# In[28]:


std = young.obs.sen_score.std() #distribution std


# In[29]:


thresh = e + 3*std
thresh


# In[30]:


#function to add senescent label
def is_putative_sen(x):
    if x >= thresh:
        return 1
    else:
        return 0


# In[31]:


#map function to a new row in adata.obs
adata_cd8_coy.obs['putative_sen'] = adata_cd8_coy.obs.sen_score.map(is_putative_sen)


# In[32]:


adata_cd8_coy.obs.head()


# In[34]:


# Filter to only include the two groups of interest
comparison_data = adata_cd8_coy.obs[adata_cd8_coy.obs['age_comparison'] != 'exclude']

# METHOD 1: Using sum/count (most explicit)
senescence_by_group_cell = comparison_data.groupby(['cell_type', 'age_comparison'])['putative_sen'].agg([
    lambda x: (x.sum() / len(x)),  # Proportion: count of 1s / total count
    'sum',  # Number of senescent cells
    'count'  # Total number of cells
]).reset_index()

# Rename columns for clarity
senescence_by_group_cell.columns = ['cell_type', 'age_comparison', 'proportion', 'n_senescent', 'n_total']
senescence_by_group_cell['percent_senescent'] = senescence_by_group_cell['proportion'] * 100

# Print summary statistics
print("CD4 senescence by cell type and age group:")
print(senescence_by_group_cell[['cell_type', 'age_comparison', 'n_senescent', 'n_total', 'percent_senescent']])
print("\nVerification - senescent/total:")
for _, row in senescence_by_group_cell.iterrows():
    print(f"{row['age_comparison']}: {row['n_senescent']}/{row['n_total']} = {row['percent_senescent']:.1f}%")

# Create the grouped bar plot
plt.figure(figsize=(6, 6))

# Create bar plot without error bars
ax = sns.barplot(
    data=senescence_by_group_cell,
    y='percent_senescent',
    hue='age_comparison',
    palette={'≤25 years': '#3498DB', '≥60 years': '#E74C3C'},
    errorbar=None  # No error bars
)

# Create a dictionary to store the x positions of bars for each cell type and age group
bar_positions = {}
cell_types = senescence_by_group_cell['cell_type'].unique()
num_cell_types = len(cell_types)

# Get the bar positions 
for i, cell_type in enumerate(cell_types):
    bar_positions[cell_type] = {}
    bar_positions[cell_type]['≤25 years'] = i - 0.2  # Position for young group
    bar_positions[cell_type]['≥60 years'] = i + 0.2   # Position for old group

# Add sample sizes and percentages
for _, row in senescence_by_group_cell.iterrows():
    cell_type = row['cell_type']
    age_group = row['age_comparison']
    n_senescent = row['n_senescent']
    n_total = row['n_total']
    percent = row['percent_senescent']
    
    # Calculate position
    x_pos = bar_positions[cell_type][age_group]
    
    # Determine y position for sample size - make it relative to bar height
    y_offset = 1.0  # Fixed offset above bar
    if percent > 5:  # For very short bars
        y_pos = percent + y_offset
    else:
        # For taller bars, place within the bar
        y_pos = percent - 0.05
    
    # Add detailed count information (senescent/total)
    plt.text(
        x_pos, 
        y_pos,
        f'{n_senescent}/{n_total}',  # Shows actual counts
        ha='center',
        va='center',
        fontsize=9,
        fontweight='bold',
        color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    )
    
    # Add percentage to the top of each bar
    plt.text(
        x_pos,
        percent + 0.1,  # Position slightly above the bar
        f'{percent:.1f}%',  # Format to 1 decimal place
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold',
        color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    )

# Formatting
plt.title('Proportion of Senescent Cells: Young vs Old Patients (Controls)', fontsize=15, fontweight='bold')
plt.ylabel('Senescent Cells (%)', fontsize=13)
plt.yticks(fontsize=10, fontweight='bold')
plt.xlabel('CD8 T cells', fontsize=12, rotation=45, ha='right')
plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
plt.legend(title='Age Group')
plt.tight_layout()

# Set spine width
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.tick_params(axis='x', width=2, length=6)
ax.tick_params(axis='y', width=1.5, length=6)

# Increase y-limit to accommodate the percentage labels
max_percent = max(senescence_by_group_cell['percent_senescent'])
plt.ylim(0, max_percent * 1.25)  # Increased headroom for labels

# Save as TIFF (publication quality)
plt.savefig('senepy_cd8_controls_oldvsyoung_7-25-25.png', dpi=300, bbox_inches='tight')
plt.savefig('senepy_cd8_controls_oldvsyoung_7-25-25.pdf', dpi=300, bbox_inches='tight')

plt.show()


# In[30]:


adata_cd8_coy.write_h5ad('cd8_scored_7-29-25.h5ad')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[36]:


adata_cd8_coy.obs.head()


# In[37]:


# Check your subset first
print("=== CD8 T CELL SUBSET ANALYSIS ===")
print(f"Shape: {adata_cd8_coy.shape}")
print(f"Available layers: {list(adata_cd8_coy.layers.keys())}")

# Check age group distribution
print(f"\nsene distribution:")
print(adata_cd8_coy.obs['putative_sen'].value_counts())

# Check samples per age group
print(f"\nSamples per sene:")
sample_counts = adata_cd8_coy.obs.groupby('putative_sen')['sample'].nunique()
print(sample_counts)

# Verify raw counts are still there
print(f"\nRaw counts layer max: {adata_cd8_coy.layers['raw_counts'].max():,.0f}")

# Create pseudobulk with this focused dataset
def create_pseudobulk_from_layer(adata, layer_name='raw_counts', sample_col='sample', condition_col='putative_sen'):
    """Create pseudobulk using raw counts layer"""
    
    count_matrix = adata.layers[layer_name]
    genes = adata.var_names
    
    print(f"Using layer '{layer_name}' with max value: {count_matrix.max():,.0f}")
    
    # Get sample info
    sample_info = adata.obs[[sample_col, condition_col]].drop_duplicates()
    print(f"Total samples: {len(sample_info)}")
    print(f"Samples by condition:")
    print(sample_info[condition_col].value_counts())
    
    # Create pseudobulk matrix
    pseudobulk_matrix = np.zeros((len(sample_info), len(genes)))
    
    for i, (_, sample_row) in enumerate(sample_info.iterrows()):
        sample_name = sample_row[sample_col]
        sample_cells = adata.obs[adata.obs[sample_col] == sample_name].index
        cell_indices = [list(adata.obs_names).index(cell) for cell in sample_cells]
        
        if len(cell_indices) > 0:
            pseudobulk_matrix[i, :] = count_matrix[cell_indices, :].sum(axis=0)
            
    pseudobulk_df = pd.DataFrame(
        pseudobulk_matrix.T,
        index=genes,
        columns=sample_info[sample_col].values
    )
    
    return pseudobulk_df, sample_info

# Run pseudobulk on your CD8 T cell subset
pseudobulk_cd8, sample_metadata_cd8 = create_pseudobulk_from_layer(adata_cd8_coy)

print(f"\nCD8 T CELL PSEUDOBULK RESULTS:")
print(f"Shape: {pseudobulk_cd8.shape}")
print(f"Max value: {pseudobulk_cd8.max().max():,.0f}")
print(f"Min value: {pseudobulk_cd8.min().min():,.0f}")

# Check senescence genes in this focused analysis
senescence_genes = ['CDKN1A', 'CDKN2A', 'GLB1', 'IL6R']
print(f"\nSenescence genes in CD8 T cells:")
for gene in senescence_genes:
    if gene in pseudobulk_cd8.index:
        values = pseudobulk_cd8.loc[gene].values
        non_zero_samples = (values > 0).sum()
        print(f"{gene}: max={values.max():,.0f}, mean={values.mean():,.1f}, expressed in {non_zero_samples}/{len(values)} samples")


# In[39]:


# Step 1: Filter and normalize your excellent pseudobulk data
print("=== FILTERING AND NORMALIZATION ===")

# Filter: >3 counts in at least 30% of samples (your approach)
min_counts = 3
min_samples = int(0.3 * len(pseudobulk_cd8.columns))  # 30% of 27 = 8 samples

genes_keep = (pseudobulk_cd8 > min_counts).sum(axis=1) >= min_samples
pseudobulk_filtered = pseudobulk_cd8.loc[genes_keep]

print(f"Genes before filtering: {len(pseudobulk_cd8)}")
print(f"Genes after filtering (>{min_counts} counts in ≥{min_samples} samples): {len(pseudobulk_filtered)}")

# Normalize to CPM
def normalize_cpm(counts_df):
    return counts_df.div(counts_df.sum(axis=0), axis=1) * 1e6

pseudobulk_cpm = normalize_cpm(pseudobulk_filtered)
pseudobulk_log_cpm = np.log2(pseudobulk_cpm + 1)

print(f"Final data shape: {pseudobulk_log_cpm.shape}")

# Check senescence genes survived
senescence_genes = ['CDKN1A', 'CDKN2A', 'GLB1', 'IL6R']
surviving = [g for g in senescence_genes if g in pseudobulk_log_cpm.index]
print(f"Senescence genes surviving filter: {surviving}")

# Step 2: Differential Expression Analysis
from scipy import stats

def run_differential_expression(log_cpm_df, metadata_df):
    """Run DE analysis on CD8 T cells"""
    
    # Get sample conditions
    sample_conditions = metadata_df.set_index('sample')['putative_sen']
    aligned_conditions = sample_conditions.loc[log_cpm_df.columns]
    
    non_sene = log_cpm_df.columns[aligned_conditions == 0]
    sene = log_cpm_df.columns[aligned_conditions == '1']
    
    print(f"non_sene: {len(non_sene)}")
    print(f"sene: {len(sene)}")
    
    results = []
    
    for gene in log_cpm_df.index:
        non_sene_expr = log_cpm_df.loc[gene, non_sene]
        sene_expr = log_cpm_df.loc[gene, sene]
        
        # T-test
        stat, pval = stats.ttest_ind(sene_expr, non_sene_expr, equal_var=False)
        
        # Calculate fold change
        mean_non_sene = non_sene_expr.mean()
        mean_sene = sene_expr.mean()
        log2fc = mean_sene - mean_non_sene
        
        results.append({
            'gene': gene,
            'log2fc': log2fc,
            'pvalue': pval,
            'mean_young': mean_non_sene,
            'mean_old': mean_sene
        })
    
    results_df = pd.DataFrame(results)
    
    # Multiple testing correction
    from statsmodels.stats.multitest import multipletests
    results_df['padj'] = multipletests(results_df['pvalue'], method='fdr_bh')[1]
    
    return results_df.sort_values('pvalue')

# Run differential expression
de_results_cd8 = run_differential_expression(pseudobulk_log_cpm, sample_metadata_cd8)

print(f"\nCD8 T CELL DIFFERENTIAL EXPRESSION RESULTS:")
print(f"Total genes tested: {len(de_results_cd8)}")
print(f"Significant genes (padj < 0.05): {sum(de_results_cd8['padj'] < 0.05)}")

# Check your senescence markers!
senescence_results = de_results_cd8[de_results_cd8['gene'].isin(senescence_genes)].copy()
print(f"\nSENESCENCE MARKER RESULTS:")
for _, row in senescence_results.iterrows():
    direction = "UP" if row['log2fc'] > 0 else "DOWN"
    significance = "***" if row['padj'] < 0.001 else "**" if row['padj'] < 0.01 else "*" if row['padj'] < 0.05 else "ns"
    print(f"{row['gene']}: {direction} {row['log2fc']:.2f} log2FC, p={row['pvalue']:.3f}, padj={row['padj']:.3f} {significance}")


# In[20]:


# Create volcano plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sig_genes = de_results_cd8['padj'] < 0.05

# All genes
plt.scatter(de_results_cd8['log2fc'], -np.log10(de_results_cd8['pvalue']), 
           c='lightgray', alpha=0.6, s=20)

# Significant genes
if sum(sig_genes) > 0:
    sig_data = de_results_cd8[sig_genes]
    plt.scatter(sig_data['log2fc'], -np.log10(sig_data['pvalue']), 
               c='red', alpha=0.7, s=30)

# Senescence genes
for _, row in senescence_results.iterrows():
    color = 'blue' if row['padj'] < 0.05 else 'orange'
    plt.scatter(row['log2fc'], -np.log10(row['pvalue']), 
               c=color, s=100, edgecolor='black')
    plt.annotate(row['gene'], (row['log2fc'], -np.log10(row['pvalue'])), 
                xytext=(5, 5), textcoords='offset points')

plt.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
plt.xlabel('log2 Fold Change (Old vs Young)')
plt.ylabel('-log10 p-value')
plt.title('CD8 T Cell Aging: Differential Expression')
plt.show()


# In[21]:


import matplotlib.pyplot as plt
import numpy as np

# Get data
genes = senescence_results['gene'].values
log2fc = senescence_results['log2fc'].values
padj = senescence_results['padj'].values

# Get sample groups
young_samples = pseudobulk_log_cpm.columns[sample_metadata_cd8.set_index('sample').loc[pseudobulk_log_cpm.columns, 'age_comparison'] == '≤25 years']
old_samples = pseudobulk_log_cpm.columns[sample_metadata_cd8.set_index('sample').loc[pseudobulk_log_cpm.columns, 'age_comparison'] == '≥60 years']

# Calculate means and standard errors
young_means = []
old_means = []
young_sems = []
old_sems = []

for gene in genes:
    young_expr = pseudobulk_log_cpm.loc[gene, young_samples]
    old_expr = pseudobulk_log_cpm.loc[gene, old_samples]
    
    young_means.append(young_expr.mean())
    old_means.append(old_expr.mean())
    young_sems.append(young_expr.std() / np.sqrt(len(young_expr)))
    old_sems.append(old_expr.std() / np.sqrt(len(old_expr)))

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create grouped bar plot
x = np.arange(len(genes))
width = 0.35

bars1 = ax.bar(x - width/2, young_means, width, yerr=young_sems, 
               label='≤25 years', color='lightblue', alpha=0.8, capsize=5)
bars2 = ax.bar(x + width/2, old_means, width, yerr=old_sems,
               label='≥60 years', color='lightcoral', alpha=0.8, capsize=5)

# Add p-values above bars
for i, p in enumerate(padj):
    # Find the highest point (bar + error bar)
    y_max = max(old_means[i] + old_sems[i], young_means[i] + young_sems[i])
    
    if p < 0.05:
        # Draw significance bracket for significant results
        bracket_height = y_max + 0.2
        ax.plot([i-width/2, i+width/2], [bracket_height, bracket_height], 'k-', linewidth=1)
        ax.plot([i-width/2, i-width/2], [bracket_height-0.05, bracket_height], 'k-', linewidth=1)
        ax.plot([i+width/2, i+width/2], [bracket_height-0.05, bracket_height], 'k-', linewidth=1)
        
        # Add p-value text
        if p < 0.001:
            p_text = f'p<0.001'
        elif p < 0.01:
            p_text = f'p={p:.3f}'
        else:
            p_text = f'p={p:.3f}'
            
        ax.text(i, bracket_height + 0.1, p_text, ha='center', fontsize=10, fontweight='bold')
    else:
        # For non-significant results, just show p-value above the higher bar
        text_height = y_max + 0.1
        p_text = f'p={p:.3f}'
        ax.text(i, text_height, p_text, ha='center', fontsize=10, color='gray')

ax.set_ylabel('Mean log2 CPM Expression', fontsize=12)
ax.set_xlabel('Senescence Markers', fontsize=12)
ax.set_title('Senescence Marker Expression: Young vs Old CD8 T Cells', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(genes)
ax.legend()

plt.tight_layout()
plt.show()


# In[23]:


# Add fold change thresholds to your volcano plot
import matplotlib.pyplot as plt
import numpy as np

# Create volcano plot with thresholds
plt.figure(figsize=(10, 8))

# Plot all genes
sig_genes = de_results_cd8['padj'] < 0.05
fc_threshold = 1.0  # log2FC threshold (2-fold change)

# Color genes by significance AND fold change
colors = []
for _, row in de_results_cd8.iterrows():
    if row['padj'] < 0.05 and abs(row['log2fc']) >= fc_threshold:
        if row['log2fc'] > 0:
            colors.append('red')      # Significant upregulation
        else:
            colors.append('blue')     # Significant downregulation
    elif row['padj'] < 0.05:
        colors.append('orange')       # Significant but small fold change
    elif abs(row['log2fc']) >= fc_threshold:
        colors.append('green')        # Large fold change but not significant
    else:
        colors.append('lightgray')    # Neither significant nor large fold change

plt.scatter(de_results_cd8['log2fc'], -np.log10(de_results_cd8['pvalue']), 
           c=colors, alpha=0.6, s=20)

# Add threshold lines
plt.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='p=0.05')
plt.axvline(fc_threshold, color='black', linestyle=':', alpha=0.5, label=f'log2FC=±{fc_threshold}')
plt.axvline(-fc_threshold, color='black', linestyle=':', alpha=0.5)

# Highlight senescence genes
for _, row in senescence_results.iterrows():
    plt.scatter(row['log2fc'], -np.log10(row['pvalue']), 
               c='purple', s=100, edgecolor='black', linewidth=2, zorder=5)
    plt.annotate(row['gene'], (row['log2fc'], -np.log10(row['pvalue'])), 
                xytext=(5, 5), textcoords='offset points', fontweight='bold')

plt.xlabel('log2 Fold Change (Old vs Young)')
plt.ylabel('-log10 p-value')
plt.title('CD8 T Cell Aging: Differential Expression\n(Thresholds: p<0.05, |log2FC|≥1)')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', label='Sig. Up (|log2FC|≥1)'),
    Patch(facecolor='blue', label='Sig. Down (|log2FC|≥1)'),
    Patch(facecolor='orange', label='Sig. but |log2FC|<1'),
    Patch(facecolor='green', label='|log2FC|≥1 but not sig.'),
    Patch(facecolor='purple', label='Senescence markers')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.grid(True, alpha=0.3)
plt.show()

# Count genes in each category
sig_and_fc = sum((de_results_cd8['padj'] < 0.05) & (abs(de_results_cd8['log2fc']) >= fc_threshold))
print(f"Genes with padj<0.05 AND |log2FC|≥{fc_threshold}: {sig_and_fc}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




