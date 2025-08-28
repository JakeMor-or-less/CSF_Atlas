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

# Load filtered atlas data
adata = sc.read_h5ad('')

# check the atlas data
print(adata)
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")
print(adata.obs.head())
adata.obs

# Subset samples with "control" in the disease_group column
cd8 = adata[(adata.obs['cell_type'] == 'CD8 T cells')].copy()

old = cd8[(cd8.obs['organ'] == 'CSF') & 
                    (cd8.obs['disease_group'] == 'control') &
                    (cd8.obs['age_comparison'] == '≥60 years')].copy()

young = cd8[(cd8.obs['organ'] == 'CSF') & 
                    (cd8.obs['disease_group'] == 'control') &
                    (cd8.obs['age_comparison'] == '≤25 years')].copy()

adata_cd8_coy = old.concatenate(young)

# Check cell overlap
print(f"Processed cells: {len(cd8)}")
print(f"Raw count cells: {len(cd8.layers['raw_counts'])}")

# Count cells per sample
print("Cells per sample:")
print(adata_cd8_coy.obs['sample'].value_counts().sort_index())

# Or get it as a sorted table
sample_counts = adata_cd8_coy.obs['sample'].value_counts().sort_index()
print(f"\nTotal samples: {len(sample_counts)}")
print(f"Cells per sample range: {sample_counts.min()} - {sample_counts.max()}")
print(f"Mean cells per sample: {sample_counts.mean():.1f}")

# convert gene names
translator = sp.translator(hub = hubs.hubs[('blood')], data = adata_cd8_coy)

# run senepy
adata_cd8_coy.obs['sen_score'] = sp.score_all_cells(adata_cd8_coy, hubs.hubs[('blood')], 
                                     identifiers = ['sex', 'age_comparison'])

# verify it worked
adata_cd8_coy.obs.head()

# subset young samples for putative sen
young = adata_cd8_coy[adata_cd8_coy.obs['age_comparison'] == '≤25 years'].copy()

# putative sen scoring
e = young.obs.sen_score.mean() #distribution mean
std = young.obs.sen_score.std() #distribution std
thresh = e + 3*std
thresh
def is_putative_sen(x):
    if x >= thresh:
        return 1
    else:
        return 0
adata_cd8_coy.obs['putative_sen'] = adata_cd8_coy.obs.sen_score.map(is_putative_sen)

# verify it worked
adata_cd8_coy.obs.head()

# Filter to only include the two young and old samples
comparison_data = adata_cd8_coy.obs[adata_cd8_coy.obs['age_comparison'] != 'exclude']

# getting senescence proportions
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

# save the object
adata_cd8_coy.write_h5ad('cd8_scored_7-29-25.h5ad')
