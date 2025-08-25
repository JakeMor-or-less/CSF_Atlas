#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import senepy as sp
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from scipy import stats

random.seed(42)

# Set working directory
os.chdir('/fs/scratch/PAS2598/Morales/CSF_workspace/csf')

# Check current working directory
print(os.getcwd())


# In[2]:


hubs = sp.load_hubs(species = 'Human') 
hubs.metadata


# In[3]:


blood_hub = hubs.metadata[(hubs.metadata.tissue == 'blood')]
blood_hub
hubs.merge_hubs(blood_hub, new_name = 'blood')


# In[4]:


# Load your h5ad file
adata = sc.read_h5ad('adata_raw_age_filt_500.h5ad')
# not with >=500 counts
#adata = sc.read_h5ad('adata_with_raw_age_filtered.h5ad')

# Check what you loaded
print(adata)
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")

# View the metadata (equivalent to @meta.data in R)
print(adata.obs.head())


# In[5]:


adata.obs


# In[6]:


# Subset samples
cd8 = adata[(adata.obs['cell_type'] == 'CD8 T cells')].copy()
cd8


# In[7]:


cd8.obs['disease'].unique()


# In[8]:


# List of samples to remove
samples_to_remove = ['IIH68490', 'GSM3984200']

# Create a boolean mask for samples to keep (those NOT in the removal list)
mask = ~cd8.obs['sample'].isin(samples_to_remove)

# Subset the AnnData object to keep only the desired samples
cd8 = cd8[mask, :].copy()

# Check if the samples were removed
print('IIH68490' in cd8.obs['sample'].values)  # Should print False
print('GSM3984200' in cd8.obs['sample'].values)  # Should print False

# Check the remaining unique samples
print(cd8.obs['sample'].unique())


# In[9]:


old = cd8[(cd8.obs['organ'] == 'CSF') & 
                    (cd8.obs['disease'] == 'HC') &
                    (cd8.obs['age_comparison'] == '≥60 years')].copy()

young = cd8[(cd8.obs['organ'] == 'CSF') & 
                    (cd8.obs['disease'] == 'IIH') &
                    (cd8.obs['age_comparison'] == '≤25 years')].copy()

ad = cd8[(cd8.obs['organ'] == 'CSF') & 
                    (cd8.obs['disease'] == 'AD')].copy()

old_young = old.concatenate(young)
adata_cd8_cam = old_young.concatenate(ad)


# In[10]:


# Check cell overlap
print(f"Processed cells: {len(cd8)}")
print(f"Raw count cells: {len(cd8.layers['raw_counts'])}")


# In[11]:


# Count cells per sample
print("Cells per sample:")
print(adata_cd8_cam.obs['sample'].value_counts().sort_index())

# Or get it as a sorted table
sample_counts = adata_cd8_cam.obs['sample'].value_counts().sort_index()
print(f"\nTotal samples: {len(sample_counts)}")
print(f"Cells per sample range: {sample_counts.min()} - {sample_counts.max()}")
print(f"Mean cells per sample: {sample_counts.mean():.1f}")
print(f"Disease types present: {adata_cd8_cam.obs['disease'].unique()}")


# In[12]:


# Check current counts
print("Before renaming:")
print(cd8.obs['disease'].value_counts())

# See how many IIH samples you have
iih_count = (cd8.obs['disease'] == 'IIH').sum()
print(f"\nNumber of IIH samples to rename: {iih_count}")


# In[13]:


# Check current counts
print("Before renaming:")
print(adata_cd8_cam.obs['age_comparison'].value_counts())


# In[14]:


# If you want to replace ALL 'IIH' with 'Young_Control' and ALL 'HC' with 'Old_Control'
adata_cd8_cam.obs['disease'] = adata_cd8_cam.obs['disease'].cat.rename_categories({
    'IIH': 'Young_Control',
    'HC': 'Old_Control'
})


# In[15]:


# Check the updated counts
print("After renaming:")
print(adata_cd8_cam.obs['disease'].value_counts())

# Verify IIH is gone
print(f"\nIIH samples remaining: {(adata_cd8_cam.obs['disease'] == 'Young_Control').sum()}")
print(f"\nHC samples remaining: {(adata_cd8_cam.obs['disease'] == 'Old_Control').sum()}")


# In[16]:


translator = sp.translator(hub = hubs.hubs[('blood')], data = adata_cd8_cam)


# In[17]:


#score_hub returns a list and we can save this directly to the adata.obs dataframe
adata_cd8_cam.obs['sen_score'] = sp.score_all_cells(adata_cd8_cam, hubs.hubs[('blood')], 
                                     identifiers = ['sex', 'disease_group', 'age_comparison'])


# In[18]:


adata_cd8_cam.obs.head()


# In[19]:


young = adata_cd8_cam[adata_cd8_cam.obs['disease'] == 'Young_Control'].copy()


# In[20]:


e = young.obs.sen_score.mean() #distribution mean


# In[21]:


std = young.obs.sen_score.std() #distribution std


# In[22]:


thresh = e + 3*std
thresh


# In[23]:


#function to add senescent label
def is_putative_sen(x):
    if x >= thresh:
        return 1
    else:
        return 0


# In[24]:


#map function to a new row in adata.obs
adata_cd8_cam.obs['putative_sen'] = adata_cd8_cam.obs.sen_score.map(is_putative_sen)


# In[25]:


adata_cd8_cam.obs.head()


# In[27]:


# Get unique samples per sex
sample_sex = adata_cd8_cam.obs[['disease', 'sex']]()
print(sample_sex['sex'].value_counts())


# In[28]:


# See which samples are which sex
sample_sex_df = adata_cd8_cam.obs[['sample', 'sex']].drop_duplicates().sort_values('sample')
print(sample_sex_df)

# Count samples by sex
print("\nSample counts by sex:")
print(sample_sex_df['sex'].value_counts())

# Get lists of samples
male_samples = sample_sex_df[sample_sex_df['sex'] == 'male']['sample'].tolist()
female_samples = sample_sex_df[sample_sex_df['sex'] == 'female']['sample'].tolist()

print(f"\nMale samples ({len(male_samples)}): {male_samples}")
print(f"Female samples ({len(female_samples)}): {female_samples}")


# In[34]:


# Get unique samples with their sex and disease status
sample_info = adata_cd8_cam.obs[['sample', 'sex', 'disease']].drop_duplicates()

# Create crosstab of samples by sex and disease
sample_crosstab = pd.crosstab(sample_info['sex'], sample_info['disease'])
print("Number of samples by sex and disease:")
print(sample_crosstab)


# In[ ]:





# In[ ]:





# In[26]:


# Filter to only include the three groups of interest
comparison_data = adata_cd8_cam.obs[adata_cd8_cam.obs['disease'] != 'exclude']

# Calculate senescence percentages grouped by BOTH cell type AND disease group
senescence_by_group_cell = comparison_data.groupby(['cell_type', 'disease'])['putative_sen'].agg(['mean', 'count', 'std']).reset_index()
senescence_by_group_cell['percent_senescent'] = senescence_by_group_cell['mean'] * 100
senescence_by_group_cell['sem'] = (senescence_by_group_cell['std'] / np.sqrt(senescence_by_group_cell['count'])) * 100

# Print summary statistics
print("cd8 senescence by cell type and disease group:")
print(senescence_by_group_cell)

# Calculate 95% confidence intervals
senescence_by_group_cell['ci_95'] = senescence_by_group_cell.apply(
    lambda row: stats.t.ppf(0.975, row['count']-1) * row['sem'] if row['count'] > 1 else 0, 
    axis=1
)

# CALCULATE SAMPLE-LEVEL SENESCENCE for individual points
# Filter for CD8 T cells and calculate per-sample senescence
cd8_obs = comparison_data[comparison_data['cell_type'] == 'CD8 T cells'].copy()
sample_senescence = cd8_obs.groupby(['sample', 'disease'])['putative_sen'].agg(['mean', 'count']).reset_index()
sample_senescence['percent_senescent_sample'] = sample_senescence['mean'] * 100

print("\nSample-level senescence data:")
print(sample_senescence)

# Main plotting code
plt.figure(figsize=(8, 6))

# Filter for just CD8 T cells if needed
cd8_data = senescence_by_group_cell[senescence_by_group_cell['cell_type'] == 'CD8 T cells'].copy()

# Sort by disease order
disease_order = ['Young_Control', 'Old_Control', 'AD']  # Adjust as needed
cd8_data['disease'] = pd.Categorical(cd8_data['disease'], 
                                      categories=disease_order, 
                                      ordered=True)
cd8_data = cd8_data.sort_values('disease')

# Create simple bar plot
colors = {'Young_Control': '#decbe4', 
          'Old_Control': '#ccebc5',  
          'AD': '#fbb4ae'}

bars = plt.bar(range(len(cd8_data)), 
               cd8_data['percent_senescent'],
               color=[colors[d] for d in cd8_data['disease']],
               edgecolor='black',
               linewidth=1.5)

# Add error bars
plt.errorbar(range(len(cd8_data)), 
             cd8_data['percent_senescent'],
             yerr=cd8_data['ci_95'],
             fmt='none',
             color='black',
             capsize=5,
             capthick=1.5,
             linewidth=1.5)

# Add value labels
for i, (idx, row) in enumerate(cd8_data.iterrows()):
    # Percentage on top
    plt.text(i, row['percent_senescent'] + row['ci_95'] + 0.2,
             f"{row['percent_senescent']:.1f}%",
             ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Sample size inside bar
    plt.text(i, row['percent_senescent'] + row['ci_95'] + 1,
             f"n={int(row['count'])}",
             ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

# Customize
plt.title('CD8 T Cell Senescence Across Disease Groups', fontsize=14, fontweight='bold')
plt.ylim(0,15) #cd8_data['percent_senescent'].max() * 1.4)
plt.ylabel('Senescent Cells (%)', fontsize=13)
plt.yticks(fontsize=10, fontweight='bold')

# Get current axes (this was missing in your original code)
ax = plt.gca()
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.tick_params(axis='x', width=2, length=6)
ax.tick_params(axis='y', width=1.5, length=6)

plt.xticks(range(len(cd8_data)), 
           ['Young\nControl', 'Old\nControl', 'AD'],
           fontsize=11)

plt.tight_layout()
plt.savefig('senepy_cd8_young_old_ad_percentSen.png', dpi=300, bbox_inches='tight')
plt.savefig('senepy_cd8_young_old_ad_percentSen.pdf', dpi=300, bbox_inches='tight')
plt.show()


# In[27]:


# Filter to only include the three groups of interest
comparison_data = adata_cd8_cam.obs[adata_cd8_cam.obs['disease'] != 'exclude']

# FIRST: Calculate sample-level senescence for CD8 T cells
cd8_obs = comparison_data[comparison_data['cell_type'] == 'CD8 T cells'].copy()
sample_senescence = cd8_obs.groupby(['sample', 'disease'])['putative_sen'].agg(['mean', 'count']).reset_index()
sample_senescence['percent_senescent_sample'] = sample_senescence['mean'] * 100

print("\nSample-level senescence data:")
print(sample_senescence)

# SECOND: Calculate the mean of sample means for the bars
senescence_by_group = sample_senescence.groupby('disease').agg({
    'percent_senescent_sample': ['mean', 'std', 'count']
}).reset_index()

# Flatten column names
senescence_by_group.columns = ['disease', 'percent_senescent', 'std', 'n_samples']

# Calculate SEM and 95% CI based on number of samples
senescence_by_group['sem'] = senescence_by_group['std'] / np.sqrt(senescence_by_group['n_samples'])
senescence_by_group['ci_95'] = senescence_by_group.apply(
    lambda row: stats.t.ppf(0.975, row['n_samples']-1) * row['sem'] if row['n_samples'] > 1 else 0, 
    axis=1
)

print("\nMean of sample means by disease group:")
print(senescence_by_group)

# Main plotting code
plt.figure(figsize=(8, 6))

# Sort by disease order
disease_order = ['Young_Control', 'Old_Control', 'AD']  # Adjust as needed
senescence_by_group['disease'] = pd.Categorical(senescence_by_group['disease'], 
                                                categories=disease_order, 
                                                ordered=True)
senescence_by_group = senescence_by_group.sort_values('disease')

# Create simple bar plot
colors = {'Young_Control': '#decbe4', 
          'Old_Control': '#ccebc5',  
          'AD': '#fbb4ae'}

bars = plt.bar(range(len(senescence_by_group)), 
               senescence_by_group['percent_senescent'],
               color=[colors[d] for d in senescence_by_group['disease']],
               edgecolor='black',
               linewidth=1.5)

# Add error bars
plt.errorbar(range(len(senescence_by_group)), 
             senescence_by_group['percent_senescent'],
             yerr=senescence_by_group['ci_95'],
             fmt='none',
             color='black',
             capsize=5,
             capthick=1.5,
             linewidth=1.5)

# ADD INDIVIDUAL SAMPLE POINTS
for i, disease in enumerate(disease_order):
    # Get samples for this disease group
    disease_samples = sample_senescence[sample_senescence['disease'] == disease]
    
    if len(disease_samples) > 0:
        # Add some jitter to x-position to avoid overlapping points
        n_samples = len(disease_samples)
        jitter_range = 0.25  # Adjust this to control spread of points
        
        if n_samples == 1:
            x_positions = [i]
        else:
            x_positions = np.linspace(i - jitter_range/2, i + jitter_range/2, n_samples)
        
        # Plot individual points
        plt.scatter(x_positions, 
                   disease_samples['percent_senescent_sample'],
                   color='black',
                   alpha=0.8,
                   s=50,  # Point size
                   zorder=10,  # Ensure points are on top
                   edgecolors='white',
                   linewidth=0.5)

# Add value labels
for i, (idx, row) in enumerate(senescence_by_group.iterrows()):
    # Percentage on top
    plt.text(i, row['percent_senescent'] + row['ci_95'] + 0.1,
             f"{row['percent_senescent']:.1f}%",
             ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Sample size inside or below bar
    #plt.text(i, row['percent_senescent'] / 2,
    #         f"n={int(row['n_samples'])}",
    #         ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')
    # Sample size inside bar
    plt.text(i, row['percent_senescent'] + row['ci_95'] + 2,
             f"n={int(row['n_samples'])}",
             ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

# Customize
plt.title('CD8 T Cell Senescence', fontsize=14, fontweight='bold')
plt.ylim(0, 42)
plt.ylabel('Senescent Cells Per Sample (%)', fontsize=13)
plt.yticks(fontsize=10, fontweight='bold')

# Get current axes
ax = plt.gca()
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.tick_params(axis='x', width=2, length=6)
ax.tick_params(axis='y', width=1.5, length=6)

plt.xticks(range(len(senescence_by_group)), 
           ['Young\nControl', 'Old\nControl', 'AD'],
           fontsize=11)

plt.tight_layout()
plt.savefig('senepy_cd8_young_old_ad_percentSen_perSample.png', dpi=300, bbox_inches='tight')
plt.savefig('senepy_cd8_young_old_ad_percentSen_perSample.pdf', dpi=300, bbox_inches='tight')
plt.show()


# In[28]:


adata_cd8_cam.write_h5ad('cd8_yoa_scored_8-12-25.h5ad')


# In[29]:


old = adata_cd8_cam[(adata_cd8_cam.obs['disease'] == 'Old_Control')].copy()

young = adata_cd8_cam[(adata_cd8_cam.obs['disease'] == 'Young_Control')].copy()

ad = adata_cd8_cam[(adata_cd8_cam.obs['disease'] == 'AD')].copy()

adata_cd8_yo = young.concatenate(old)
adata_cd8_oa = old.concatenate(ad)


# In[30]:


adata_cd8_yo.write_h5ad('cd8_yo_scored_8-12-25.h5ad')
adata_cd8_oa.write_h5ad('cd8_oa_scored_8-12-25.h5ad')


# In[ ]:





# In[31]:


old_sen = old[(old.obs['putative_sen'] == 1)].copy()

ad_sen = ad[(ad.obs['putative_sen'] == 1)].copy()

adata_cd8_oa_sen = old_sen.concatenate(ad_sen)


# In[32]:


old_nonsen = old[(old.obs['putative_sen'] == 0)].copy()

ad_nonsen = ad[(ad.obs['putative_sen'] == 0)].copy()

adata_cd8_oa_nonsen = old_nonsen.concatenate(ad_nonsen)


# In[33]:


adata_cd8_oa_sen.write_h5ad('cd8_oa_sen_scored_8-13-25.h5ad')
adata_cd8_oa_nonsen.write_h5ad('cd8_oa_nonsen_scored_8-13-25.h5ad')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[76]:


print("=== Range (Min/Max) of % Senescence by Disease Group ===")
range_stats = sample_senescence.groupby('disease')['percent_senescent_sample'].agg(['min', 'max', 'count']).round(2)
range_stats['range'] = range_stats['max'] - range_stats['min']
print(range_stats)


# In[78]:


print("\n=== Detailed Summary Statistics by Disease Group ===")
detailed_stats = sample_senescence.groupby('disease')['percent_senescent_sample'].describe().round(2)
print(detailed_stats)


# In[73]:


# Method 3: Individual sample breakdown (if you want to see each sample)
print("\n=== Individual Sample Values by Disease Group ===")
for disease in sample_senescence['disease'].unique():
    disease_data = sample_senescence[sample_senescence['disease'] == disease]
    print(f"\n{disease}:")
    print(f"  Samples: {disease_data['sample'].tolist()}")
    print(f"  % Senescence: {disease_data['percent_senescent_sample'].round(2).tolist()}")
    print(f"  Range: {disease_data['percent_senescent_sample'].min():.2f} - {disease_data['percent_senescent_sample'].max():.2f}")


# In[79]:


# Your current calculation:
# 1. Group by cell_type and disease, then calculate mean of 'putative_sen'
senescence_by_group_cell['percent_senescent'] = senescence_by_group_cell['mean'] * 100

# 2. For sample-level (what we added):
sample_senescence['percent_senescent_sample'] = sample_senescence['mean'] * 100

# This suggests 'putative_sen' is likely a binary column (0/1 or True/False)
# When you take the mean of binary values, you get the proportion of 1s/True values
# Multiplying by 100 converts proportion to percentage

print("=== Let's examine what's in the 'putative_sen' column ===")

# Check the unique values in putative_sen
print("Unique values in 'putative_sen':")
print(comparison_data['putative_sen'].unique())

# Check data type
print(f"\nData type of 'putative_sen': {comparison_data['putative_sen'].dtype}")

# Check value counts
print(f"\nValue counts for 'putative_sen':")
print(comparison_data['putative_sen'].value_counts())

# Check some basic statistics
print(f"\nBasic statistics for 'putative_sen':")
print(comparison_data['putative_sen'].describe())

# Example calculation breakdown for one group
print(f"\n=== Example calculation breakdown ===")
# Let's take CD8 T cells from one disease group as an example
example_group = comparison_data[
    (comparison_data['cell_type'] == 'CD8 T cells') & 
    (comparison_data['disease'] == comparison_data['disease'].iloc[0])
]

if len(example_group) > 0:
    disease_name = example_group['disease'].iloc[0]
    total_cells = len(example_group)
    senescent_cells = example_group['putative_sen'].sum()
    mean_value = example_group['putative_sen'].mean()
    percent_calc = mean_value * 100
    
    print(f"Disease group: {disease_name}")
    print(f"Total CD8 T cells: {total_cells}")
    print(f"Senescent cells (sum of 1s): {senescent_cells}")
    print(f"Mean of putative_sen: {mean_value:.4f}")
    print(f"Percent senescent: {percent_calc:.2f}%")
    print(f"Alternative calculation: {(senescent_cells/total_cells)*100:.2f}%")

print(f"\n=== Sample-level calculation example ===")
# Show how sample-level calculation works
if 'sample' in comparison_data.columns:
    # Take first sample as example
    first_sample = comparison_data['sample'].iloc[0]
    sample_data = comparison_data[
        (comparison_data['sample'] == first_sample) & 
        (comparison_data['cell_type'] == 'CD8 T cells')
    ]
    
    if len(sample_data) > 0:
        total_cells_sample = len(sample_data)
        senescent_cells_sample = sample_data['putative_sen'].sum()
        mean_sample = sample_data['putative_sen'].mean()
        percent_sample = mean_sample * 100
        
        print(f"Sample: {first_sample}")
        print(f"Total CD8 T cells in this sample: {total_cells_sample}")
        print(f"Senescent cells in this sample: {senescent_cells_sample}")
        print(f"Mean putative_sen for this sample: {mean_sample:.4f}")
        print(f"Percent senescent for this sample: {percent_sample:.2f}%")


# In[80]:


# Create a clean dataframe for export
sample_export = sample_senescence[['sample', 'disease', 'percent_senescent_sample', 'count']].copy()

# Round the percentage to 2 decimal places for cleaner export
sample_export['percent_senescent_sample'] = sample_export['percent_senescent_sample'].round(2)

# Rename columns for clarity in the CSV
sample_export = sample_export.rename(columns={
    'sample': 'Sample_ID',
    'disease': 'Disease_Group', 
    'percent_senescent_sample': 'Percent_Senescent',
    'count': 'Total_Cells_Counted'
})

# Sort by disease group and then by sample for organized output
disease_order = ['Young_Control', 'Old_Control', 'MCI', 'AD']  # Adjust as needed
sample_export['Disease_Group'] = pd.Categorical(sample_export['Disease_Group'], 
                                               categories=disease_order, 
                                               ordered=True)
sample_export = sample_export.sort_values(['Disease_Group', 'Sample_ID'])

# Display the dataframe before export
print("=== Sample-Level Senescence Data for Export ===")
print(sample_export)

# Export to CSV
csv_filename = 'sample_level_senescence_cd8.csv'
sample_export.to_csv(csv_filename, index=False)
print(f"\nCSV file exported as: {csv_filename}")

# Optional: Also create a summary by disease group in the same CSV or separate file
print(f"\n=== Summary by Disease Group ===")
summary_by_disease = sample_export.groupby('Disease_Group').agg({
    'Percent_Senescent': ['count', 'mean', 'std', 'min', 'max'],
    'Total_Cells_Counted': 'sum'
}).round(2)

# Flatten column names for the summary
summary_by_disease.columns = ['_'.join(col).strip() for col in summary_by_disease.columns.values]
summary_by_disease = summary_by_disease.rename(columns={
    'Percent_Senescent_count': 'Number_of_Samples',
    'Percent_Senescent_mean': 'Mean_Percent_Senescent',
    'Percent_Senescent_std': 'Std_Percent_Senescent', 
    'Percent_Senescent_min': 'Min_Percent_Senescent',
    'Percent_Senescent_max': 'Max_Percent_Senescent',
    'Total_Cells_Counted_sum': 'Total_Cells_All_Samples'
})

print(summary_by_disease)

# Export summary to separate CSV
summary_filename = 'senescence_summary_by_disease_cd8.csv'
summary_by_disease.to_csv(summary_filename)
print(f"\nSummary CSV file exported as: {summary_filename}")

# Optional: Create a combined file with both individual samples and summary
print(f"\n=== Creating Combined Export File ===")

# Create a combined dataframe with a separator row
combined_data = []

# Add individual sample data
for _, row in sample_export.iterrows():
    combined_data.append(row.to_dict())

# Add separator rows
combined_data.append({
    'Sample_ID': '',
    'Disease_Group': '=== SUMMARY BY DISEASE GROUP ===',
    'Percent_Senescent': '',
    'Total_Cells_Counted': ''
})

combined_data.append({
    'Sample_ID': 'Disease_Group',
    'Disease_Group': 'Number_of_Samples', 
    'Percent_Senescent': 'Mean_Percent_Senescent',
    'Total_Cells_Counted': 'Total_Cells_All_Samples'
})

# Add summary data
for disease, row in summary_by_disease.iterrows():
    combined_data.append({
        'Sample_ID': disease,
        'Disease_Group': int(row['Number_of_Samples']),
        'Percent_Senescent': row['Mean_Percent_Senescent'],
        'Total_Cells_Counted': int(row['Total_Cells_All_Samples'])
    })

combined_df = pd.DataFrame(combined_data)
combined_filename = 'sample_and_summary_senescence_cd8.csv'
combined_df.to_csv(combined_filename, index=False)
print(f"Combined CSV file exported as: {combined_filename}")

print(f"\n=== Files Created ===")
print(f"1. {csv_filename} - Individual sample data only")
print(f"2. {summary_filename} - Summary statistics by disease group") 
print(f"3. {combined_filename} - Both individual and summary data")


# In[ ]:


#adata_cd8_coy.write_h5ad('cd8_scored_7-29-25.h5ad')

