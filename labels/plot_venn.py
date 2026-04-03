"""
Script 1: Venn Diagram Analysis of TF Binding Patterns (IMPROVED)
=================================================================

This script analyzes the co-occurrence of TF binding (CTCF, REST, EP300) and ATAC accessibility.
Creates a single figure with two Venn diagram subplots (one for ATAC=U, one for ATAC=B) showing 
the overlap between the three TFs with professional formatting and visible counts.

Improvements:
  - Dual subplot layout for easy comparison
  - Enhanced color scheme with better contrast
  - Visible, formatted numbers on all regions
  - CSV export of all binding pattern counts for reproducible plotting
  - Professional formatting with consistent styling
  - Better legend and annotation handling

Output: 
  - venn_diagrams_combined.png (publication-quality figure)
  - venn_diagram_data.csv (tabular data for reproducible analysis)
  - Printed summary with counts for all 2^3 = 8 combinations per ATAC state
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import numpy as np
from pathlib import Path


def categorize_binding_patterns(csv_path):
    """
    Read the dataset and categorize bins by their binding pattern.
    
    Returns:
        dict: Nested structure {atac_state: {tf_combination: count}}
        dict: Nested structure {atac_state: {tf_combination: sequence_list}}
    """
    
    print("Reading dataset...")
    df = pd.read_csv(csv_path)
    
    # Initialize counters and storage
    pattern_counts = {
        'U': {},  # ATAC = U (closed chromatin)
        'B': {}   # ATAC = B (open chromatin)
    }
    
    pattern_sequences = {
        'U': {},
        'B': {}
    }
    
    count = 0
    # Iterate through dataset
    for idx, row in df.iterrows():
        count += 1
        if count % 10000 == 0:
            print(f"  Processed {count} rows...")
        atac = row['ATAC']
        ctcf = row['CTCF']
        rest = row['REST']
        ep300 = row['EP300']
        seq = row['sequence']
        
        # Create a tuple representing the TF binding pattern
        tf_pattern = (ctcf, rest, ep300)
        
        # Initialize if not seen before
        if tf_pattern not in pattern_counts[atac]:
            pattern_counts[atac][tf_pattern] = 0
            pattern_sequences[atac][tf_pattern] = []
        
        # Increment count and store sequence
        pattern_counts[atac][tf_pattern] += 1
        pattern_sequences[atac][tf_pattern].append(seq)
    
    print(f"✓ Processed {count} total rows\n")
    
    return pattern_counts, pattern_sequences


def extract_venn_data(pattern_counts, atac_state):
    """
    Extract counts for all 8 regions of the Venn diagram.
    
    Returns both the tuple for venn3() and a detailed dict for CSV export.
    """
    counts = pattern_counts[atac_state]
    
    # All unbound (no TF binding)
    none_bound = counts.get(('U', 'U', 'U'), 0)
    
    # Single TF binding
    ctcf_only = counts.get(('B', 'U', 'U'), 0)
    rest_only = counts.get(('U', 'B', 'U'), 0)
    ep300_only = counts.get(('U', 'U', 'B'), 0)
    
    # Pairwise binding
    ctcf_rest = counts.get(('B', 'B', 'U'), 0)
    ctcf_ep300 = counts.get(('B', 'U', 'B'), 0)
    rest_ep300 = counts.get(('U', 'B', 'B'), 0)
    
    # All three TFs bound
    all_three = counts.get(('B', 'B', 'B'), 0)
    
    # Tuple for venn3: (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    # Where A=CTCF, B=REST, C=EP300
    venn_tuple = (
        ctcf_only,      # Only CTCF
        rest_only,      # Only REST
        ctcf_rest,      # CTCF and REST, not EP300
        ep300_only,     # Only EP300
        ctcf_ep300,     # CTCF and EP300, not REST
        rest_ep300,     # REST and EP300, not CTCF
        all_three       # All three
    )
    
    detailed_dict = {
        'CTCF_only': ctcf_only,
        'REST_only': rest_only,
        'EP300_only': ep300_only,
        'CTCF_REST_only': ctcf_rest,
        'CTCF_EP300_only': ctcf_ep300,
        'REST_EP300_only': rest_ep300,
        'All_three': all_three,
        'None_bound': none_bound
    }
    
    return venn_tuple, detailed_dict


def create_combined_venn_figure(pattern_counts):
    """
    Create a single figure with two Venn diagrams as subplots.
    One for ATAC=U, one for ATAC=B.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Transcription Factor Binding Overlap by ATAC-Seq State', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    atac_states = ['U', 'B']
    atac_labels = ['Closed Chromatin (ATAC = U)', 'Open Chromatin (ATAC = B)']
    
    # Color schemes for each subplot
    colors_schemes = [
        ['#D55E00', '#0072B2', '#000000'],  # vermillion → blue → black
        ['#E69F00', '#009E73', '#56B4E9']   # orange → green → sky blue
    ]
    
    venn_data_all = {}
    
    for idx, (atac_state, atac_label, color_scheme) in enumerate(zip(atac_states, atac_labels, colors_schemes)):
        ax = axes[idx]
        
        counts = pattern_counts[atac_state]
        total_bins = sum(counts.values())
        
        # Extract venn data
        venn_tuple, detailed_dict = extract_venn_data(pattern_counts, atac_state)
        venn_data_all[atac_state] = detailed_dict
        
        # Create venn diagram
        v = venn3(subsets=venn_tuple, set_labels=('CTCF', 'REST', 'EP300'), ax=ax)
        
        # Customize colors
        if v.patches:
            for i, patch in enumerate(v.patches):
                if patch is not None:
                    patch.set_facecolor(color_scheme[i % len(color_scheme)])
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(2)
        
        # Customize text labels (set names)
        if v.set_labels:
            for label in v.set_labels:
                label.set_fontsize(12)
                label.set_fontweight('bold')
        
        # Make count text larger and bold
        if v.subset_labels:
            for text in v.subset_labels:
                if text is not None:
                    text.set_fontsize(11)
                    text.set_fontweight('bold')
                    text.set_color('black')
        
        # Title
        ax.set_title(f'{atac_label}\nTotal bins: {total_bins:,}', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig, venn_data_all


def create_data_csv(venn_data_all, output_dir):
    """Export all venn diagram data to CSV for reproducible plotting."""
    
    output_dir = Path(output_dir)
    csv_path = output_dir / "venn_diagram_data.csv"
    
    # Create a comprehensive dataframe
    rows = []
    for atac_state in ['U', 'B']:
        atac_label = "Closed (U)" if atac_state == 'U' else "Open (B)"
        data = venn_data_all[atac_state]
        
        for region_name, count in data.items():
            rows.append({
                'ATAC_State': atac_state,
                'ATAC_Label': atac_label,
                'Region': region_name,
                'Count': count
            })
    
    df_export = pd.DataFrame(rows)
    df_export.to_csv(csv_path, index=False)
    print(f"✓ Saved venn diagram data to {csv_path}")
    
    return df_export


def print_detailed_report(pattern_counts):
    """Print a detailed report of all binding patterns."""
    
    print("\n" + "="*90)
    print("DETAILED BINDING PATTERN ANALYSIS")
    print("="*90 + "\n")
    
    for atac_state in ['U', 'B']:
        atac_label = "CLOSED (U)" if atac_state == 'U' else "OPEN (B)"
        print(f"\n{'-'*90}")
        print(f"ATAC-Seq State: {atac_label}")
        print(f"{'-'*90}")
        
        counts = pattern_counts[atac_state]
        total = sum(counts.values())
        
        # Create sorted list for display
        patterns_display = []
        for pattern, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            ctcf, rest, ep300 = pattern
            pct = (count / total) * 100 if total > 0 else 0
            patterns_display.append({
                'pattern': pattern,
                'count': count,
                'percentage': pct
            })
        
        # Print table with better formatting
        print(f"\n{'CTCF':^6} | {'REST':^6} | {'EP300':^6} | {'Count':>10} | {'Percentage':>10}")
        print("-" * 70)
        
        for p in patterns_display:
            ctcf, rest, ep300 = p['pattern']
            print(f"{ctcf:^6} | {rest:^6} | {ep300:^6} | {p['count']:>10,} | {p['percentage']:>9.2f}%")
        
        print(f"{'-'*70}")
        print(f"{'TOTAL':>20} | {total:>10,} | {100.0:>9.2f}%")


def print_summary_statistics(pattern_counts):
    """Print summary statistics about TF binding patterns."""
    
    print("\n" + "="*90)
    print("SUMMARY STATISTICS")
    print("="*90 + "\n")
    
    for atac_state, label in [('U', 'ATAC = U (Closed Chromatin)'), ('B', 'ATAC = B (Open Chromatin)')]:
        counts = pattern_counts[atac_state]
        total = sum(counts.values())
        
        # Count TFs bound
        bound_at_least_one = sum(count for pattern, count in counts.items() 
                                 if any(tf == 'B' for tf in pattern))
        bound_at_least_two = sum(count for pattern, count in counts.items() 
                                 if sum(1 for tf in pattern if tf == 'B') >= 2)
        bound_all_three = counts.get(('B', 'B', 'B'), 0)
        
        # Per-TF stats
        ctcf_count = sum(count for pattern, count in counts.items() if pattern[0] == 'B')
        rest_count = sum(count for pattern, count in counts.items() if pattern[1] == 'B')
        ep300_count = sum(count for pattern, count in counts.items() if pattern[2] == 'B')
        
        print(f"\n{label}")
        print(f"  {'─' * 70}")
        print(f"  Total bins: {total:,}")
        print(f"  ├─ At least one TF bound:    {bound_at_least_one:>10,} ({100*bound_at_least_one/total:>6.2f}%)")
        print(f"  ├─ At least two TFs bound:   {bound_at_least_two:>10,} ({100*bound_at_least_two/total:>6.2f}%)")
        print(f"  └─ All three TFs bound:      {bound_all_three:>10,} ({100*bound_all_three/total:>6.2f}%)")
        print(f"\n  Individual TF binding:")
        print(f"  ├─ CTCF bound:               {ctcf_count:>10,} ({100*ctcf_count/total:>6.2f}%)")
        print(f"  ├─ REST bound:               {rest_count:>10,} ({100*rest_count/total:>6.2f}%)")
        print(f"  └─ EP300 bound:              {ep300_count:>10,} ({100*ep300_count/total:>6.2f}%)")


def main():
    # Configuration
    csv_path = "data/dataset.csv"
    output_dir = "labels"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*90)
    print("VENN DIAGRAM ANALYSIS: TF Binding Patterns (IMPROVED)")
    print("="*90 + "\n")
    
    # Load and categorize data
    pattern_counts, pattern_sequences = categorize_binding_patterns(csv_path)
    
    # Print detailed report
    print_detailed_report(pattern_counts)
    
    # Print summary statistics
    print_summary_statistics(pattern_counts)
    
    # Create combined figure
    print("\n" + "="*90)
    print("CREATING VENN DIAGRAMS")
    print("="*90 + "\n")
    
    print("Creating combined Venn diagram figure...")
    fig, venn_data_all = create_combined_venn_figure(pattern_counts)
    fig_path = Path(output_dir) / "venn_diagrams_combined.png"
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    
    # Export data to CSV
    print("\nExporting venn diagram data...")
    df_export = create_data_csv(venn_data_all, output_dir)
    print("\nVenn Diagram Data (CSV):")
    print(df_export.to_string(index=False))
    
    plt.close('all')
    
    print("\n" + "="*90)
    print("VENN DIAGRAM ANALYSIS COMPLETE")
    print("="*90)
    print(f"\nGenerated files:")
    print(f"  • {output_dir}/venn_diagrams_combined.png (publication-quality figure)")
    print(f"  • {output_dir}/venn_diagram_data.csv (tabular data for reproducible plotting)")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()