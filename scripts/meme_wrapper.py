"""
MEME Motif Discovery Wrapper
=============================

This script runs MEME on peak sequences for transcription factors.
It uses data/dataset.csv to filter sequences by TF binding status
and generates comprehensive visualizations and analysis.

Requires:
  - MEME Suite installed locally (meme, meme-chip)
  - Python packages: pandas, matplotlib, subprocess, re, os

Usage:
  python 4_meme_motif_discovery.py
"""

import pandas as pd
import numpy as np
import subprocess
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile


class MEMEWrapper:
    """Wrapper for running MEME motif discovery."""
    
    def __init__(self, dataset_path, results_dir='results', meme_dir='meme_results'):
        """
        Initialize MEME wrapper.
        
        Args:
            dataset_path (str): Path to data/dataset.csv
            results_dir (str): Directory containing peak FASTA files
            meme_dir (str): Directory to save MEME results
        """
        self.dataset_path = dataset_path
        self.results_dir = results_dir
        self.meme_dir = meme_dir
        self.meme_results = {}
        
        os.makedirs(meme_dir, exist_ok=True)
        
        # Check if MEME is installed
        self.check_meme_installation()
    
    def check_meme_installation(self):
        """Verify MEME Suite is installed."""
        try:
            result = subprocess.run(['meme', '-version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                print("✓ MEME Suite is installed")
                # Extract version
                version_line = result.stdout.split('\n')[0]
                print(f"  {version_line}\n")
            else:
                raise Exception("MEME check failed")
        except FileNotFoundError:
            print("ERROR: MEME not found in PATH")
            print("Please install MEME Suite:")
            print("  macOS: brew install meme-suite")
            print("  Linux/conda: conda install -c bioconda meme")
            exit(1)
        except Exception as e:
            print(f"ERROR checking MEME: {e}")
            exit(1)
    
    def load_dataset(self):
        """Load and display dataset information."""
        print("="*100)
        print("LOADING DATASET")
        print("="*100 + "\n")
        
        df = pd.read_csv(self.dataset_path)
        print(f"Dataset loaded: {len(df)} total bins")
        print(f"Columns: {list(df.columns)}\n")
        
        # Show TF distribution
        print("TF Binding Distribution:")
        for tf in ['CTCF', 'REST', 'EP300']:
            if tf in df.columns:
                bound = (df[tf] == 'B').sum()
                total = len(df)
                print(f"  {tf}: {bound}/{total} bins bound ({bound/total*100:.1f}%)")
        
        return df
    
    def extract_sequences_by_tf(self, df, tf_name, bound_status='B'):
        """
        Extract sequences from dataset where TF is bound or unbound.
        
        Args:
            df (pd.DataFrame): Dataset
            tf_name (str): TF name (CTCF, REST, EP300)
            bound_status (str): 'B' for bound, 'U' for unbound
        
        Returns:
            list: List of (header, sequence) tuples
        """
        # Filter to TF binding status
        filtered_df = df[df[tf_name] == bound_status].copy()
        
        print(f"\n{tf_name} - {bound_status}ound bins:")
        print(f"  Found {len(filtered_df)} sequences")
        
        # Create FASTA-style headers from the data
        sequences = []
        for idx, row in filtered_df.iterrows():
            header = f"{tf_name}_{idx}_chr{row['chrom']}_{row['start']}_{row['stop']}"
            seq = row['sequence']
            sequences.append((header, seq))
        
        return sequences
    
    def write_fasta(self, sequences, output_path):
        """
        Write sequences to FASTA file.
        
        Args:
            sequences (list): List of (header, sequence) tuples
            output_path (str): Output FASTA file path
        
        Returns:
            str: Path to written file
        """
        with open(output_path, 'w') as f:
            for header, seq in sequences:
                f.write(f">{header}\n{seq}\n")
        
        print(f"  ✓ Wrote {len(sequences)} sequences to {output_path}")
        return output_path
    
    def run_meme(self, fasta_path, tf_name, analysis_type, 
                 nmotifs=5, minw=6, maxw=20, mod='zoops', seed=42):
        """
        Run MEME on sequence file.
        
        Args:
            fasta_path (str): Path to input FASTA
            tf_name (str): TF name
            analysis_type (str): 'bound' or 'unbound'
            nmotifs (int): Number of motifs to find
            minw (int): Minimum motif width
            maxw (int): Maximum motif width
            mod (str): MEME model ('zoops', 'oops', 'tcm')
            seed (int): Random seed for reproducibility
        
        Returns:
            dict: Results dictionary with output paths
        """
        output_dir = os.path.join(self.meme_dir, f'{tf_name}_{analysis_type}')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nRunning MEME on {tf_name} ({analysis_type})...")
        print(f"  Output directory: {output_dir}")
        print(f"  Parameters: nmotifs={nmotifs}, minw={minw}, maxw={maxw}, mod={mod}")
        
        # Build MEME command
        cmd = [
            'meme',
            fasta_path,
            '-dna',
            f'-nmotifs', str(nmotifs),
            f'-minw', str(minw),
            f'-maxw', str(maxw),
            f'-mod', mod,
            f'-seed', str(seed),
            '-p', '4',  # Use 4 processors
            '-o', output_dir,
            '-nostatus'  # No progress output
        ]
        
        try:
            print("  Running MEME... (this may take several minutes)")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f"  ✓ MEME completed successfully")
                
                # Check for output files
                meme_out = os.path.join(output_dir, 'meme.txt')
                meme_html = os.path.join(output_dir, 'meme.html')
                
                results = {
                    'tf': tf_name,
                    'analysis_type': analysis_type,
                    'output_dir': output_dir,
                    'meme_txt': meme_out if os.path.exists(meme_out) else None,
                    'meme_html': meme_html if os.path.exists(meme_html) else None,
                    'success': True
                }
                
                return results
            else:
                print(f"  ✗ MEME failed with return code {result.returncode}")
                print(f"  Error: {result.stderr[:500]}")
                return {'success': False, 'error': result.stderr}
        
        except subprocess.TimeoutExpired:
            print(f"  ✗ MEME timed out (>1 hour)")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            print(f"  ✗ Error running MEME: {e}")
            return {'success': False, 'error': str(e)}
    
    def parse_meme_output(self, meme_txt_path):
        """
        Parse MEME text output to extract motif information.
        
        Args:
            meme_txt_path (str): Path to meme.txt file
        
        Returns:
            dict: Parsed motif information
        """
        if not os.path.exists(meme_txt_path):
            return None
        
        motifs = []
        
        with open(meme_txt_path, 'r') as f:
            content = f.read()
        
        # Extract MOTIF blocks
        motif_pattern = r'MOTIF\s+(\d+)\s+(\S+).*?E-value\s+.*?\n'
        matches = re.findall(motif_pattern, content, re.DOTALL)
        
        # Simple parsing - extract key stats
        if 'MOTIF' in content:
            # Count number of motifs found
            motif_count = len(re.findall(r'^MOTIF\s+\d+', content, re.MULTILINE))
            motifs.append({'count': motif_count})
        
        return motifs if motifs else None
    
    def create_summary_plots(self, all_results):
        """
        Create summary plots comparing MEME results.
        
        Args:
            all_results (dict): Results from all MEME runs
        """
        print("\n" + "="*100)
        print("CREATING SUMMARY VISUALIZATIONS")
        print("="*100 + "\n")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('MEME Motif Discovery Summary', fontsize=16, fontweight='bold')
        
        # Parse results for plotting
        tf_names = ['CTCF', 'REST']
        analysis_types = ['bound', 'unbound']
        
        analysis_status = []
        for tf in tf_names:
            for atype in analysis_types:
                key = f"{tf}_{atype}"
                if key in all_results and all_results[key].get('success'):
                    analysis_status.append((key, 'Success'))
                else:
                    analysis_status.append((key, 'Failed'))
        
        # Plot 1: Analysis completion status
        ax1 = axes[0, 0]
        labels = [a[0] for a in analysis_status]
        colors = ['#27ae60' if a[1] == 'Success' else '#e74c3c' for a in analysis_status]
        y_pos = np.arange(len(labels))
        ax1.barh(y_pos, [1]*len(labels), color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels)
        ax1.set_xlim(0, 1.2)
        ax1.set_xticks([])
        ax1.set_title('MEME Analysis Status')
        for i, (label, status) in enumerate(analysis_status):
            ax1.text(0.5, i, status, ha='center', va='center', fontweight='bold', color='white')
        
        # Plot 2: CTCF bound vs unbound
        ax2 = axes[0, 1]
        ctcf_bound = all_results.get('CTCF_bound', {}).get('success', False)
        ctcf_unbound = all_results.get('CTCF_unbound', {}).get('success', False)
        ax2.bar(['Bound', 'Unbound'], [ctcf_bound, ctcf_unbound], 
                color=['#3498db', '#95a5a6'], alpha=0.7)
        ax2.set_ylabel('Completed')
        ax2.set_title('CTCF Analysis Status')
        ax2.set_ylim(0, 1.2)
        
        # Plot 3: REST bound vs unbound
        ax3 = axes[1, 0]
        rest_bound = all_results.get('REST_bound', {}).get('success', False)
        rest_unbound = all_results.get('REST_unbound', {}).get('success', False)
        ax3.bar(['Bound', 'Unbound'], [rest_bound, rest_unbound], 
                color=['#e67e22', '#95a5a6'], alpha=0.7)
        ax3.set_ylabel('Completed')
        ax3.set_title('REST Analysis Status')
        ax3.set_ylim(0, 1.2)
        
        # Plot 4: Summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "MEME Discovery Summary\n"
        summary_text += "="*40 + "\n\n"
        
        for tf in tf_names:
            summary_text += f"{tf}:\n"
            for atype in analysis_types:
                key = f"{tf}_{atype}"
                if key in all_results:
                    status = "✓ Complete" if all_results[key].get('success') else "✗ Failed"
                    summary_text += f"  {atype.capitalize()}: {status}\n"
            summary_text += "\n"
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        png_path = os.path.join(self.meme_dir, 'meme_discovery_summary.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved summary plot to {png_path}\n")
    
    def create_summary_report(self, all_results, df):
        """
        Create comprehensive text report of MEME results.
        
        Args:
            all_results (dict): Results from all analyses
            df (pd.DataFrame): Original dataset
        """
        report_path = os.path.join(self.meme_dir, 'meme_summary_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("MEME MOTIF DISCOVERY - COMPREHENSIVE REPORT\n")
            f.write("="*100 + "\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-"*100 + "\n")
            f.write(f"Total bins analyzed: {len(df)}\n\n")
            
            for tf in ['CTCF', 'REST']:
                if tf in df.columns:
                    bound = (df[tf] == 'B').sum()
                    unbound = (df[tf] == 'U').sum()
                    f.write(f"{tf}:\n")
                    f.write(f"  Bound bins: {bound}\n")
                    f.write(f"  Unbound bins: {unbound}\n")
                    f.write(f"  Total: {bound + unbound}\n\n")
            
            # MEME Results
            f.write("\n" + "="*100 + "\n")
            f.write("MEME ANALYSIS RESULTS\n")
            f.write("="*100 + "\n\n")
            
            for tf in ['CTCF', 'REST']:
                f.write(f"\n{tf} TRANSCRIPTION FACTOR\n")
                f.write("-"*100 + "\n")
                
                for atype in ['bound', 'unbound']:
                    key = f"{tf}_{atype}"
                    f.write(f"\n{atype.upper()} SEQUENCES:\n")
                    
                    if key in all_results:
                        result = all_results[key]
                        if result.get('success'):
                            f.write(f"  Status: SUCCESS\n")
                            f.write(f"  Output directory: {result['output_dir']}\n")
                            f.write(f"  MEME output: {result['meme_txt']}\n")
                            f.write(f"  HTML report: {result['meme_html']}\n")
                        else:
                            f.write(f"  Status: FAILED\n")
                            f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
                    else:
                        f.write(f"  Status: NOT RUN\n")
            
            # Next steps
            f.write("\n" + "="*100 + "\n")
            f.write("NEXT STEPS\n")
            f.write("="*100 + "\n\n")
            f.write("1. View HTML reports:\n")
            f.write("   - Open HTML files in results/ directory to view interactive MEME results\n\n")
            f.write("2. Parse motif results:\n")
            f.write("   - Use meme.txt files for programmatic analysis\n")
            f.write("   - Extract and visualize discovered motifs\n\n")
            f.write("3. Compare to JASPAR:\n")
            f.write("   - Match discovered motifs against JASPAR database\n")
            f.write("   - Identify known vs novel binding patterns\n\n")
            f.write("4. Validate findings:\n")
            f.write("   - Check enrichment of discovered motifs in bound vs unbound sequences\n")
            f.write("   - Look for co-occurring motifs\n")
        
        print(f"✓ Saved summary report to {report_path}\n")
    
    def run_all_analyses(self, sample_size=None):
        """
        Run complete MEME analysis on all TFs.
        
        Args:
            sample_size (int): If set, subsample to this many sequences per category
        """
        print("\n" + "="*100)
        print("MEME MOTIF DISCOVERY WRAPPER")
        print("="*100 + "\n")
        
        # Load dataset
        df = self.load_dataset()
        
        all_results = {}
        
        # Run MEME on each TF and binding status
        for tf_name in ['CTCF', 'REST']:
            print(f"\n{'='*100}")
            print(f"PROCESSING {tf_name}")
            print(f"{'='*100}\n")
            
            for bound_status, analysis_type in [('B', 'bound'), ('U', 'unbound')]:
                print(f"\n{tf_name} - {analysis_type.upper()} sequences")
                print("-"*100)
                
                # Extract sequences
                sequences = self.extract_sequences_by_tf(df, tf_name, bound_status)
                
                if len(sequences) == 0:
                    print(f"  WARNING: No sequences found for {tf_name} {analysis_type}")
                    all_results[f"{tf_name}_{analysis_type}"] = {
                        'success': False, 
                        'error': 'No sequences found'
                    }
                    continue
                
                # Subsample if requested
                if sample_size and len(sequences) > sample_size:
                    sequences = sequences[:sample_size]
                    print(f"  (Subsampled to {sample_size} sequences)")
                
                # Write FASTA
                fasta_path = os.path.join(
                    self.meme_dir, 
                    f'{tf_name}_{analysis_type}_sequences.fa'
                )
                self.write_fasta(sequences, fasta_path)
                
                # Run MEME
                result = self.run_meme(fasta_path, tf_name, analysis_type)
                all_results[f"{tf_name}_{analysis_type}"] = result
        
        # Create visualizations
        self.create_summary_plots(all_results)
        
        # Create summary report
        self.create_summary_report(all_results, df)
        
        # Print final summary
        print("\n" + "="*100)
        print("MEME ANALYSIS COMPLETE")
        print("="*100 + "\n")
        
        print("Results saved in: results/meme_results/\n")
        print("Files generated:")
        for tf in ['CTCF', 'REST']:
            for atype in ['bound', 'unbound']:
                key = f"{tf}_{atype}"
                if all_results[key].get('success'):
                    print(f"  ✓ {tf} ({atype}): {all_results[key]['output_dir']}")
                else:
                    print(f"  ✗ {tf} ({atype}): FAILED")
        
        print(f"\n  ✓ Summary plot: {os.path.join(self.meme_dir, 'meme_discovery_summary.png')}")
        print(f"  ✓ Summary report: {os.path.join(self.meme_dir, 'meme_summary_report.txt')}")


def main():
    """Main execution."""
    # Configuration
    dataset_path = 'data/dataset.csv'
    results_dir = 'results'
    meme_dir = 'results/meme_results'
    
    # Check paths exist
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please run data pipeline first: python merged_data_pipeline.py")
        exit(1)
    
    # Initialize wrapper
    wrapper = MEMEWrapper(dataset_path, results_dir, meme_dir)
    
    # Run analysis
    # Uncomment sample_size line below to subsample for testing (e.g., 5000 sequences per category)
    # wrapper.run_all_analyses(sample_size=5000)
    
    wrapper.run_all_analyses()  # Run on full dataset


if __name__ == "__main__":
    main()