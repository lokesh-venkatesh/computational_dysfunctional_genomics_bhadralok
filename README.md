(29th January 2026)

### @Bhadralok - MAKE A NOTE OF THE FOLLOWING (pwetty pleease :3):
I have about 4-5 tasks for you in this repo, most of the other stuff I have written a skeleton as well as code for.
1. I want you to implement K-fold cross-validation in the model.py script. All you have to do is basically randomise the positive and negative data-subsets generated in the __main__ codeblock of model.py independently, and do this for all 'k'.
2. Similar to how I used the scikit-learn library to plot the ROC curve and calculate AUC-ROC, I would like you to do the same for PRC and AUC-PRC. Very easily doable with some googling + copy-pasting.
3. Next, fill out the run.py script. The description of what should go in there is already written as a comment at the top.
4. Finally, I want you to run through the entire repo and proof-read everything. There is stuff that I may have missed, or minor details (for example, the labels on the plots being saved, since I haven't modified those for the K-fold cross validation stuff in my version). I'd also like you to implement the entire pipeline starting from the first script and go all the way to the last one and see if everything is working alright on your machine.
5. In particular, I want you to look at the nth_order_markov_matrix function in model.py, and see if it is to your liking, or if there is any way of making it better. I'm pretty sure I've made a very lazy implementation and overlooked some things to speed it up. As of now it takes O(minutes) for the training dataset of 19 chromosomes.

I've put as many details about what I've tried to implement below. Have a look as you proof-read the code and complete the above tasks.

Also I've turned the repo to private visibility for the time-being. We can turn it public later on whenever it feels appropriate.
I've also created a requirements.txt file, containing a list of libraries to pip install, although I think this is overkill since it's just us both for now.
Ignore the plots in the 'results' folder, it was just me trying to test-run and make sure whatever code I have written is running fine (I definitely want you to look at and possibly improve the main nth_order_markov_matrix function though, that is the code memory-and-time-consumer when running this entire thing)

---

# Description of the repo structure so far:

Here's what each of the scripts currently in the repo are supposed to do:

1. `data_all_download.py`: this script auto-downloads all of the **hg38** genome files from `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes`, and saves them into the newly-created folder `hg38` in the working directory. This consists only of the raw sequenced genome files.

We also manually download the dataset provided along with the project description on the Google Classroom. This folder, `projectData`, contains the `.tsv` files that contain the Chip-Seq information

2. `data_as_fasta.py`: this file links the data provided in the `.tsv` files with the downloaded hg38 genome files. The resulting 'bins' are then saved into fasta files for each chromosome, say for chromosome 1, as `data/chr1_seqs.fa`. Each 200-length bin that is identified and saved, is labelled with the format `chr{CHROMOSOME NUMBER}_{BIN START}_{BIN STOP}_{ATAC-SEQ BINDING}_{CTCF BINDING}_{REST BINDING}_{EP300 BINDING}`. 

3. `data_as_one_tsv.py`: this file takes all of the fasta files saved in `data/`, and compiles all of these sequences into one massive dataframe, and then saves it as `dataset.csv` locally. 

Note that on further processing, I found that the .tsv files themselves contain 9 bins that contain ambigous nucleotides, which appear as 'N's in the dataset. I have promptly identified and removed these 9 sequences, and ensured that all remaining sequences that were saved to the `.csv` file are of length 200. There were roughly `4.2 million` entries in the resulting dataframe, most of which were non-binding to all three transcription factors. 

Just to explicitly mention, Chromosomes 3, 10 and 17 do not have any labels for the last three columns, and have therefore been saved into the same file with `None` values in those three columns corresponding to the three transcription factors. 

4. `model.py`: This contains the entire script and functions for taking in data from the `dataset.csv` file and then for given specifications of 'm' (order of the markov model), 'k' (number of folds in the cross-validation), 'c' (chromosome number as dataset for model set-up, either any number from 1 to 22, except for 3/10/17, or 'All'), and TF (the name of the transcription factor we are building a markov classifier for, any one of CTCF/REST/EP300).

With these specifications, the model then loads the `.csv` file, extracts the subset of sequences for the corresponding TF as well as Chromosome, and then splits the dataset into positive and negative subsets. It then proceeds with constructing an 'm-th' order markov classifier. 

**As of 29th January, we are yet to implement K-fold cross validation**, which will need to implemented by modifying this same script, and by randomising and splitting the negative and positive datasets separately, so as to ensure uniform data ratio. **We also need to implement PRC (Precision-Recall Curve) plotting**, and averaging across the K-folds for calculation of *AUC-PRC* as well as *AUC-ROC*.

All other requirements for the assignment are taken care of already by the current version of `model.py`.

5. `run.py`: A revamp of the earlier `master_script.py` script that we had written, this script is supposed to run `data_as_one_tsv.py` first, followed by `model.py`. This script can be run in the CLI with arguments for `k`, `m`, `c`, and `TF`, and these will then be taken up by `model.py`. 

We are not going to try and optimise for a specific 'm' just yet. Or never unless that is given to us as a further challenge to do.