(14th March 2026)

#### JASPAR-related links:

1. https://jaspar.elixir.no/matrix/MA0139.1/
2. https://jaspar.elixir.no/matrix/MA0138.1/

---
---

(15th February 2026)

# @Bakash: Instructions and tasks for you to do

1. Optimise the model.py script as much as possible. Use an LLM if you have to. Also rewrite run.py so that I can use it to run multiple versions of model.py on a cluster/remote-server for different hyperparameter sets.
2. Run model.py for K = 3 and 5, and for each of the 3 TFs, and for m = 0 to 10 (both included), and all 19 chromosomes that are given to us as part of the training dataset. Modify plot_stats.py such that it generates order-statistics for the different markov model classifiers obtained from all of these runs. The goal is to see what 'm' has the highest AUROC and ARPRC for a given transcription factor and whether they all more or less come out to be roughly 6-7 base pairs long (or different). This will also act as a baseline performance for us to improve upon when constructing better and better models.
3. Write a script to analyse just the four labels given to each of the sequences. Look at whether any and all regions that have a 'B' for any one of the three TFs, CTCF/REST/EP300, also have ATAC-Seq as a 'B', or if there are bins with a 'U' on the ATAC-Seq column instead. Look at how many such sequences there are and separate them. Also using this same script, look at how much 'overlap' there are between labels. As in, check whether the probability that any pair or trio of 'B's occur together is significantly more than expected. Use a venn diagram to visualise the numbers, if it helps. Finally, using this same script, isolate the number of sequences corresponding to each of the 2^4 possible binary label-vectors
4. Next, write a script to group adjacently-positioned bins together. As in, from the dataset that you have, if there are bins with one or more 'B's in them, and those labels for each corresponding transcription factor are the same for adjacent bins, then group those bins together and generate and save 'grouped' bins/sequences of varying lengths into a fasta file. Plot the distribution of such variable sequence lengths and look at the plot. Do this for each of the three transcription factors.
5. Obtain the Position-Weight Matrices (PWMs) for each of the three transcription factors from the JASPAR database. Then, run motif discovery/Gibbs Sampling/similar algorithms on (i.) the individual bins and (ii.) the grouped bins/sequences and see whether you find the corresponding binding site motifs according to the associated sequence label vectors. Record the position of occurence of the binding site, especially if there is an overlap between bins or if the binding site sits across two individual bins. Plot the distribution of locations and do some further analysis.
6. Look at whether, regardless of what the label vectors tell you, you are able to actually find the binding sites for more than one type of transcription factor (or more than one site for one chosen factor) in an individual bin versus a grouped bin. Look for any other related patterns or correlations.

The goal of all of this is to see whether it makes sense to even run a conventionally-deterministic inference algorithm or use a deep learning architecture directly, and the hope is that these results will give us some idea to the following conjectures that I have:
- multiple transcription factor binding sites occur in close proximity to each other (is my guess), therefore the sequence labels need to be predicted in one go, and therefore the model must train and infer all three at once, not independently
- ATAC-Seq information needs to be taken into account separately and properly
- Depending on binding site and location, one also needs to decide between individual versus grouped bins, and either increase the sequence sizes or consider positional information to be another input into the model
- Remember that the ultimate goal of this project is NOT to search for binding sites!! It is to reproduce and accurately predict the ChIP-Seq data, since that is all that we have from experiments! THERE IS A CLEAR DIFFERENCE, AND THESE TWO TASKS ARE STRICRLY **NOT** THE SAME!!!

Hopefully the answers and hints to some of these questions will give us a guide to what sort of modelling architecture/algorithm we are going to have to resort to using.

---

(12th February 2026)

# Ideas for CFG:

- optimise your markov classifier code using an LLM (without changing whatever you did in it)
- check and confirm that the labels are not 'independent' - there is a more than random chance that a sequence encodes for both CTCF and REST together or otherwise, than for any one at a time
- How do the ATAC-seq labels feature in this?... do the ATAC-seq labels help with figuring out the ChiP-Seq data?
-----------------------------------------------------------------------------
- All of the ChIP-Seq peaks have been further divided into 200-bps bins, our goal is NOT to identify binding sites, but to predict whether the bin falls within the ChIP-Seq data or not...
- And we can use any and all information for this, even spatial/positional information... the only thing we are not allowed to use is more ChIP-Seq data!
- Remember that the ATAC-Seq information is also quite useful in this regard
-----------------------------------------------------------------------------
- we have chromosomes + bin-positions
- bin-positions need to be accounted for if you cannot find a motif in any one sequence/bin - first do an analysis of the positions and how 'clustered' all these bins/sequences are
- use JASPAR - see if you can find binding sites in each 'cluster' of bins, and also see if you can find more than one motif for the same TF
- look at the locations... do you see binding sites that cross-over bins (i.e. they lie half in one bin and half in the other?...)
- do you find more than one TF's binding site per sequence/bin?
-----------------------------------------------------------------------------
- what about using K-mer information?
- Atreyi's method?... does it even make sense to use it for such short sequences?
-----------------------------------------------------------------------------
- try out other algorithms: SVM, Naive Bayes, etc
- maybe even cook up an "Ensemble" classifier...
-----------------------------------------------------------------------------
- neural net -> variational autoencoder
- but using CNN layers or Recurrent layers?
- Or just use simple feed-forward layers?
- this is a generative classifier... can an analysis of the latent space provide useful insights here?
- should we dare to try attention-based mechanisms on this stuff? or is that too much for this course?

---

(11th February 2026)

**Note**: I believe I've made an indexing errormethod when I obtained the sequences as fasta files... I think I assumed 0-indexing when in fact the .tsv files are written with 1-indexing (1 is the 'start' like in the sane world, and not '0' like in python). 

This should hopefully not affect learning rates too much since the binding sites hopefully lie somewhere in between, but we will need to correct this anyways before we submit the midsem assignment stuff.

--- 

(from late-January 2026)

**@Bhadralok:** I spoke to Leelavati the other day... she said that the midsem evaluation will be a **viva + a google form** that we will have to fill out with results obtained from running our code for different use-cases. As of now, we should be able to make quick changes or execute multiple runs over different hyperparams if the google form requires us to, or even modify the code-blocks in `model.py` accordingly to make fresh plots.

But I would like you to go over the code properly **(and please give me feedback if you think my code is illegible, or if you think I should add in more comments, and so on)**. This is so that you don't blank out during the viva, since most of the code in this repo currently is stuff written by me. So plims, _make sure you understand at least the functions in model.py_, the rest of the repo is just additional stuff.

### A further note: 

We can adapt the same structure of `model.py` for other algorithms that we may be interested in applying post-midsem. In case it hasn't hit you, the part before the markov_classifier_function and the part from score_arrays and so on, will more or less remain the same. In between, for each fold during K-Fold C.V., we will have a train set as well as a test set. We will use the train set to test the neural network, and then use the test set for the C.V., and get our scores for calculating the two AUC-values. But again, this is stuff for later, and we will have to figure this out based on how ambitious we are + how much time and effort we can both put into coding up the algorithm + doing exploratory data analysis. 