# mscs-lewisu-thesis-code
Classification with Reject in Bayesian Deep Learning for Semantic Segmentation: Automating Pixelwise Reject Thresholds Using Bayes Error

Data from here (http://amsacta.unibo.it/id/eprint/6706/) should be placed in the 'data/' directory in its raw unzipped format. Run the Python scripts 'resize.py' and 'resize1.py' in order to process that data into the datasets used in this experiment.

I originally confused the positive and negative class values, which affected TP, TN, FP, FN values that were recorded/generated. TP and TN were swapped and FP and FN were swapped. I manually edited the tsv files in the results/ directory to correct this. The error occurred because I thought a pixel value of 0 was white and 255 was black; the reverse is true.
