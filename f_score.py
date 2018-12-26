"""
This module provides functionality for calculating the F-Score
"""

def macro_f1(true_labels,predicted_labels):
	assert len(true_labels)==len(predicted_labels)

	false_negatives = dict()
	false_positives= dict()
	true_positives= dict()
	f1_scores=dict()

	for i in range(len(predicted_labels)):
		predicted_label=predicted_labels[i]
		true_label=true_labels[i]

		if not predicted_label in false_negatives:
			false_negatives[predicted_label]=0.0
		if not predicted_label in false_positives:
			false_positives[predicted_label]=0.0
		if not predicted_label in true_positives:
			true_positives[predicted_label]=0.0

		if not true_label in false_negatives:
			false_negatives[true_label]=0.0
		if not true_label in false_positives:
			false_positives[true_label]=0.0
		if not true_label in true_positives:
			true_positives[true_label]=0.0

		if predicted_label==true_label:
			true_positives[true_label]=true_positives[true_label]+1.0
		else:
			false_negatives[true_label]=false_negatives[true_label]+1.0
			false_positives[predicted_label]=false_positives[predicted_label]+1.0

	for label in true_positives:
		try:
			precision=true_positives[label]/(true_positives[label] + false_positives[label])
			recall=true_positives[label]/(true_positives[label] + false_negatives[label])

			f1_scores[label]=((2.0*precision*recall)/(precision+recall))
		except Exception:
			f1_scores[label]=0.0

	s=float(sum(f1_scores.values()))
	l=float(len(f1_scores.keys()))

	return s/l