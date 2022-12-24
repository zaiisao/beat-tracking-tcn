"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: scripts/evaluate_model.py

Descrption: Evaluate a single given beat tracking model according to metrics as
            outlined in Davies et al 2009 [1]. Provides functions for other
            evaluation scripts to use.

References:
    [1] M. E. P. Davies, N. Degara, and M. D. Plumbley, ‘Evaluation Methods for
        Musical Audio Beat Tracking Algorithms’, p. 17.
"""
from argparse import ArgumentParser

from mir_eval.beat import evaluate

from beat_tracking_tcn.beat_tracker import predict_beats_from_spectrogram
from beat_tracking_tcn.dataswherets.beat_dataset import BeatDataset

def parse_args():
    parser = ArgumentParser(
        description="Perform evaluation on a given BeatNet model")
    
    
    parser.add_argument("spectrogram_dir", type=str)  #MJ:     #   == "/mount/beat-tracking/ballroom/spectrogram_dir",
    parser.add_argument("label_dir", type=str)                 # ==  "/mount/beat-tracking/ballroom/label",
    parser.add_argument("model_checkpoint", type=str)          # ==  "/home/yeol/moon/beat-tracking-tcn/checkpoints",
    parser.add_argument("--downbeats", action="store_true")

    return parser.parse_args()

def evaluate_model_on_spectrogram(
        model_checkpoint,
        spectrogram,
        ground_truth,
        downbeats=False):
    """
    Given a model checkpoint, a single spectrogram, and the corresponding
    ground truth, evaluate the model's performance on all beat tracking metrics
    offered by mir_eval.beat.
    """        

    #MJ:   Given a spectrogram, predict a list of beat times using the TCN model and  a DBN post-processor.
    
    prediction = predict_beats_from_spectrogram(
        spectrogram,
        model_checkpoint,
        downbeats=downbeats)

#mir_eval/beat.py[ evaluate()]: Compute all measures:
# # F-Measure
#     scores['F-measure'] = util.filter_kwargs(f_measure, reference_beats,
#                                              estimated_beats, **kwargs)

#     # Cemgil
#     scores['Cemgil'], scores['Cemgil Best Metric Level'] = \
#         util.filter_kwargs(cemgil, reference_beats, estimated_beats, **kwargs)

#     # Goto
#     scores['Goto'] = util.filter_kwargs(goto, reference_beats,
#                                         estimated_beats, **kwargs)

#     # P-Score
#     scores['P-score'] = util.filter_kwargs(p_score, reference_beats,
#                                            estimated_beats, **kwargs)

#     # Continuity metrics
#     (scores['Correct Metric Level Continuous'],
#      scores['Correct Metric Level Total'],
#      scores['Any Metric Level Continuous'],
#      scores['Any Metric Level Total']) = util.filter_kwargs(continuity,
#                                                             reference_beats,
#                                                             estimated_beats,
#                                                             **kwargs)

#     # Information gain
#     scores['Information gain'] = util.filter_kwargs(information_gain,
#                                                     reference_beats,
#                                                     estimated_beats,
#                                                     **kwargs)

    if downbeats:
        scores = (evaluate(ground_truth[0], prediction[0]),
                  evaluate(ground_truth[1], prediction[1]))
    else:
        scores = evaluate(ground_truth, prediction)

    return scores

def evaluate_model_on_dataset(
        model_checkpoint,
        dataset,
        ground_truths,
        downbeats=False,
        print_callback=None):
    """
    Run through a whole instance of torch.utils.data.Dataset and compare the
    model's predictions to the given ground truths.
    """        

    # Create dicts to store scores and histories
    mean_scores = {}
    mean_downbeat_scores = {}
    
    running_scores = {}
    
    running_downbeat_scores = {}

    # Iterate over dataset: compute the beat scores for each audio
    for i in range(len(dataset)):
        spectrogram = dataset[i]["spectrogram"].unsqueeze(0)
        ground_truth = ground_truths[i]

        #MJ:   Given a spectrogram, predict a list of beat times using the TCN model and  a DBN post-processor.
        scores = evaluate_model_on_spectrogram(
            model_checkpoint,
            spectrogram,
            ground_truth,
            downbeats)

    #MJ: scores: scores for all metrics:
        
    # F-Measure
    # scores['F-measure'] = util.filter_kwargs(f_measure, reference_beats,
    #                                          estimated_beats, **kwargs)

    # # Cemgil
    # scores['Cemgil'], scores['Cemgil Best Metric Level'] = \
    #     util.filter_kwargs(cemgil, reference_beats, estimated_beats, **kwargs)

    # # Goto
    # scores['Goto'] = util.filter_kwargs(goto, reference_beats,
    #                                     estimated_beats, **kwargs)

    # # P-Score
    # scores['P-score'] = util.filter_kwargs(p_score, reference_beats,
    #                                        estimated_beats, **kwargs)


        # If we're tracking downbeats, separate out the evaluation scores and
        # process independently. Otherwise, we only need to worry about beat  scores
        if downbeats:
            beat_scores = scores[0]
            downbeat_scores = scores[1]
            
            for metric in downbeat_scores:
                if metric not in running_downbeat_scores:
                    running_downbeat_scores[metric] = 0.0
                
                running_downbeat_scores[metric] += downbeat_scores[metric]
        else:
            beat_scores = scores

        for metric in beat_scores:
            if metric not in running_scores:
                running_scores[metric] = 0.0
            
            running_scores[metric] += beat_scores[metric]
        
        # Each iteration, pass our current index and our running score total
        # to a print callback function.
        if print_callback is not None:
            print_callback(i, running_scores)
            
    #END for i in range(len(dataset))
    
    # After all iterations, calculate mean scores.
    for metric in running_scores:
        mean_scores[metric] = running_scores[metric] / (i + 1)     #MJ: now i+1 is equal to the num of files in dataset   
    if downbeats:
        for metric in running_downbeat_scores:
            mean_downbeat_scores[metric] =\
                running_downbeat_scores[metric] / (i + 1)        

    # Return a dictionary of helpful information
    return {
        "total_examples": i + 1,
        "scores": mean_scores,
        "downbeat_scores": mean_downbeat_scores
    }

if __name__ == "__main__":
    args = parse_args()

    #MJ: to redirct the print outs to a file
    filename = "/home/yeol/moon/beat-tracking-tcn/score_file2"
    f = open(filename,'w')
    
    def print_callback(i, running_scores):
        """
        Evaluation function set up such that scores are passed to a callback
        after each iteration — this allows for printing or logging of results.
        This function prints results to a table in real time.
        """        
        def make_metric_heading(metric):
            # In order to fit the results on screen, let's strip all vowels,
            # except the first one, and spaces from the given metric name.
            words = metric.split(" ")
            for i, _ in enumerate(words):
                for vowel in "aeiouAEIOU":
                    words[i] = words[i][0] + words[i][1:].replace(vowel, "")
            return "".join(words)
        #END def make_metric_heading(metric)
        
        # The first iteration of the first fold, we also need to print the
        # table headnings.
        if i == 0:
            line = ""
            for metric in running_scores:
                metric_heading = make_metric_heading(metric)
                heading = " %s " % metric_heading
                # Pad any headings shorter than 6 characters so that we have
                # enough space for at least 4 decimal places.
                if len(metric_heading) < 6:
                    padding_length = int((6 - len(metric_heading)) / 2)
                    padding = " " * padding_length
                    heading = padding + heading + padding
                    if len(metric_heading) % 2 == 1:
                        heading += " "
                heading += "|"
                line += heading
            print(line, file=f)

        # Build a line of scores, truncating the decimal places to match the
        # length of the given heading.
        line = ""
        for metric in running_scores:
            metric_heading = make_metric_heading(metric)
            number_length = len(metric_heading) - 2
            line += " {1:.{0}f} |".format(
                max(4, number_length),
                running_scores[metric] / (i + 1))
        # Print, overwriting the previously printed line each time.
        print(line, end="\r", file=f)  #MJ: https://stackoverflow.com/questions/7152762/how-to-redirect-print-output-to-a-file
        
    #END def print_callback(i, running_scores)
    
    #JA: Load the dataset from the given directories
    dataset = BeatDataset(
        args.spectrogram_dir,
        args.label_dir,
        dataset_name="ballroom",
        subset="test",
        validation_fold=0,
        downbeats=args.downbeats)
    

    # Process downbeats and beats independently if necessary
    if args.downbeats:
        ground_truths = tuple(zip(
            [dataset.get_ground_truth(i) for i in range(len(dataset))],
            [dataset.get_ground_truth(i, downbeats=True)
                for i in range(len(dataset))]))
    else:
        ground_truths = (dataset.get_ground_truth(i) for i in range(len(dataset)))

    # Run evaluation
    evaluate_model_on_dataset(
        args.model_checkpoint,
        dataset,
        ground_truths,
        args.downbeats,
        print_callback)