import numpy as np
import pandas as pd


def evaluate_false_positives(detector, image_list, class_names, thresholds,
                             ignore_class_names=list("none")):
    detections = detector.im_detect(image_list, "", "", True)
    all_scores = {}
    for threshold in thresholds:
        scores = [__process_image_detection(detection, class_names, threshold, ignore_class_names) for
                  detection in detections]
        scores_df = pd.DataFrame(scores)
        all_scores[threshold] = scores_df.mean(axis=0, skipna=True).to_dict()
    return all_scores


def __process_image_detection(detection, class_names, threshold, ignore_class_names):
    agg_scores = {k:np.nan for k in class_names}
    scores = {k:[] for k in class_names}
    for i in range(detection.shape[0]):
        cls_id = int(detection[i, 0])
        if cls_id >= 0:
            score = float(detection[i, 1])
            class_name = str(cls_id)
            if class_names and len(class_names) > cls_id:
                class_name = str(class_names[cls_id])
            if class_name not in ignore_class_names and score >= threshold:
                scores[class_name].append(score)

    for class_name in scores:
        if len(scores[class_name]) > 0:
            cls_scores = np.asarray(scores[class_name])
            cls_score = np.mean(np.square(cls_scores))
            agg_scores[class_name] = cls_score

    return agg_scores
