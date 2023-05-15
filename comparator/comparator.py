from typing import Union
from collections.abc import Iterable

import pandas as pd
import numpy as np

from comparator import utils
from comparator.cfg_values import *

class Comparator:

    def __init__(self):
        

        pass
        

    def compare(
            self,
            reference_annots: Union[str, dict],
            test_annots: Union[str, dict],
            geoptis: Iterable,
            plot=False
    ) -> dict :
        """
        Compares two annotation sets for a video and returns the results.
        Args:
            reference_annots: a JSON string or dict-like structure that
            contains the annotations used as reference in the comparison.

            test_annots: a JSON string or dict-like structure that
            contains the annotations we want to compare with the reference
            annotations.

            geoptis: an iterable that contains the geolocation
            data for the video (in GEOPTIS format)

            plot: if set to True the function will also plot a global
            overview of the comparison, as well as the comparison results
            for each class (not implemented yet).
        
        Returns a dictionary where each key is a class, plus a 'total' key
        that sums up the metrics regardless of the class.
        Each item in the dictionary is a dictionary containing the following 
        elements:
            - 'precision'
            - 'recall'
            - 'f-score'
            - 'TP' (absolute value)
            - 'FP' (absolute value)
            - 'TN' (absolute value)
            - 'FN' (absolute value)
        """

        if plot==True: 
            raise NotImplementedError("Comparison results visualization not implemented yet")
        
        if type(geoptis) == pd.DataFrame: 
            first_row = geoptis.columns
            geoptis = np.vstack([first_row, geoptis.values])

        
        # ÉTAPES À CODER :
        metrics = self.__get_comparison_metrics(
            reference_annots, 
            test_annots,
            geoptis
            )



    def compare_many(
            self,
            reference_annots: Iterable,
            reference_geoptis: Iterable,
            geoptis: Iterable,
            plot=False
    ) -> dict :
        """
        Compares the annotations contained in reference_annots with
        those contained in tested_annots, and returns global results.
        Args:
            reference_annots: an iterable that contains the annotations 
            used as reference in the comparison.

            test_annots: an iterable that contains the annotations 
            we want to compare with the reference annotations.

            geoptis: an iterable that contains the geolocation
            data for the videos (in GEOPTIS format)

            plot: if set to True the function will also plot a global
            overview of the comparison, as well as the comparison results
            for each class (not implemented yet).
        
        Returns a dictionary where each key is a class, plus a 'total' key
        that sums up the metrics regardless of the class.
        Each item in the dictionary is a dictionary containing the following 
        elements:
            - 'precision'
            - 'recall'
            - 'f-score'
            - 'TP' (proportion of TP out of all anomalies)
            - 'FP' (proportion of FP out of all anomalies)
            - 'TN' (proportion of TN out of all anomalies)
            - 'FN' (proportion of FN out of all anomalies)
        """

        if plot==True: 
            raise NotImplementedError("Comparison results visualization not implemented yet")
    
    def __get_comparison_metrics(
            self,
            annot_ref: Union[str, dict],
            annot_test: Union[str, dict],
            geoptis: Iterable
    ) -> dict:
        """
        This function is based on the code produced by Théo Megy at 
        https://gitlab.logiroad.com/theom/ai-vs-videocoding/blob/master/scripts/compare_AI_videocoding.py

        It computes the comparison metrics between the annotations annot_ref and annot_test
        using the optimal distance threshold and returns a dictionary with all the metrics.
        """

        # Checker le format des annotations
        # Si format videocoder, appeler convert_from_videocoder
        # Si format AI, appeler convert_from_AI

        annot_ref_format = utils.check_annot_format(annot_ref)
        if annot_ref_format == "AI":
            annot_ref = utils.convert_from_ai(annot_ref, geoptis)
        else:
            annot_ref = utils.convert_from_video(annot_ref, geoptis)

        annot_test_format = utils.check_annot_format(annot_test)
        if annot_test_format == "AI":
            annot_test = utils.convert_from_ai(annot_test, geoptis)
        else:
            annot_test = utils.convert_from_video(annot_test, geoptis)
        


        # MAINTENANT, ON UTILISE LE CODE DE THÉO POUR CALCULER LES MÉTRIQUES
        metrics = {}
        for anomaly_class in classes_comp:
            print(f"PROCESSING CLASS {anomaly_class}...")
            class_index = classes_comp[anomaly_class]

            length_ref = np.array(annot_ref[class_index])
            length_test = np.array(annot_test[class_index])

            distances_full = [] # for AP, dim N_detection_AI
            distances_full += utils.compute_smallest_distances(length_ref, length_test).tolist()
            print(distances_full)
            """if len(lv) > 0 and len(lai) > 0:
                distances_array, score_array = utils.compute_distances(lv, lai, score, N_ai) # for average recall
                distances_array_full.append(distances_array)
                score_array_full.append(score_array)"""




