from typing import Union
from collections.abc import Iterable

from comparator import utils
from comparator.cfg_values import *

class Comparator:

    def __init__(self):
        

        self.threshold_dist = 6 # Distance threshold between reference and test
        # annotations below which the test annotation is considered a True Positive
        # THIS MIGHT NOT BE OPTIMAL, SHOULD IT BE CHANGED?

        self.threshold_prediction = 0.44 # Score threshold beyond which a predicted 
        # class is considered 'True'
        # THIS MIGHT NOT BE OPTIMAL, SHOULD IT BE CHANGED? + probably not necessary here


        # Loading classes for AI and videocoder annotations, as well as the classes
        # used for the comparison
        

    def compare(
            self,
            reference_annots: Union[str, dict],
            tested_annots: Union[str, dict],
            plot=False
    ) -> dict :
        """
        Compares two annotation sets for a video and returns the results.
        Args:
            reference_annots: a JSON string or dict-like structure that
            contains the annotations used as reference in the comparison.
            [COMPLETER EN ELABORANT SUR LE FORMAT SOUHAITÉ]

            tested_annots: a JSON string or dict-like structure that
            contains the annotations we want to compare with the reference
            annotations.
            [COMPLETER EN ELABORANT SUR LE FORMAT SOUHAITÉ]

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
        
        # ÉTAPES À CODER :
        # Utiliser la fonction 



    def compare_many(
            self,
            reference_annots: Iterable,
            tested_annots: Iterable,
            plot=False
    ) -> dict :
        """
        Compares the annotations contained in reference_annots with
        those contained in tested_annots, and returns global results.
        Args:
            reference_annots: an iterable that contains the annotations 
            used as reference in the comparison.
            [COMPLETER EN ELABORANT SUR LE FORMAT SOUHAITÉ]

            tested_annots: an iterable that contains the annotations 
            we want to compare with the reference annotations.
            [COMPLETER EN ELABORANT SUR LE FORMAT SOUHAITÉ]

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
            annot_test: Union[str, dict]
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
            annot_ref = utils.convert_from_ai(annot_ref)
        else:
            annot_ref = utils.convert_from_video(annot_ref)

        annot_test_format = utils.check_annot_format(annot_test)
        if annot_test_format == "AI":
            annot_test = utils.convert_from_ai(annot_test)
        else:
            annot_test = utils.convert_from_video(annot_test)




        # Format voulu au final : liste


