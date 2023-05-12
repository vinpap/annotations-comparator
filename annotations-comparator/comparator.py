from typing import Union
from collections.abc import Iterable

class Comparator:

    def __init__(self):
        pass

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
        
        Returns a dictionary with the following keys:
            [A DETERMINER]

        """

        if plot==True: 
            raise NotImplementedError("Comparison results visualization not implemented yet")



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
        
        Returns a dictionary with the following keys:
            [A DETERMINER]
        """

        if plot==True: 
            raise NotImplementedError("Comparison results visualization not implemented yet")