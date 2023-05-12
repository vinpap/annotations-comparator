threshold_dist = 6 # Distance threshold between reference and test
# annotations below which the test annotation is considered a True Positive
# THIS MIGHT NOT BE OPTIMAL, SHOULD IT BE CHANGED?

threshold_prediction = 0.44 # Score threshold beyond which a predicted 
# class is considered 'True'
# THIS MIGHT NOT BE OPTIMAL, SHOULD IT BE CHANGED? + probably not necessary here


# Loading classes for AI and videocoder annotations, as well as the classes
# used for the comparison
classes_vid = {
    'Affaissement de rive significatif' : '',
    'Affaissement de rive grave' : '',
    'Affaissement hors rive significatif' : '',
    'Affaissement hors rive grave' : '',
    'Ornierage significatif' : '',
    'Ornierage grave' : '',
    'Arrachement significatif' : 'Arrachement_pelade',
    'Arrachement grave' : 'Arrachement_pelade',
    'Faiencage significatif non BDR' : 'Faiencage',
    'Faiencage grave non BDR' : 'Faiencage',
    'Faiencage specifique BDR' : 'Faiencage',
    'Fissure longitudinale BDR reparee' : 'Pontage',
    'Fissure longitudinale BDR significative' : 'Longitudinale',
    'Fissure longitudinale BDR grave' : 'Longitudinale',
    'Fissure longitudinale HBDR reparee' : 'Pontage',
    'Fissure longitudinale HBDR significative' : 'Longitudinale',
    'Fissure longitudinale HBDR grave' : 'Longitudinale',
    'Fissure transversale reparee' : 'Pontage',
    'Fissure transversale significative' : 'Transversale',
    'Fissure transversale grave' : 'Transversale',
    'Nid de poule significatif' : 'Nid_de_poule',
    'Nid de poule grave' : 'Nid_de_poule',
    'Ressuage - Glacage localise' : '',
    'Ressuage - Glacage generalise' : '',
    'Reparation en BB sur decoupe Petite largeur' : 'Remblaiement_de_tranchees',
    'Reparation en BB sur decoupe Pleine largeur' : 'Remblaiement_de_tranchees',
    'Autre reparation Petite largeur' : 'Autre_reparation',
    'Autre reparation Pleine largeur' : 'Autre_reparation'
    }

classes_AI = {
    'Arrachement_pelade' : 'Arrachement_pelade',
    'Faiencage' : 'Faiencage',
    'Nid_de_poule' : 'Nid_de_poule',
    'Transversale' : 'Transversale',
    'Longitudinale' : 'Longitudinale',
    'Pontage_de_fissures' : 'Pontage',
    'Remblaiement_de_tranchees' : 'Remblaiement_de_tranchees',
    'Raccord_de_chaussee' : '',
    'Comblage_de_trou_ou_Projection_d_enrobe' : 'Autre_reparation',
    'Bouche_a_clef' : '',
    'Grille_avaloir' : '',
    'Regard_tampon' : ''
    }

classes_comp = {
    'Arrachement_pelade' : 0,
    'Faiencage' : 1,
    'Nid_de_poule' : 2,
    'Transversale' : 3,
    'Longitudinale' : 4,
    'Remblaiement_de_tranchees' : 5,
    'Pontage' : 6,
    'Autre_reparation' : 7,
    }