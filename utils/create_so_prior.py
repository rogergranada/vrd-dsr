#!/usr/bin/python
#-*- coding: utf-8 -*-
""" 
Create `so_prior.pkl` containing the conditional probability given subject and object. It is represented 
as a matrix of  (subject, object, predicate) with shape (nb_objects, nb_objects, nb_relations), where 
`nb_objects` is the number of object categories and `nb_relations` is the number of relations categories 
in the dataset. 

Each cell contains the probability of a relation be associated with two objects as subject and object. 
For example, if we want to calculate p(hold|person, spoon), we should find all the relationships describing 
person and spoon `<person, *, spoon>` (M) and the exact relationships `<person, hold, spoon>` (N). Thus, the
probability `p(hold|person, spoon)` is acquired as by dividing `N` by `M` (\frac{N}{M}).
"""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse
from os.path import join, dirname, splitext, basename

#import progressbar as pb
import filehandler as fh
import numpy as np
import cPickle

import progressbar as pbar

def main(inputfile, output=None, class_file='classes.cfg', rels_file='relations.cfg'):
    """
    Create a `so_prior.pkl` file containing the relationship between objects. 
    """  
    if not output:
        output = join(dirname(inputfile), 'so_prior.pkl')

    # Load classes for objects from dict {0: 'rel0', 1: 'rel1'}
    # DO NOT LOAD `__background__`, thus id_person=0
    do = fh.ConfigFile(class_file, background=False).load_classes(cnames=True)
    logger.info('Loaded dictionary with {} objects.'.format(len(do)))
    dr = fh.ConfigFile(rels_file).load_classes(cnames=True)
    logger.info('Loaded dictionary with {} relations.'.format(len(dr)))

    so_prior = np.zeros((len(do), len(do), len(dr)), dtype='float64')
    objsub = np.zeros((len(do), len(do)), dtype='float64')
    logger.info('Matrix of objects and relations with shape: {}'.format(so_prior.shape))
    logger.info('Matrix of only objects with shape: {}'.format(objsub.shape))

    filerels = fh.DecompressedFile(inputfile)
    logger.info('Loading information from file: {}'.format(inputfile))
    nb_lines = filerels.nb_lines()
    pb = pbar.ProgressBar(nb_lines)
    logger.info('Processing {} lines...'.format(nb_lines))
    with filerels as frels:
        for arr in frels:
            fr, o1, r, o2 = arr[0], arr[1], arr[2], arr[3]
            idsub = do[o1]
            idrel = dr[r]
            idobj = do[o2]
            so_prior[idsub][idobj][idrel] += 1
            objsub[idsub][idobj] += 1
            pb.update()
    print
    for i in range(so_prior.shape[2]):
        so_prior[:,:,i] = np.divide(so_prior[:,:,i], objsub, out=np.zeros_like(so_prior[:,:,i]), where=objsub!=0)
    
    fout = open(output, 'wb')
    cPickle.dump(so_prior, fout)
    fout.close()
    logger.info('Saved content in file: {}'.format(output))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', metavar='relations_file', help='Path to the file containing relations between objects.')
    parser.add_argument('-o', '--output', help='Path to the file to save the conditional probabilities.')
    parser.add_argument('-c', '--class_file', help='File containing ids and their classes', default='classes.cfg')
    parser.add_argument('-r', '--relation_file', help='File containing ids and their relations', default='relations.cfg')
    args = parser.parse_args()

    main(args.inputfile, args.output)

