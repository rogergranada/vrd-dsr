#!/usr/bin/python
#-*- coding: utf-8 -*-
""" 
Create `train.pkl` or `test.pkl` containing a Python list where each item is a dictionary 
with the following keys: 

'img_path': path to access the image in the computer
   : string : '../data/sg_dataset/sg_train_images/5526173107_447a4419bf_b.jpg'
'classes': list of classes of the objects contained in a single image
   : array : [1, 97, 47, 8, 8, 8]
'boxes': list bounding boxes of the objects contained in a single image
   : array : [[0, 0, 1023, 357],
              [1, 352, 1017, 679],
              [484, 210, 619, 251],
              [540, 345, 859, 473],
              [0, 352, 407, 510],
              [490, 354, 587, 400]]
'ix1': index of the box in `boxes` that belongs to subjects
    : array : [0, 1, 4, 2, 5]
'ix2':  index of the box in `boxes` that belongs to objects
    : array : [1, 2, 1, 3, 1]
'rel_classes': the relationship for a subject-object pair.
    : array : [[26], [15], [0], [10], [0]]
"""
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse
from os.path import join, dirname, splitext, basename
from collections import defaultdict

import filehandler as fh
import numpy as np
import cPickle
import operator

import progressbar as pbar 


def save_dictionary(filename, dic):
    """ Save the dictionary into file sorted by id """
    logger.info('Saving dictionary at: {}'.format(filename))
    sorted_dic = sorted(dic.items(), key=operator.itemgetter(1))
    with open(filename, 'w') as fout:
        for key, id in sorted_dic:
            fout.write('{}\n'.format(key))


def main(fileobj, filerel, output=None, class_file='classes.cfg', rels_file='relations.cfg', map_paths='map_paths.txt'):
    """
    Create a `train.pkl` or 'test.pkl` file containing the relationship between objects. 

    TODO: Implement relations for two objects of the same class in the same image
    """  
    if not output:
        output = join(dirname(fileobj), 'train.pkl')
    fdicobj = join(dirname(output), 'obj.txt')
    fdicrel = join(dirname(output), 'rel.txt')

    if map_paths:
        fmap = fh.MapFile(map_paths)
        dmap = fmap.load_dictionary(key='kscgr')
        logger.info('Loaded map file containing {} entries.'.format(len(dmap)))
        home = fmap.path
   
    # Load classes for objects from dict {0: 'rel0', 1: 'rel1'}
    # DO NOT LOAD `__background__`. Thus, id_person=0
    do = fh.ConfigFile(class_file, background=False).load_classes(cnames=True)
    logger.info('Loaded dictionary with {} objects.'.format(len(do)))
    dr = fh.ConfigFile(rels_file).load_classes(cnames=True)
    logger.info('Loaded dictionary with {} relations.'.format(len(dr)))

    dic_rels = defaultdict(list) # relations for each image
    logger.info('Loading information from file: {}'.format(filerel))
    filerls = fh.DecompressedFile(filerel)
    pb = pbar.ProgressBar(filerls.nb_lines())

    with filerls as frels:
        for fr, o1, r, o2, path in frels:
            idsub = do[o1]
            idrel = dr[r]
            idobj = do[o2]
            pathimg = join(path, str(fr)+'.jpg')
            if map_paths:
                pathimg = dmap[join(home, pathimg)]
            dic_rels[pathimg].append((idsub, idrel, idobj))
            pb.update()
    print 
    info = []
    # Load objects
    logger.info('Loading information from file: {}'.format(fileobj))
    flis = fh.LisFile(fileobj)
    nb_frames = filerls.nb_frames()
    pb = pbar.ProgressBar(nb_frames)
    logger.info('Processing {} frames.'.format(nb_frames))
    with flis as fin:
        for imgname, arr in flis.iterate_frames():
            filepath = dmap[join(home, imgname)]
            classes, boxes = [], []
            vsub, vobj, vrel = [], [], []
            dor = {}
            for i in range(len(arr)):
                obj, x, y, w, h = arr[i]
                iobj = do[obj]
                dor[iobj] = i
                classes.append(iobj)
                boxes.append([x, y, x+w, y+w]) # [xmin,ymin,xmax,ymax]
            for idsub, idrel, idobj in dic_rels[filepath]:
                vsub.append(dor[idsub])
                vobj.append(dor[idobj])
                vrel.append([idrel])
            
            info.append({
                'img_path': filepath,
                'classes': np.array(classes),
                'boxes': np.array(boxes),
                'ix1': np.array(vsub),
                'ix2': np.array(vobj),
                'rel_classes': vrel
            })
            pb.update()
    
    logger.info('Saving pickle file...')
    fout = open(output, 'wb')
    cPickle.dump(info, fout)
    fout.close()
    logger.info('Saved content in file: {}'.format(output))

    save_dictionary(fdicobj, do)
    save_dictionary(fdicrel, dr)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('objfile', metavar='file_objects', help='Path to the file containing relations between objects.')
    parser.add_argument('relfile', metavar='file_relations', help='Path to the file containing relations between objects.')
    parser.add_argument('-o', '--output', help='Path to the file to save the conditional probabilities.')
    parser.add_argument('-c', '--cfg_objects', help='File containing ids and their classes', default='classes.cfg')
    parser.add_argument('-r', '--cfg_relations', help='File containing ids and their relations', default='relations.cfg')
    parser.add_argument('-m', '--map_voc', help='File containing a mapping between LIS and VOC', default='map_paths.txt')
    args = parser.parse_args()

    main(args.objfile, args.relfile, args.output, args.cfg_objects, args.cfg_relations, args.map_voc)

