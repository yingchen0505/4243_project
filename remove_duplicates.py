import numpy as np

def IoU(bbox1, bbox2):
    """ Compute IoU of two bounding boxes

    Args:
        bbox1 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
        bbox2 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
    Returns:
        score - IoU score
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    score = 0
    ### YOUR CODE HERE
    x_intersect = min(x1+w1, x2+w2) - max(x1,x2)
    y_intersect = min(y1+h1, y2+h2) - max(y1,y2)
    area = x_intersect * y_intersect
    score = area/(h1*w1+h2*w2-area)
    ### END YOUR CODE

    return score

def comparator(bbox1, bbox2):
    """ Compares the size of two bounding boxes

    Args:
        bbox1 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
        bbox2 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
    Returns:
        int - 2 if bbox2 is bigger, 1 if bbox1 is bigger
    """
    
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    if area1 < area2:
        return 2
    else:
        return 1
    
def remove_duplicates(ndarray):
    """ Compares the ndarray of its bounding boxes, if boxes overlap, the smaller is removed

    Args:
        ndarray - of picture number, score, followed by 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.

    Returns:
        ndarray - with duplicates gone
    """
    
    list_size = ndarray.shape[0]
    current_picture = 0
    # we will keep the possible duplicate bbox indices here, with duplicates
    overlap_list = []
    
    for ind1 in range (list_size):
        if ndarray[ind1,0] > current_picture:
            current_picture = ndarray[0,0]
        prunning_list = []
        
        
        for ind2 in range (ind1+1, list_size):
            if ndarray[ind2,0] == current_picture:
                prunning_list.append(ind2)
                
        for ind3 in prunning_list:
            overlap = IoU(ndarray[ind1,2:6], ndarray[ind3,2:6])
            if overlap > 0:
                bigger = comparator(ndarray[ind1,2:6], ndarray[ind3,2:6])
                if bigger == 1:
                    overlap_list.append(ind3)
    
    # remove duplcates in the overlap_list, a set cannot have dupes
    overlap_list = list(set(overlap_list))
    # list of indices without duplicated bboxes
    iter_ind = list(range(0, list_size))
    output_ind = [x for x in iter_ind if x not in overlap_list]
    output = ndarray[output_ind]
    
    return output
    
    
    
    