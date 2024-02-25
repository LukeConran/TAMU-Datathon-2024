from evaluate import test

def LoopArray(image_arry):
    drowsiness_count = 0
    total = 0
    for image in image_array:
        if(test(image)):
            drowsiness_count += 1
        total += 1
    if (drowsiness_count/total > .65):
        raise ZeroDivisionError
    else:
        return 0
    
