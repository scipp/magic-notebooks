import numpy

def remove_coords_in_da_except(da, *argv):
    names = numpy.copy(list(da.coords.keys()))
    for name in names:
        if not(name in argv):
            da.coords.pop(name)
    return

def remove_coords_in_da(da, *argv):
    for name in argv:
        if name in da.coords.keys():
            da.coords.pop(name)
    return