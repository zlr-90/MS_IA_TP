##############################################################################
# Algorithmic Information and Artificial Intelligence                        #
#          http://teaching.dessalles.fr/FCI            Jean-Louis Dessalles  #
# Telecom Paris  2021                                      www.dessalles.fr  #
# -------------------------------------------------------------------------- #
#                                              with Pierre-Alexandre Murena  #
##############################################################################
"""
	Test for repeated zip
"""

import sys
import os
import zlib

if __name__ == "__main__":	
	print(__doc__)
	if len(sys.argv) != 3 or not sys.argv[1].isdigit():	
		print('Usage:\t%s <number of iteration> <initial file>' % os.path.basename(sys.argv[0]))	
		print('Performs zipping iteratively on <initial file>')
		sys.exit(1)
	Str = open(sys.argv[2], 'rb').read().replace(b'\r\n', b' ').replace(b'\r', b' ').replace(b'\n', b' ')
	for zipIteration in range(int(sys.argv[1])):
		print('Iteration %d: length = %d' % (zipIteration, len(Str)))
		Str = zlib.compress(Str)



__author__ = 'Dessalles'
