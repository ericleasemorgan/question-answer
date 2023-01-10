#!/usr/bin/env python

# list-questions.py - given a file, output a list of questions

# Eric Lease Morgan <emorgan@nd.edu>
# (c) University of Notre Dame; distribured under a GNU Public License

# December 31, 2022 - first investigations


# require
from haystack.nodes import QuestionGenerator
import sys

# get input
if len( sys.argv ) != 2 : sys.exit( 'Usage: ' + sys.argv[ 0 ] + " <file>" )
file = sys.argv[ 1 ]

# initialize
with open( file ) as handle : text = handle.read()
pipeline = QuestionGenerator()

# do the work, output, and done
questions = pipeline.generate( text )
for question in questions : print( question )
exit()

	