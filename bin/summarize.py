#!/usr/bin/env python

# summarize.py - given a file, output a list of questions


# Eric Lease Morgan <emorgan@nd.edu>
# (c) University of Notre Dame; distribured under a GNU Public License

# December 31, 2022 - first investigations


# require
from haystack.nodes import TransformersSummarizer
from haystack       import Document
import sys

# get input
if len( sys.argv ) != 2 : sys.exit( 'Usage: ' + sys.argv[ 0 ] + " <file>" )
file = sys.argv[ 1 ]

with open( file ) as handle : text = handle.read()
docs   = [ Document( text ) ]

summarizer = TransformersSummarizer(model_name_or_path="google/pegasus-xsum",  max_length=len( text ) )
summaries = summarizer.predict(documents=docs )
for summary in summaries :

	summary = summary.to_dict()
	print( summary[ 'meta' ][ 'summary' ] )

