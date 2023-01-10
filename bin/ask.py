#!/usr/bin/env python

# ask.py - given a corpus (study carrel) and file of questions, output a set of answers
# see: https://haystack.deepset.ai/tutorials/03_basic_qa_pipeline_without_elasticsearch

# Eric Lease Morgan <emorgan@nd.edu>
# (c) University of Notre Dame; distribured under a GNU Public License

# December 31, 2022 - first investigations; very cool
# January   2, 2023 - added smarter documents
# January   4, 2023 - added many questions as input


# configure
MAXIMUM       = 12
THRESHOLD     = .1
SPLITLENGTH   = 100
RESPECT       = False
INDEXES       = './indexes'
MODEL         = 'deepset/roberta-base-squad2'
EMBEDDEDMODEL = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
MODELFORMAT   = 'sentence_transformers'
DB            = '##CARREL##.db'
INDEX         = '##CARREL##.faiss'
CONFIG        = '##CARREL##.json'
URL           = 'sqlite:///##DB##'

# require
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes           import EmbeddingRetriever
from haystack.nodes           import FARMReader
from haystack.nodes           import PreProcessor
from haystack.pipelines       import ExtractiveQAPipeline
from pathlib                  import Path
import rdr
import re
import sys

# get input
if len( sys.argv ) != 3 : sys.exit( 'Usage: ' + sys.argv[ 0 ] + " <carrel> <questions>" )
carrel    = sys.argv[ 1 ]
questions = sys.argv[ 2 ]


def carrel2documents( carrel ) :

	'''Given a study carrel, output a list of documents amenable to an FAISSDocumentStore'''
	
	# configure and initialize
	PATTERN   = '*.txt'
	documents = []
	corpus    = rdr.configuration( 'localLibrary' )/carrel/rdr.TXT
	
	# process each plain text file in the given carrel
	for file in corpus.glob( PATTERN ) :
		
		# slurp up the given file
		with open( file ) as handle : content = handle.read()
		
		# normalize the content and file name
		content = content.replace( '-\n', '' )
		content = content.replace( '- ', '' )
		content = content.replace( " 's", "'s" )
		name    = file.name
		
		# create a document and update the list of documents
		document = { 'content':content, 'meta':{ 'name':name } }
		documents.append( document )
				
	# done
	return( documents )


# initialize index file names
indexes = Path( INDEXES )
db      = indexes/( DB.replace( '##CARREL##', carrel ) )
index   = indexes/( INDEX.replace( '##CARREL##', carrel ) )
config  = indexes/( CONFIG.replace( '##CARREL##', carrel ) )

# get questions
with open( questions ) as handle : questions = handle.read().splitlines()

# check to see if the files exist; index things
if ( not db.exists() ) or ( not index.exists() ) or ( not config.exists() ) :

	# index, and create a data store
	sys.stderr.write( 'Indexing. Please be patient; if you had to read whole books, it would take you a long time too.\n' )
	preprocessor = PreProcessor( split_length=SPLITLENGTH, split_respect_sentence_boundary=RESPECT )
	documents    = preprocessor.process( carrel2documents( carrel ) )
	store        = FAISSDocumentStore( sql_url=URL.replace( '##DB##', str( db ) ) )
	store.write_documents( documents )
	
	# index some more and save
	retriever = EmbeddingRetriever( document_store=store, embedding_model=EMBEDDEDMODEL, model_format=MODELFORMAT )
	store.update_embeddings( retriever )
	store.save( index_path=index )
	
# load the index files
else :

	# load the previously created index
	store  = FAISSDocumentStore( faiss_index_path=index )
	store.load( index_path=index, config_path=config )
	retriever = EmbeddingRetriever( document_store=store, embedding_model=EMBEDDEDMODEL, model_format=MODELFORMAT, progress_bar=False )

# initialize a retriever, reader, and pipeline; search
reader   = FARMReader( model_name_or_path=MODEL, use_gpu=False, progress_bar=False )
pipeline = ExtractiveQAPipeline( reader, retriever )

# process each question
for question in questions :

	# search
	results  = pipeline.run( query=question, params={ "Reader" : { "top_k": MAXIMUM } } )

	# initialize output and process each result
	print( '\n' + question + '\n' )
	for result in results[ 'answers' ] :

		# parse
		result  = result.to_dict()
		answer  = result[ 'answer' ]
		score   = result[ 'score' ]
		context = result[ 'context' ]
		file    = result[ 'meta' ][ 'name' ]
	
		# we only want relevant documents
		if score < THRESHOLD : break
	
		# normalize
		answer  = answer.replace( '\n', ' ' ).replace( '\t', ' ' )
		answer  = re.sub( ' +', ' ', answer )
		context = context.replace( '\n', ' ' ).replace( '\t', ' ' )
		context = re.sub( '^ +', '', context )
		context = re.sub( ' +', ' ', context )

		# output
		print( '   answer:', answer )
		print( '  context:', context )
		print( '    score:', score )
		print( '     file:', file )
		print()
	
# done
exit()

	