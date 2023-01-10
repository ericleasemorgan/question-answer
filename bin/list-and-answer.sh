#!/usr/bin/env bash

# list-and-answer.sh - a front-end to both list-questions and ask-a-file


# configure
QUESTIONS='./questions'
ANSWERS='./answers'
LISTQUESTIONS='./bin/list-questions.py'
ASKAFILE='./bin/ask-a-file.py'

# sanity check
if [[ -z $1 ]]; then
	echo "Usage: $0 <file>" >&2
	exit
fi

# get input
FILE=$1

# initialize
BASENAME=$( basename $FILE )

# extract questions, save, and output
$LISTQUESTIONS $FILE 2>/dev/null | sort > "$QUESTIONS/$BASENAME"
echo "Questions:"
echo
cat "$QUESTIONS/$BASENAME"

# answer questions, save, and ouput
$ASKAFILE $FILE "$QUESTIONS/$BASENAME" > "$ANSWERS/$BASENAME"
echo
echo
echo "Questions and answers:"
cat "$ANSWERS/$BASENAME"

# done
exit
