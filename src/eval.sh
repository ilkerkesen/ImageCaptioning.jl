#!/bin/bash
THIS=$1
perl $1/multi-bleu.pl -lc $1/reference < $1/hypothesis
