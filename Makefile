WIDTH = 10 #set flag
THETA = 45

.PHONY: carbon golf #set phony

carbon: carbon.py #if not run to get the pdf
	python3 carbon.py --plot=$(WIDTH)

golf: golf.py
	python3 golf.py --plot=$(THETA)
