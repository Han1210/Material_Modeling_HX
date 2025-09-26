FXN = cos,sin,sinc #set flag
TXT = test.txt
FMT = pdf
.PHONY: plot write read #set phony

plot: function_figure.pdf  #check if pdf exist
function_figure.pdf: trignometry.py #if not run to get the pdf
	python3 trignometry.py --function=$(FXN) --print=$(FMT)

write: test.txt  #check if the txt file exist
test.txt: #if not, run to get the txt
	python3 trignometry.py --function=$(FXN) --write=$(TXT)

read: read_file_figure.pdf  #check if the pdf file exist
read_file_figure.pdf: trignometry.py #if not run the python to get the pdf
	python3 trignometry.py --function=$(FXN) --read=$(TXT) --print=$(FMT)


