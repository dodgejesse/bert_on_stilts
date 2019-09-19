# to get the requirements file:
# pip freeze > requirements.txt
# however, the above line caused more than one version of numpy to be installed, which
# then crashed the `import torch` line.

# either run from the directory above this one, or `cd ..`

# to examine inside the docker image
#docker run -it bert_init_eval sh

docker build -t bert_init_eval .
beaker image rename bert_init_eval bert_init_eval_old11
beaker image create -n bert_init_eval bert_init_eval
