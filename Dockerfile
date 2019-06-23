FROM nvcr.io/nvidia/pytorch:19.05-py3

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install curl -y

RUN mkdir -p /src

WORKDIR /src

COPY GPT2 /src/GPT2
COPY main.py /src/main.py
COPY requirements.txt /src/requirements.txt
COPY GPT2_Pytorch.ipynb /src/GPT2_Pytorch.ipynb
COPY run_notebook.sh /src/run_notebook.sh
COPY set_password.py /src/set_password.py

RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
RUN chmod +x Anaconda3-2019.03-Linux-x86_64.sh
RUN bash Anaconda3-2019.03-Linux-x86_64.sh -b -p $HOME/miniconda

RUN curl --output gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin

RUN conda install --file requirements.txt

EXPOSE 8888

ENTRYPOINT ["sh", "/src/run_notebook.sh"]
