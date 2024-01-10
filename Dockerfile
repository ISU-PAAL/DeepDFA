FROM nvcr.io/nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Install dependencies
RUN apt clean && \
    apt update -y && \
    apt install -y \
        curl git build-essential \
        openjdk-11-jdk unzip

# Install conda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-py311_23.10.0-1-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py311_23.10.0-1-Linux-x86_64.sh -b -u -p /opt/conda \
    && rm -f Miniconda3-py311_23.10.0-1-Linux-x86_64.sh
ENV PATH /opt/conda/bin:$PATH

RUN mkdir /DeepDFA
WORKDIR /DeepDFA

# Create environment
COPY environment.yml .
RUN conda env create -f environment.yml && \
    conda init bash && \
    echo "conda activate deepdfa" >> $HOME/.bashrc
ENV LD_LIBRARY_PATH /opt/conda/envs/deepdfa/lib:$LD_LIBRARY_PATH
ENV PYTHONPATH /DeepDFA/DDFA:$PYTHONPATH

# Install Joern
COPY scripts/install_joern.sh .
RUN bash install_joern.sh
ENV PATH /DeepDFA/joern/joern-cli:$PATH

COPY . .
