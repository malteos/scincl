FROM nvcr.io/nvidia/pytorch:20.11-py3
#FROM nvcr.io/nvidia/pytorch:19.10-py3
# See https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# - https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-11.html#rel_20-11

#FROM nvcr.io/nvidia/pytorch:20.12-py3
#FROM nvcr.io/nvidia/pytorch:21.04-py3

# Free up some space (avoid GitHub action space limit)
RUN df -h
#RUN sudo rm -rf /usr/share/dotnet
#RUN sudo rm -rf /opt/ghc
#RUN sudo rm -rf "/usr/local/share/boost"
#RUN sudo rm -rf "$AGENT_TOOLSDIRECTORY"

RUN mkdir /app
WORKDIR /app

# copy req file first
COPY requirements.txt /app/requirements.txt

# Annoy fix. See https://github.com/spotify/annoy/issues/472#issuecomment-669260570
ENV ANNOY_COMPILER_ARGS -D_CRT_SECURE_NO_WARNINGS,-DANNOYLIB_MULTITHREADED_BUILD,-mtune=native

# install our dependencies (with extra PyTorch index)
RUN pip install --find-links https://download.pytorch.org/whl/torch_stable.html -r requirements.txt

# install spacy model (needed for scidocs)
RUN python -m spacy download en_core_web_sm

# install faiss (see https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
# TODO specific cuda version?
#RUN conda install -y -c pytorch faiss-gpu
# Inofficial package / PyPi alternative to Conda https://github.com/kyamagu/faiss-wheels
#RUN pip install faiss-gpu==1.7.1

# copy all other project code
COPY . /app

# expose the Jupyter port 8888
EXPOSE 8888

# print versions
RUN nvcc --version
RUN python --version
RUN python -c "import torch; print(torch.__version__); print(torch.version.cuda)"


CMD ["jupyter", "notebook"]