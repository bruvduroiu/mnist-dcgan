FROM continuumio/anaconda3
COPY main.py /
RUN ["conda", "install", "-y", "keras"]
RUN python main.py
