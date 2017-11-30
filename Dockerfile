FROM 10.202.11.142/sfai/pytorch

RUN apt-get install nano
RUN pip install sconce
RUN mkdir /code

WORKDIR /code
#ADD ./ /code

