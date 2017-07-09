FROM mxnet:python3-mxnet0.10.0

COPY ./ /ssd
WORKDIR /ssd

# Install special requirements for visual backprop
RUN cd visualBackprop && pip install -r requirements.txt
