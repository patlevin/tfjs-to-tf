FROM python:3.9

WORKDIR /usr/src/tfjs_graph_converter

RUN git clone https://github.com/patlevin/tfjs-to-tf.git .

RUN pip install --no-cache-dir .

ENTRYPOINT [ "tfjs_graph_converter" ]
