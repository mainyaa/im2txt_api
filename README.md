# Show and Tell: A Neural Image Caption Generator Demo API

Demo source files extracted from original TensorFlow Models source. (TensorFlow r0.10)

To build this demo, you don't need to prepare build environment with Bazel.

Original repository: http://github.com/tensorflow/models/im2txt/
Full text: http://arxiv.org/abs/1609.06647

## Setup

Tensorflow setup: https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html
Git Large File Storage setup: https://git-lfs.github.com/

```
git clone https://github.com/mainyaa/im2txt_api
cd im2txt_api
git lfs pull
pip install -r requirements.txt
```

```
cd im2txt/data
wget "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
tar -xvf "inception_v3_2016_08_28.tar.gz"
rm "inception_v3_2016_08_28.tar.gz"
```

## Run simple

```
python -m im2txt.run_inference --input_files wii.jpg
```

## Run API Server

```
sudo gunicorn im2txt.run_inference:app --log-file=-
open http://localhost:8000/
```

## Hubot

script/im2txt.coffee
```
module.exports = (robot) ->
  robot.respond /im2txt +(.*)/i, (msg) ->
    # Send POST request
    request = require 'request'
    endpoint = 'http://[ADDRESS]/api/url?q=' + msg.match[1]
    request.get
      url: endpoint
    , (err, response, body) ->
      # Reply
      msg.reply "TensorFlow Show and Tell:\n```\n#{body}\n```"

```
