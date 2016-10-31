# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "im2txt/model/train/model.ckpt-2000000",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "im2txt/data/mscoco/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)
  print(filenames)


  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    result = []

    for filename in filenames:
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        prob = math.exp(caption.logprob)
        print("  %d) %s (p=%f)" % (i, sentence, prob))
        result.append({
          "prob": "%f" % prob,
          "sentence": sentence
        })
    return {"result": result}



# webapp
from flask import Flask, jsonify, render_template, request, redirect
import os
import uuid
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from werkzeug.utils import secure_filename
import urllib
import pprint


UPLOAD_FOLDER = '/tmp/'
app = Flask(__name__)

app.config['DEBUG'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/url', methods=['GET', 'POST'])
def url():
  # check if the post request has the file part
  url = request.query_string[2:]
  if url == "":
    print("not url")
    return redirect("/")
  title = uuid.uuid4().hex
  path = os.path.join(app.config['UPLOAD_FOLDER'], title)

  # get url file
  req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
  any_url_obj = urllib.request.urlopen(req)
  local = open(path, 'wb')
  local.write(any_url_obj.read())
  # convert to jpg
  im = Image.open(path)
  if im.mode != "RGB":
    im = im.convert("RGB")
  im.save(path+".jpg", "JPEG")

  any_url_obj.close()
  local.close()
  FLAGS.input_files = path+".jpg"
  # inference
  result = main(None)
  return jsonify(result)

@app.route('/api/upload', methods=['POST'])
def upload():
  pprint.pprint(request.files)

  # check if the post request has the file part
  if 'file' not in request.files:
    print("not file")
    return redirect("/")
  file = request.files['file']
  # if user does not select file, browser also
  # submit a empty part without filename
  if file.filename == '':
    return redirect("/")
  filename = secure_filename(file.filename)
  path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
  file.save(path)
  FLAGS.input_files = path
  # inference
  result = main(None)
  print(result)
  return jsonify(result)

@app.route('/')
def root():
  return render_template('index.html')

if __name__ == "__main__" and FLAGS.input_files != "":
  tf.app.run()
