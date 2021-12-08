import streamlit as st
import numpy as np 
import pandas as pd
import tensorflow as tf
import logging
import pandas as pd
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv1D,
    Dropout,
    Input,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from transformers import TFBertModel
from transformers import BertTokenizer
model_name = "aubmindlab/bert-base-arabertv2"
max_length = 512
batch_size = 128
num_class = 95

tf.compat.v1.logging.set_verbosity(2)
tokenizer = BertTokenizer.from_pretrained(model_name)


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)

def arabert_encode(data):
    tokens = tokenizer.batch_encode_plus(
        data, max_length=max_length, padding="max_length", truncation=True
    )
    return tf.constant(tokens["input_ids"])

def arabert_model():
    bert_encoder = TFBertModel.from_pretrained(model_name, output_attentions=True)
    input_word_ids = Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    last_hidden_states = bert_encoder(input_word_ids)[0]
    clf_output = Flatten()(last_hidden_states)
    net = Dense(512, activation="relu")(clf_output)
    net = Dropout(0.3)(net)
    net = Dense(440, activation="relu")(net)
    net = Dropout(0.3)(net)
    output = Dense(num_class, activation="softplus")(net)
    model = Model(inputs=input_word_ids, outputs=output)
    return model

from keras import backend as K

steps_per_exe =32

with strategy.scope():
  model = arabert_model()
  adam_optimizer = Adam(learning_rate=1e-5)
  model.compile(
        loss="categorical_crossentropy",
        optimizer=adam_optimizer,
        metrics=['acc'],
        steps_per_execution=steps_per_exe
    )

@st.cache(allow_output_mutation=True)
#model.load_weights("gs://axial-trail-334408-tf2-models/book-mnist")


#text = 'الرأي فإنه متى ما اتبع الرأي جاءه رجل آخر أقوى في الرأي منه فاتبعه فكلما غلبه رجل اتبعه أرى أن هذا بعد لم يتم واعملوا من الآثار بما روي عن جابر رضي الله عنه أن النبي صلى الله عليه وسلم قال قد تركت فيكم ما لن تضلوا بعدي إذا اعتصمتم به كتاب الله وسنتي ولن يتفرقا حتى يردا على الحوض وروي عن عمرو بن شعيب عن أبيه عن جده خرج رسول الله صلى الله عليه وسلم يوما وهم يجادلون في القرآن فخرج ووجهه أحمر كالدم فقال يا قوم على هذا هلك من كان قبلكم جادلوا في القرآن وضربوا بعضه ببعض فما كان من حلال فاعملوا به وما كان من حرام فانتهوا عنه وما كان من متشابه فآمنوا به'
text_encoded = arabert_encode(text)
p = (
    tf.data.Dataset.from_tensor_slices((text_encoded))
    .batch(batch_size)
).cache()
y_pred = model.predict(p, verbose=2)
	
	
author , book = df.iloc[[np.argmax(x) for x in y_pred][0]].tolist()
st.write('Author: ',author , '\nBook: ', book, '\nConfidence:', y_pred[0][[np.argmax(x) for x in y_pred][0]])
	
	
	
	
	
	
	
#pio.renderers.default = 'chrome'
st.set_option('deprecation.showPyplotGlobalUse', False)
#st.set_page_config(layout="wide")

#st.title('Recommended for you!')
st.markdown(' <p align="center" class="big-font">  <b>Authorship Attribution <u> 🌟 T5 🇸🇦</b>   </p>', unsafe_allow_html=True)	


st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)



st.markdown("""
إسناد التأليف العربي هو مهمة البحث عن مؤلف المستند. لتحقيق هذا الغرض ، يقارن المرء نص الاستعلام بنموذج المؤلف المرشح ويحدد احتمال نموذج الاستعلام.

Arabic authorship attribution is the task of finding the author of a document.
To achieve this purpose, one compares a query text with a model of the candidate author and determines the likelihood of the model for the query.
	""")

#text = st.text_area("Authorship Attribution Check","Enter Text Here")










st.write('---')
st.write('## Contact Our Group')


st.write("""
[Authorship Attribution](https://github.com/A-safarji) - feel free to contact!
""")
                               
