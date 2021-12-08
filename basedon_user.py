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

raw_text = st.text_area("Authorship Attribution Check","Enter Text Here")










st.write('---')
st.write('## Contact Our Group')


st.write("""
[Authorship Attribution](https://github.com/A-safarji) - feel free to contact!
""")
                               
