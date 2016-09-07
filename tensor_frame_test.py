import os
import sys

# Path for spark source folder
os.environ['SPARK_HOME']="/Users/marhamil/Documents/Spark/spark-2.0.0-bin-hadoop2.7"
# Append pyspark  to Python Path
sys.path.append("/Users/marhamil/Documents/Spark/spark-2.0.0-bin-hadoop2.7/python/")

from pyspark import SparkConf
from pyspark import SparkContext
import tensorflow as tf
import tensorframes as tfs
from pyspark.sql import Row



sc = SparkContext('local', 'pyspark')

file = "C:\Users\marhamil\Documents\Data\\training-monolingual-europarl\small_data_10000"
text_file = sc.textFile(file)
counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)
print counts.collect()



data = [Row(x=float(x)) for x in range(10)]
df = sc.createDataFrame(data)
with tf.Graph().as_default() as g:
    # The TensorFlow placeholder that corresponds to column 'x'.
    # The shape of the placeholder is automatically inferred from the DataFrame.
    x = tfs.block(df, "x")
    # The output that adds 3 to x
    z = tf.add(x, 3, name='z')
    # The resulting dataframe
    df2 = tfs.map_blocks(z, df)

# The transform is lazy as for most DataFrame operations. This will trigger it:
df2.collect()
