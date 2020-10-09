import tensorflow.compat.v1 as tf
import os 
import shutil
import csv
import pandas as pd
import IPython

print("1")

tf.get_logger().setLevel('ERROR')

from tapas.utils import tf_example_utils
from tapas.protos import interaction_pb2
from tapas.utils import number_annotation_utils
from tapas.scripts import prediction_utils

print("2")

os.makedirs('results/sqa/tf_examples', exist_ok=True)
os.makedirs('results/sqa/model', exist_ok=True)
with open('results/sqa/model/checkpoint', 'w') as f:
  f.write('model_checkpoint_path: "model.ckpt-0"')
for suffix in ['.data-00000-of-00001', '.index', '.meta']:
  shutil.copyfile(f'tapas_sqa_base/model.ckpt{suffix}', f'results/sqa/model/model.ckpt-0{suffix}')


print("3")

max_seq_length = 512
vocab_file = "tapas_sqa_base/vocab.txt"
config = tf_example_utils.ClassifierConversionConfig(
    vocab_file=vocab_file,
    max_seq_length=max_seq_length,
    max_column_id=max_seq_length,
    max_row_id=max_seq_length,
    strip_column_names=False,
    add_aggregation_candidates=False,
)
converter = tf_example_utils.ToClassifierTensorflowExample(config)

print("4")

def convert_interactions_to_examples(tables_and_queries):
  """Calls Tapas converter to convert interaction to example."""
  for idx, (table, queries) in enumerate(tables_and_queries):
    interaction = interaction_pb2.Interaction()
    for position, query in enumerate(queries):
      question = interaction.questions.add()
      question.original_text = query
      question.id = f"{idx}-0_{position}"
    for header in table[0]:
      interaction.table.columns.add().text = header
    for line in table[1:]:
      row = interaction.table.rows.add()
      for cell in line:
        row.cells.add().text = cell
    number_annotation_utils.add_numeric_values(interaction)
    for i in range(len(interaction.questions)):
      try:
        yield converter.convert(interaction, i)
      except ValueError as e:
        print(f"Can't convert interaction: {interaction.id} error: {e}")
        
def write_tf_example(filename, examples):
  with tf.io.TFRecordWriter(filename) as writer:
    for example in examples:
      writer.write(example.SerializeToString())

def predict(table_data, queries):
  print("5")	
  table = [list(map(lambda s: s.strip(), row.split("|"))) 
           for row in table_data.split("\n") if row.strip()]
  examples = convert_interactions_to_examples([(table, queries)])
  write_tf_example("results/sqa/tf_examples/test.tfrecord", examples)
  write_tf_example("results/sqa/tf_examples/random-split-1-dev.tfrecord", [])
  print("6")
  os.system(''' python tapas/tapas/run_task_main.py \
    --task="SQA" \
    --output_dir="results" \
    --noloop_predict \
    --test_batch_size=3 \
    --tapas_verbosity="ERROR" \
    --compression_type= \
    --init_checkpoint="tapas_sqa_base/model.ckpt" \
    --bert_config_file="tapas_sqa_base/bert_config.json" \
    --mode="predict" 2> error''')



  results_path = "results/sqa/model/test_sequence.tsv"
  all_coordinates = []
  df = pd.DataFrame(table[1:], columns=table[0])
  #display(IPython.display.HTML(df.to_html(index=False)))
  print("7")
  with open(results_path) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
      coordinates = prediction_utils.parse_coordinates(row["answer_coordinates"])
      all_coordinates.append(coordinates)
      answers = ', '.join([table[row + 1][col] for row, col in coordinates])
      position = int(row['position'])
      print(">", queries[position])
      print(answers)
  return all_coordinates



# Example nu-1000-0
result = predict("""
Pos | No | Driver               | Team                           | Laps | Time/Retired | Grid | Points
1   | 32 | Patrick Carpentier   | Team Player's                  | 87   | 1:48:11.023  | 1    | 22    
2   | 1  | Bruno Junqueira      | Newman/Haas Racing             | 87   | +0.8 secs    | 2    | 17    
3   | 3  | Paul Tracy           | Team Player's                  | 87   | +28.6 secs   | 3    | 14
4   | 9  | Michel Jourdain, Jr. | Team Rahal                     | 87   | +40.8 secs   | 13   | 12
5   | 34 | Mario Haberfeld      | Mi-Jack Conquest Racing        | 87   | +42.1 secs   | 6    | 10
6   | 20 | Oriol Servia         | Patrick Racing                 | 87   | +1:00.2      | 10   | 8 
7   | 51 | Adrian Fernandez     | Fernandez Racing               | 87   | +1:01.4      | 5    | 6
8   | 12 | Jimmy Vasser         | American Spirit Team Johansson | 87   | +1:01.8      | 8    | 5
9   | 7  | Tiago Monteiro       | Fittipaldi-Dingman Racing      | 86   | + 1 Lap      | 15   | 4
10  | 55 | Mario Dominguez      | Herdez Competition             | 86   | + 1 Lap      | 11   | 3
11  | 27 | Bryan Herta          | PK Racing                      | 86   | + 1 Lap      | 12   | 2
12  | 31 | Ryan Hunter-Reay     | American Spirit Team Johansson | 86   | + 1 Lap      | 17   | 1
13  | 19 | Joel Camathias       | Dale Coyne Racing              | 85   | + 2 Laps     | 18   | 0
14  | 33 | Alex Tagliani        | Rocketsports Racing            | 85   | + 2 Laps     | 14   | 0
15  | 4  | Roberto Moreno       | Herdez Competition             | 85   | + 2 Laps     | 9    | 0
16  | 11 | Geoff Boss           | Dale Coyne Racing              | 83   | Mechanical   | 19   | 0
17  | 2  | Sebastien Bourdais   | Newman/Haas Racing             | 77   | Mechanical   | 4    | 0
18  | 15 | Darren Manning       | Walker Racing                  | 12   | Mechanical   | 7    | 0
19  | 5  | Rodolfo Lavin        | Walker Racing                  | 10   | Mechanical   | 16   | 0
""", ["what were the team names?",
      "of these, which points did Mario Haberfeld and Oriol Servia score?",
      "who scored 2?"])
